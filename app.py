import os
import io
import requests
import streamlit as st
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

st.set_page_config(page_title="Private Executive AI Agent", page_icon="🔐")

# --- 1. SECURE PASSWORD PROTECTION ---
APP_PASSWORD = st.secrets.get("APP_PASSWORD") or "mysecret123"  # Change this default password!

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Restricted Access")
    password_input = st.text_input("Enter Access Password:", type="password")
    if st.button("Login"):
        if password_input == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# --- 2. MAIN SECURE APP INTERFACE ---
st.title("🤖 Private Executive AI Agent")

# Sidebar setup
st.sidebar.header("Security & Controls")
if st.sidebar.button("🔒 Logout & Purge Session"):
    st.session_state.clear()
    st.rerun()

st.sidebar.header("Capabilities")
st.sidebar.markdown(
    "- 🔐 Password Protected\n"
    "- 📄 Confidential PDF Analysis\n"
    "- 🌤️ Live Weather Data\n"
    "- 💬 General Q&A & Writing"
)

# Read API key securely
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("Missing GEMINI_API_KEY secret in Streamlit Cloud settings.")
    st.stop()

# Initialize Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash", 
    google_api_key=api_key,
    system_instruction=(
        "You are a secure, professional, and executive AI assistant. "
        "Analyze documents carefully when provided. Always respond in clear, well-structured narrative prose. "
        "Never output raw code blocks, JSON objects, or tool signatures."
    )
)

# --- 3. IN-MEMORY PDF UPLOADER ---
st.sidebar.subheader("📄 Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload Company PDF", type=["pdf"])

pdf_context = ""
if uploaded_file is not None:
    try:
        # Read file entirely in RAM memory without writing to disk
        pdf_bytes = io.BytesIO(uploaded_file.read())
        reader = PdfReader(pdf_bytes)
        extracted_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        
        # Limit context size to avoid token limits
        pdf_context = extracted_text[:15000] 
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error("Failed to parse PDF.")

# --- 4. WEATHER TOOL ---
@tool
def get_live_weather(city: str) -> str:
    """Fetches real-time temperature and weather conditions for a given city."""
    headers = {"User-Agent": "MyStreamlitAgent/1.0"}
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    geo_res = requests.get(geo_url, headers=headers, timeout=10).json()
    
    if not geo_res.get("results"):
        return f"Could not find coordinates for {city}."
        
    lat = geo_res["results"][0]["latitude"]
    lon = geo_res["results"][0]["longitude"]
    city_name = geo_res["results"][0]["name"]
    
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    w_res = requests.get(weather_url, headers=headers, timeout=10).json()
    
    current = w_res.get("current_weather", {})
    temp_c = current.get("temperature", "N/A")
    wind = current.get("windspeed", "N/A")
    
    return f"Current weather in {city_name}: {temp_c}°C with wind speed of {wind} km/h."

tools = [get_live_weather]
llm_with_tools = llm.bind_tools(tools)

# --- 5. CHAT EXECUTION LOOP ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
        if not msg.tool_calls and not msg.content.startswith("["):
            st.chat_message("assistant").write(msg.content)

user_input = st.chat_input("Ask a question or request PDF analysis...")

if user_input:
    # Inject document context into user query dynamically if PDF is present
    full_prompt = user_input
    if pdf_context:
        full_prompt = f"[DOCUMENT CONTEXT]:\n{pdf_context}\n\n[USER QUESTION]:\n{user_input}"

    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=full_prompt))
    
    with st.spinner("Analyzing..."):
        try:
            response = llm_with_tools.invoke(st.session_state.messages)
            st.session_state.messages.append(response)
            
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    result = get_live_weather.invoke(tool_call["args"])
                    st.session_state.messages.append(
                        ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                    )
                final_response = llm_with_tools.invoke(st.session_state.messages)
                st.session_state.messages.append(final_response)
                
                if isinstance(final_response.content, str):
                    st.chat_message("assistant").write(final_response.content)
            else:
                if isinstance(response.content, str) and response.content.strip():
                    st.chat_message("assistant").write(response.content)
        except Exception:
            st.warning("Rate limit reached. Please wait 60 seconds and try again!")
