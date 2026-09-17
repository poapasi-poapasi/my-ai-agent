import os
import io
import requests
import streamlit as st
import matplotlib.pyplot as plt
from pypdf import PdfReader
from docx import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from duckduckgo_search import DDGS
import yfinance as yf

st.set_page_config(page_title="Ultimate AI Assistant", page_icon="⚡", layout="wide")

# --- 1. PASSWORD PROTECTION ---
APP_PASSWORD = st.secrets.get("APP_PASSWORD") or "mysecret123"

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

# --- 2. SIDEBAR NAVIGATION & FILE UPLOADER ---
st.sidebar.title("⚡ Agent Hub")
if st.sidebar.button("🔒 Logout & Purge Memory"):
    st.session_state.clear()
    st.rerun()

st.sidebar.header("Loaded Tools")
st.sidebar.markdown(
    "- 🌐 Live Web & News Search\n"
    "- 📈 Stocks & Crypto Tracker\n"
    "- 🌤️ Real-Time Weather\n"
    "- 📄 PDF & Word Document Reader\n"
    "- 🧮 Python Code Exec / Math\n"
    "- 📊 Chart & Diagram Generator"
)

# File Uploader
st.sidebar.subheader("📄 Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload PDF or Word Doc", type=["pdf", "docx"])

doc_context = ""
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".pdf"):
            pdf_bytes = io.BytesIO(uploaded_file.read())
            reader = PdfReader(pdf_bytes)
            extracted_text = "".join([page.extract_text() or "" for page in reader.pages])
            doc_context = extracted_text[:15000]
            st.sidebar.success(f"Loaded PDF: {uploaded_file.name}")
        elif uploaded_file.name.endswith(".docx"):
            doc_bytes = io.BytesIO(uploaded_file.read())
            doc = Document(doc_bytes)
            extracted_text = "\n".join([p.text for p in doc.paragraphs])
            doc_context = extracted_text[:15000]
            st.sidebar.success(f"Loaded Word Doc: {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error("Failed to parse document.")

# API Key Validation
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("Missing GEMINI_API_KEY secret.")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    system_instruction=(
        "You are an executive multi-tool AI assistant. "
        "Always respond in clean, formal, narrative prose. "
        "Never display raw JSON, function signatures, or raw tool blocks to the user."
    )
)

# --- 3. DEFINING AGENT TOOLS ---

@tool
def live_web_search(query: str) -> str:
    """Searches the web for up-to-date facts, current events, and general information."""
    try:
        results = list(DDGS().text(query, max_results=3))
        if not results:
            return "No web results found."
        summary = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
        return summary
    except Exception as e:
        return f"Web search failed: {e}"

@tool
def get_stock_or_crypto_price(ticker: str) -> str:
    """Fetches real-time stock or crypto prices using ticker symbols (e.g., AAPL, TSLA, BTC-USD, ETH-USD)."""
    try:
        data = yf.Ticker(ticker)
        fast_info = data.fast_info
        price = fast_info["lastPrice"]
        currency = fast_info["currency"]
        return f"The current price for {ticker.upper()} is {price:,.2f} {currency}."
    except Exception:
        return f"Could not retrieve stock/crypto price for ticker '{ticker}'."

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

@tool
def python_math_calculator(expression: str) -> str:
    """Evaluates mathematical or numerical expressions using Python code execution."""
    try:
        allowed = "0123456789+-*/(). "
        if all(c in allowed for c in expression):
            val = eval(expression)
            return f"Mathematical Result: {val}"
        return "Invalid characters in mathematical expression."
    except Exception as e:
        return f"Calculation error: {e}"

tools = [live_web_search, get_stock_or_crypto_price, get_live_weather, python_math_calculator]
llm_with_tools = llm.bind_tools(tools)

# --- 4. CHAT LOOP & INTERFACE ---
st.title("🤖 Ultimate Executive AI Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render prior messages cleanly
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
        if not msg.tool_calls and not msg.content.startswith("["):
            st.chat_message("assistant").write(msg.content)

user_input = st.chat_input("Ask a question, check prices, search the web, or analyze a document...")

if user_input:
    full_prompt = user_input
    if doc_context:
        full_prompt = f"[ATTACHED DOCUMENT CONTEXT]:\n{doc_context}\n\n[USER QUESTION]:\n{user_input}"

    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=full_prompt))
    
    with st.spinner("Processing request..."):
        try:
            response = llm_with_tools.invoke(st.session_state.messages)
            st.session_state.messages.append(response)
            
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    t_name = tool_call["name"]
                    t_args = tool_call["args"]
                    
                    if t_name == "live_web_search":
                        res = live_web_search.invoke(t_args)
                    elif t_name == "get_stock_or_crypto_price":
                        res = get_stock_or_crypto_price.invoke(t_args)
                    elif t_name == "get_live_weather":
                        res = get_live_weather.invoke(t_args)
                    elif t_name == "python_math_calculator":
                        res = python_math_calculator.invoke(t_args)
                    else:
                        res = "Tool failed."

                    st.session_state.messages.append(
                        ToolMessage(content=str(res), tool_call_id=tool_call["id"])
                    )
                
                final_response = llm_with_tools.invoke(st.session_state.messages)
                st.session_state.messages.append(final_response)
                
                if isinstance(final_response.content, str):
                    st.chat_message("assistant").write(final_response.content)
            else:
                if isinstance(response.content, str) and response.content.strip():
                    st.chat_message("assistant").write(response.content)
        except Exception as e:
            st.warning("Rate limit hit or request timed out. Please wait 60 seconds and try again!")
