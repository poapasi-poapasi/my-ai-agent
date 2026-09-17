import os
import requests
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

st.set_page_config(page_title="AI Agent", page_icon="🤖")
st.title("🤖 My AI Agent")

st.sidebar.header("Agent Capabilities")
st.sidebar.markdown("- 💬 General Chat & Q&A\n- 🌤️ Real-time Weather Updates")

api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("Missing GEMINI_API_KEY secret.")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash", 
    google_api_key=api_key,
    system_instruction=(
        "You are a helpful AI assistant. Always reply using conversational, "
        "well-formatted plain text paragraphs. Never output raw tool definitions or JSON."
    )
)

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

if "messages" not in st.session_state:
    st.session_state.messages = []

# Only display messages that contain actual text string content
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
        # Hide raw tool calls or non-string list objects
        if not msg.tool_calls and not msg.content.startswith("["):
            st.chat_message("assistant").write(msg.content)

user_input = st.chat_input("Ask me anything...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    with st.spinner("Thinking..."):
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
                else:
                    st.chat_message("assistant").write("I can help answer general questions and fetch real-time weather information.")
        except Exception as e:
            st.warning("Request limit reached. Please wait a minute and try again!")
