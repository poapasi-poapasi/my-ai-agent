import os
import requests
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

st.title("🤖 My Permanent AI Agent")

st.sidebar.header("Agent Capabilities")
st.sidebar.markdown("- 💬 General Q&A & Writing\n- 🌤️ Real-time Weather Updates")

api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("Missing GEMINI_API_KEY secret.")
    st.stop()

# System instruction forcing strictly conversational narrative prose
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash", 
    google_api_key=api_key,
    system_instruction=(
        "You are a helpful, professional, and friendly AI assistant. "
        "When describing your capabilities, speak naturally in full narrative sentences. "
        "Never output raw code, JSON objects, internal tool definitions, or function signatures."
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

# Chat Session Storage
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation (only show narrative content, filter out raw tool objects)
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        st.chat_message("assistant").write(msg.content)

user_input = st.chat_input("Ask me about the weather or general questions...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Process initial user prompt
    response = llm_with_tools.invoke(st.session_state.messages)
    st.session_state.messages.append(response)
    
    # If a tool execution was triggered, execute it silently and fetch the final text answer
    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = get_live_weather.invoke(tool_call["args"])
            st.session_state.messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
        # Fetch the model's final conversational translation of the result
        final_response = llm_with_tools.invoke(st.session_state.messages)
        st.session_state.messages.append(final_response)
        
        if final_response.content:
            st.chat_message("assistant").write(final_response.content)
        else:
            st.chat_message("assistant").write("I have retrieved the details for you.")
    else:
        # Only render if the AI generated actual text (avoids rendering raw tool call signatures)
        if response.content:
            st.chat_message("assistant").write(response.content)
        else:
            # Handle edge cases where the AI tries to answer with an empty message
            prompt_fix = HumanMessage(content="Please explain what you can do in clear, natural paragraphs without using raw code.")
            st.session_state.messages.append(prompt_fix)
            fallback = llm_with_tools.invoke(st.session_state.messages)
            st.session_state.messages.append(fallback)
            st.chat_message("assistant").write(fallback.content)
