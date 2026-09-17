import os
import requests
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

st.set_page_config(page_title="Supercharged AI Agent", page_icon="🤖")
st.title("🤖 Supercharged AI Agent")

# Sidebar indicating updated capabilities
st.sidebar.header("Agent Capabilities")
st.sidebar.markdown(
    "- 💬 General Q&A & Writing\n"
    "- 🌤️ Live Weather Data\n"
    "- 📚 Wikipedia Knowledge Search\n"
    "- 📈 Crypto & Market Prices"
)

api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("Missing GEMINI_API_KEY secret.")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash", 
    google_api_key=api_key,
    system_instruction=(
        "You are a helpful, professional, and friendly AI assistant equipped with specialized tools. "
        "Always respond in natural, clear, conversational prose. "
        "Never output raw tool definitions, JSON objects, or code snippets."
    )
)

# Tool 1: Live Weather
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

# Tool 2: Wikipedia Search
@tool
def search_wikipedia(query: str) -> str:
    """Searches Wikipedia summaries for factual historical, biographical, or general knowledge questions."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    res = requests.get(url, headers={"User-Agent": "MyStreamlitAgent/1.0"}, timeout=10)
    if res.status_code == 200:
        data = res.json()
        return data.get("extract", "No relevant summary found on Wikipedia.")
    return f"Could not find a Wikipedia page for '{query}'."

# Tool 3: Crypto Price Tracker
@tool
def get_crypto_price(coin_id: str) -> str:
    """Fetches current USD prices for cryptocurrencies (e.g., bitcoin, ethereum, solana)."""
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id.lower()}&vs_currencies=usd"
    res = requests.get(url, timeout=10).json()
    if coin_id.lower() in res:
        price = res[coin_id.lower()]["usd"]
        return f"The current price of {coin_id.title()} is ${price:,.2f} USD."
    return f"Could not retrieve price for '{coin_id}'. Ensure you use full names like 'bitcoin' or 'ethereum'."

# Register all tools with the agent
tools = [get_live_weather, search_wikipedia, get_crypto_price]
llm_with_tools = llm.bind_tools(tools)

# Session Memory Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior conversation
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
        if not msg.tool_calls and not msg.content.startswith("["):
            st.chat_message("assistant").write(msg.content)

user_input = st.chat_input("Ask about weather, crypto prices, or search Wikipedia...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    with st.spinner("Thinking..."):
        try:
            response = llm_with_tools.invoke(st.session_state.messages)
            st.session_state.messages.append(response)
            
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    # Match tool names dynamically
                    tool_name = tool_call["name"]
                    args = tool_call["args"]
                    
                    if tool_name == "get_live_weather":
                        result = get_live_weather.invoke(args)
                    elif tool_name == "search_wikipedia":
                        result = search_wikipedia.invoke(args)
                    elif tool_name == "get_crypto_price":
                        result = get_crypto_price.invoke(args)
                    else:
                        result = "Tool execution failed."
                        
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
            st.warning("Request limit reached or temporary network issue. Please wait 60 seconds and try again!")
