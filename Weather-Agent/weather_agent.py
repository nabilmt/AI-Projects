from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import requests
import os
from langchain.tools import tool
import gradio as gr
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm=ChatOpenAI(model='gpt-4.1-mini')


@tool
def check_weather(location: str) -> str:
    """Return the current weather stats for the specified location."""
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Weather service error: {str(e)}"

    data = response.json()
    
    if data.get("cod") != 200:
        return f"Could not fetch weather for {location}. Please check the location name."
    
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    description = data["weather"][0]["description"]
    wind_speed = data["wind"]["speed"]
    
    return (
        f"Weather in {location}:\n"
        f"  Temperature : {temp}°C\n"
        f"  Feels like  : {feels_like}°C\n"
        f"  Humidity    : {humidity}%\n"
        f"  Condition   : {description}\n"
        f"  Wind speed  : {wind_speed} m/s"
    )

memory = MemorySaver()

# Create the agent
chat = create_agent(
    name='agent_101',
    model=llm,
    tools=[check_weather],
    checkpointer=memory,
    system_prompt="You are a helpful weather assistant. When asked about weather, always use the check_weather tool."
)

    
# Gradio function
def ask_agent(user_message, history):
    result = chat.invoke(
    {
        "messages": [{"role": "user","content": user_message}]
    },
    config={
        "configurable": {
            "thread_id": "weather-chat"
        }
    }
)
    return result['messages'][-1].content

# Gradio UI
demo = gr.ChatInterface(
    fn=ask_agent,
    title="Weather Agent",
    description="Ask me about the weather anywhere in the world!",
    examples=[
        "What's the weather in London?",
        "Give me weather stats for Tokyo",
        "How's the weather in Kerala today?"
    ]
)

demo.launch(inbrowser=True)