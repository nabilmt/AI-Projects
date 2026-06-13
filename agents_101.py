from langchain.tools import tool
from typing import List,Dict,Any,Optional
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent



load_dotenv()

openai=ChatOpenAI(model='gpt-4.1-nano')

@tool
def add_numbers(text : str) -> dict:
    """Extracts all numbers from the given text and returns the sum."""
    numbers=[int(x) for x in re.findall(r'\d+',text)]
    result=sum(numbers)
    return {'sum':result}


@tool
def check_weather(location: str) -> str:
    """Return the weather forecast for the specified location."""
    return f"The temperature in {location} is"




# Create the agent
chat = create_agent(
    name='agent_101',
    model=openai,
    tools=[add_numbers,check_weather],
    system_prompt="You are a helpful assistant"
)

# result=chat.invoke({'messages':[{'role':'user','content':'the gdp is 20000$, for india its 670000$ and for us 1000000$. can you add all this gdp'}]})
result=chat.invoke({'messages':[{'role':'user','content':'look up the tempereature in celcius at kerala ambalavayal'}]})

print(result['messages'][-1].content)
