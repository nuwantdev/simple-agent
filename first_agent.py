"""
LangChain for AI Agents
=================================================
This program demonstrates:
    - How an AI Agent uses the ReAct pattern (Reason -> Act -> Observe)
    - How to build an agent using LangChain's create_agent()

Pre-requisites:
    - pip install langchain langchain-openai langgraph python-dotenv
    - Alternatively install ollama3.2 modal locally and use by installing langchain-ollama module

Setup:
    - Create a .env file with OPENAI_API_KEY=your-api-key

"""

import os
from math import sqrt
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_agent

# -------------------------------------------------------------
# STEP 0: Load environment variables
# -------------------------------------------------------------
# We store sensitive information like API keys in a .env file.
# This keeps secrets out of the code (best practice)
parent_path = Path(__file__).resolve().parent
env_path = parent_path / ".env"

load_dotenv(env_path)
# Quick sanity check that the API key actually loaded before calls to the model
openai_api_key = os.getenv("OPENAI_API_KEY")
if (
    openai_api_key and openai_api_key.strip()
):  # check for key is in the .env and key is with a value
    print("OpenAI API key found")
else:
    raise ValueError("OPENAI_API_KEY is missing or empty.")

# -------------------------------------------------------------
# STEP 1: Initialize the Model (the 'brain')
# -------------------------------------------------------------
# This is the LLM that will:
# - Understand the question
# - Decide which tool to use
# - Generate the final answer

# model = init_chat_model("ollama:llama3.2")
model = init_chat_model("openai:gpt-5.5")

# -------------------------------------------------------------
# STEP 2: Define your tools (the 'hands')
# -------------------------------------------------------------
# Tools allow the agent to DO things instead of just responding.
# Each tool must have:
# 1. A clear name
# 2. A descriptive docstring (VERY important!)
# 3. Type hints (so LLM knows expected inputs and ouput)


@tool
def add(a: float, b: float) -> float:
    """
    Add two numbers together.
    The agent will use this when it detects an addition problem
    """
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers together.
    Use for multiplication tasks
    """
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """
    Divide the first number by the second.
    Includes error handling for division by zero
    """
    if b == 0:
        return "Error: Cannot divide by 0"

    return a / b


@tool
def square_root(number: float) -> float:
    """
    Calculate the square root of a number
    Include error handling for negative inputs
    """
    if number < 0:
        return "Error: Cannot take square root of a negative number"
    return sqrt(number)


# Combine all tools into a list
tools = [add, multiply, divide, square_root]

# Print available tools (helps to see which tool that agent has access to)
print("=== Available Tools ===")
for t in tools:
    print(f"- {t.name}: {t.description}")
print()

# -------------------------------------------------------------
# STEP 3:  Create the Agent (the 'loop')
# -------------------------------------------------------------
# create_agent automatically builds the ReAct loop:
# 1. Reason - LLM decides what to do
# 2. Act - calls a tool (if needed)
# 3. Observe - gets the result
# 4. Repeat until done
#
# We do not need to write manually write the loop - the freamework does it


agent = create_agent(model=model, tools=tools)

# -------------------------------------------------------------
# STEP 4:  Run the agent
# -------------------------------------------------------------
# This function sends a question to the agent and prints:
# - The final answer
# - (optional) the internal execution trace


def run_agent(question: str):
    """Run the agenr and print a clean, beginner-friendly execution trace"""
    print(f"\n User: {question}")

    result = agent.invoke({"messages": [("user", question)]})
    # Normally result["messages"] includes the following message types
    # Message class -> HumanMessage, msg.type -> "human", User Message
    # Message class -> AIMessage, msg.type -> "ai", AI response
    # Message class -> ToolMessage, msg.type -> "tool", Tool result
    step = 1
    for msg in result["messages"]:
        # print("Class:", type(msg))
        # print("Type:", msg.type)
        # print("Content:", msg.content)
        # print("Additional kwargs:", msg.additional_kwargs)
        # print("-" * 60)

        # 1. Human message =  Original user question
        if msg.type == "human":
            print(f"{step}. User asked:")
            print(f"{msg.content}")
            step += 1

        # 2. AI message with tool_calls = agent decided to use a tool
        elif msg.type == "ai" and getattr(msg, "tool_calls"):
            for tool_call in msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                print(f"{step}. Agent decision:")
                print(f"I need to use the tool: {tool_name}")
                print(f"Tool input: {tool_args}")
                step += 1
        # 3. Tool message = results return by the tool
        elif msg.type == "tool":
            print(f"{step}. Tool observation:")
            print(f"Tool returned: {msg.content}")
            step += 1
        # 4. Final AI message = final response to the user
        elif msg.type == "ai" and msg.content:
            print(f"{step}. Final answer:")
            print(f"{msg.content}")
            step += 1

    print("-" * 60)


# -------------------------------------------------------------
# TEST CASES:  Watch the Agent in Action
# -------------------------------------------------------------
# 1. Simple CASES -> Single tool call
# Agent should use directly use "add"
run_agent("What is 42 + 58?")

# 2. Medium Complexity -> Multiple Steps
# Agent must:
#   multiply -> then divide
run_agent("What is 15 multiply by 8, then divided by 3?")

# 3. Complex reasoning -> Planning required
# Agent must:
#   step 1: Calculate area
#   step 2: Calculate square root

run_agent(
    "I have a rectangle with width 12 and height 7. What is its area and what is the square root of that area?"
)

# 4. Edge case -> error hanling
# Agent should handle divide-by-zero gracefully
run_agent("What is 100 divided by 0?")
