import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import os
from pathlib import Path
from dotenv import load_dotenv


script_dir = Path(__file__).resolve().parent
env_path = script_dir / ".env"

load_dotenv(env_path)

# Quick sanity check that the API key actually loaded before calls to the model
# print(f"OPENAI_API_KEY found: {bool(os.getenv('OPENAI_API_KEY'))}")
print(f"OPENAI_API_KEY found: {bool(os.getenv('CLAUDE_API_KEY'))}")


# -----------------------------------------------
# 1. MODELS - The Reasoning Engine
# -----------------------------------------------
# The Model is the LLM itself - the part that actually thinks (reasoning)
# init_chat_model gives ONE interface to every provider to switch from
# OpenAI to Anthropic or Google, you change only the string below
from langchain.chat_models import init_chat_model

from langchain.chat_models import init_chat_model

model = init_chat_model("ollama:llama3.2")
# model = init_chat_model("openai:gpt-5.5")
# model = init_chat_model("anthropic:claude-3-3-sonnet-latest")
# model = init_chat_model("google_genai:gemini-2.0-flash")


# The simplest possible use: send text in get an answer back
# invoke() is the universal "run it" method across LangChain

response = model.invoke("What is LangChain in one sentence?")
print("=== Model Response ===")
print(response.content)  # content = just the text of the reply
print()
# -----------------------------------------------
# 2. PROMPT TEMPLATES - Steering the Model
# -----------------------------------------------
# A Prompt Template is resuable sentence with blanks ({placeholders})
# you fill in later - write the wording once and reuse it many times
# There are two types of Templates as PromptTemplates and ChatPromptTemplates

# PromptTemplates = a single plain-text string with blanks
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

simple_template = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} to a complete beginner in 2-3 sentences.",
)

formatted = simple_template.format(topic="AI agents")
print("=== Formatted Prompt ===")
print(formatted)
print()

# ChatPromptTemplate = build from ROLES (system/human), which is how chat
# models expect their input. "system" sets behavior while "human" is the user turn

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful coding tutor. Keep answers short and clear"),
        ("human", "Explain {concept} with a simple Python example"),
    ]
)

# .format_messages() fills the blanks and returns a LIST of messages

messages = chat_template.format_messages(concept="list comprehension")
# {concept} in the template is the empty blank, and concept="list comprehension" is you handing it the word that goes in the blank
print("=== Chat Messages ===")
for msg in messages:
    print(f" [{msg.type}]:{msg.content[:80]}")
print()

# -----------------------------------------------
# 3. CHAINS - Connecting the Pieces with LCEL
# -----------------------------------------------
# LCEL = LangChain Expression Language = Connecting LangChain components together to build AI workflows.
# The pipe | glues components into a pipeline. Each step's output flows into the next, left to right: prompt | model (LLM) | parser
from langchain_core.output_parsers import StrOutputParser

# prompt fills the blanks -> model answers -> parser pulls out clean text.
# StrOutputParser juest extracts the plain string from the model's reply
# object, so you don't have to write .content everytime
chain = chat_template | model | StrOutputParser()

# Run the whole pipeline with a single .invoke()
result = chain.invoke({"concept": "list comprehension"})
print("=== Chain Ouput ===")
print(result)
print()


# -----------------------------------------------
# 4. MEMORY - Giving the model context
# -----------------------------------------------
# Models are stateless - They forget everything between calls
# "Memory" is simply as storing past messeges and feeding them back in
# (Modern LangChain uses ChatMessageHistory; the old
# ConversationBufferMemory is deprecated and out of the core package)

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# A simple in-memory store that holds the conversation
memory = InMemoryChatMessageHistory()

# hand-build a short conversation so we have something to "remember"
memory.add_message(HumanMessage(content="My name is Nuwan and I am learning LangChain"))
memory.add_message(
    AIMessage(content="Nice to meet you, Nuwan! LangChain is a great choice")
)
memory.add_message(HumanMessage(content="What tools should I learn first?"))
memory.add_message(
    AIMessage(
        content="Start with PromptTemplates and simple chains, then move to tools and agents"
    )
)

print("=== Memory Contents ===")
for msg in memory.messages:
    print(f"[{msg.type}]:{msg.content[:80]}...")
print()


# The "placeholder" slot is where the stored messages get injected into the
# prompt, so the model can SEE the earlier conversation

chat_with_memory = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful tutor. Use the conversation history to personalized your response.",
        ),
        (
            "placeholder",
            "{history}",  # past messages get dropped in here
        ),
        ("human", "{question}"),
    ]
)

chain_with_memory = chat_with_memory | model | StrOutputParser()

# We pass the stored history in alongside the new question
# Watch the model correctly recall the name "Nuwan" - that's "memory"

result = chain_with_memory.invoke(
    {"history": memory.messages, "question": "What was my name again?"}
)


print("=== Memory-Aware Response ===")
print(result)
print()
