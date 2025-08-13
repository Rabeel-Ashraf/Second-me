# second_me.py
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools import Tool
import textwrap
import os

# --- 🔐 SET YOUR API KEYS ---
# Get free keys: 
# OpenAI: https://platform.openai.com/api-keys
# Tavily (search): https://app.tavily.com/home
os.environ["OPENAI_API_KEY"] = "sk-..."  # ← PUT YOUR KEY HERE
os.environ["TAVILY_API_KEY"] = "tvly-..."  # ← AND HERE

# --- 🧬 YOUR AI DNA (from ai-dna.md) ---
PAST_REPLIES = [
    "nah I'm good thanks",
    "bro that's interesting, tell me more",
    "I'm busy rn but let's connect next week",
    "not for me, but I appreciate you reaching out",
    "let's collab after I finish this project",
    "send me the details and I'll check"
]

# --- 🧠 MEMORY: Turn your past replies into retrievable knowledge ---
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(PAST_REPLIES, embeddings)
retriever = vectorstore.as_retriever()

def search_past_replies(query):
    """Tool: Find how 'you' replied to similar messages before"""
    results = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in results])

memory_tool = Tool(
    name="Past Replies Memory",
    func=search_past_replies,
    description="Search how I replied to similar messages in the past"
)

# --- 🔍 SEARCH TOOL (for context) ---
from langchain_community.utilities import TavilySearchAPIWrapper
search_tool = TavilySearchAPIWrapper()
tavily_tool = Tool(
    name="Web Search",
    func=search_tool.run,
    description="Use to search the web for current info"
)

# --- 🤖 THE AGENT: Your Digital Twin ---
second_me = Agent(
    role="Digital Twin",
    goal="Respond exactly like Rabeel — same tone, same judgment, same values",
    backstory=textwrap.dedent("""
        I am Rabeel's AI twin. I know his tone (casual, direct, no fluff),
        his values (honest, respectful, no spam), and his style.
        I use his past replies to respond authentically.
        I never oversell. I never lie. I never waste time.
        If unsure, I ask for approval.
    """),
    tools=[memory_tool, tavily_tool],
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7),
    verbose=True,
    allow_delegation=False
)

# --- 📬 TASK: Simulate a real DM ---
task = Task(
    description="Someone just DMed you: 'Hey can we collab on an AI project?'",
    expected_output="A short, natural reply in Rabeel's voice — like he'd actually say it."
)

# --- 🚀 RUN THE CREW ---
crew = Crew(
    agents=[second_me],
    tasks=[task],
    verbose=2
)

print("🤖 SECOND ME IS THINKING...\n")
result = crew.kickoff()

print("\n" + "="*50)
print("🎯 SECOND ME SAYS:")
print("-" * 50)
print(result)
print("="*50)
