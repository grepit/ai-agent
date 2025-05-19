# agents.py
from langchain.llms import Ollama
from langchain.agents import Tool, initialize_agent, AgentType
from datetime import datetime
from langchain.chains import RetrievalQA
from file_loader import retriever


# Initialize the LLM
#llm = Ollama(model="mistral")
llm = Ollama(model="phi:latest")

file_qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Define different "characters"
def science_tool(input): return f"[Science Expert] Response to: {input}"
def history_tool(input): return f"[History Expert] Response to: {input}"


def file_tool(query: str):
    return file_qa.run(query)


def date_tool(_):
    return f"Today's date is: {datetime.now().strftime('%A, %B %d, %Y')}"


tools = [
    Tool(name="Science Expert", func=science_tool, description="Handles science questions."),
    Tool(name="History Expert", func=history_tool, description="Handles history questions."),
    Tool(name="Date Expert", func=date_tool, description="Tells today's date."),
    Tool(name="File Expert", func=file_tool, description="Answers questions about local documents in the folder")

]

# Initialize the agent
mcp_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

