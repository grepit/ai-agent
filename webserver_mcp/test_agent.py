from agents import mcp_agent

# Test the agent with different types of queries
queries = [
    "What is the capital of France?",
    "What's today's date?",
    "What can you tell me about the files in the files_lib directory?"
]

for query in queries:
    print(f"\nQuery: {query}")
    print("Response:", mcp_agent.run(query)) 