from flask import Flask, request, render_template_string
from datetime import date
from duckduckgo_search import DDGS

# LangChain imports
from langchain_community.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_ollama import OllamaLLM

app = Flask(__name__)

TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
  <title>Local Chat - LangChain Agents</title>
  <style>
    body { font-family: Arial; max-width: 600px; margin: 2em auto; }
    .user, .bot { margin: 1em 0; }
    .user { color: blue; }
    .bot { color: green; }
  </style>
</head>
<body>
  <h2>Chat with Mistral + LangChain Agents</h2>
  {% for entry in history %}
    <div class="{{ entry.role }}">{{ entry.role.title() }}: {{ entry.content }}</div>
  {% endfor %}
  <form method="post">
    <input name="prompt" autofocus style="width: 80%;" />
    <button type="submit">Send</button>
  </form>
</body>
</html>
'''

history = []

# --- LangChain Tools ---
def get_today():
    return str(date.today())

def web_search(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query)
        summaries = [r['body'] for r in results][:3]
        return "\n\n".join(summaries) if summaries else "No results found."

tools = [
    Tool(name="Current Date", func=lambda _: get_today(), description="Use to get today's date."),
    Tool(name="Web Search", func=web_search, description="Use to search the web for current information.")
]

llm = OllamaLLM(model="mistral:instruct")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

@app.route('/', methods=['GET', 'POST'])
def chat():
    global history
    if request.method == 'POST':
        user_input = request.form['prompt']
        history.append({'role': 'user', 'content': user_input})

        try:
            answer = agent.run(user_input)
        except Exception as e:
            answer = f"[Error] {str(e)}"

        history.append({'role': 'assistant', 'content': answer})

    return render_template_string(TEMPLATE, history=history)

if __name__ == '__main__':
    app.run(debug=True)

