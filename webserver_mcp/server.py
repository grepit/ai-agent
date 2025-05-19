# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from agents import mcp_agent

app = FastAPI()

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
def handle_chat(data: ChatInput):
    reply = mcp_agent.run(data.message)
    return {"response": reply}

