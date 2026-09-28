"""Minimal Python API example for the LangGraph agent."""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.agent.health_agent import AgentSettings, HealthAgent, public_result


load_dotenv()

settings = AgentSettings(
    profile_file=Path("data/user/profile.json"),
    data_root=Path("data"),
    database_url=os.getenv("DATABASE_URL"),
    embedding_model_id=os.getenv("RAG_EMBEDDING_MODEL_ID"),
    embedding_dimensions=int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "1024")),
    embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3"),
    embedding_backend=os.getenv("RAG_EMBEDDING_BACKEND", "flag_embedding"),
    chat_model=os.getenv("RAG_CHAT_MODEL", "gpt-5-mini"),
)

agent = HealthAgent(settings)
state = agent.invoke("我42岁，女，有高血压。最近总是夜间出汗怎么办。")
result = public_result(state, settings.profile_file)

print(result["intent"])
print(result["keywords"])
print(result.get("rag", {}).get("ranked", []))
print(result["original_source_read"])
print(result["final_answer"])
