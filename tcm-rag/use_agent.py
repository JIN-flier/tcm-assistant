"""Launch the interactive terminal UI: python use_agent.py."""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.agent.health_agent import AgentSettings, HealthAgent, public_result
from src.agent.tui import HealthAgentApp


def main() -> None:
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    settings = AgentSettings(
        profile_file=Path(os.getenv("TCM_PROFILE_FILE", str(root / "data/user/profile.json"))),
        data_root=Path(os.getenv("TCM_DATA_ROOT", str(root / "data"))),
        database_url=os.getenv("DATABASE_URL"),
        embedding_model_id=os.getenv("RAG_EMBEDDING_MODEL_ID"),
        embedding_dimensions=int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "1024")),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_backend=os.getenv("RAG_EMBEDDING_BACKEND", "flag_embedding"),
        chat_model=os.getenv("RAG_CHAT_MODEL", "gpt-5-mini"),
    )
    agent = HealthAgent(settings)

    def ask(question: str) -> dict:
        state = agent.invoke(question)
        return public_result(state, settings.profile_file)

    HealthAgentApp(ask).run()


if __name__ == "__main__":
    main()
