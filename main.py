from dotenv import load_dotenv
from typing import Any, cast

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG")
    print(app.invoke(input=cast(Any, {"question": "agent memory?"})))
