from dotenv import load_dotenv
from langchain.agents import create_agent

# from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# from tavily import TavilyClient

load_dotenv()

# tavily = TavilyClient()


# @tool
# def search(query: str) -> str:
#     """Searches the web for a query and returns the results.

#     Args:
#         query (str): The search query.
#     """
#     print(f"Searching for {query}")
#     return str(tavily.search(query=query, search_depth="advanced"))


llm = ChatOpenAI(model="gpt-5", temperature=0)
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
                )
            ]
        }
    )
    print(result)


if __name__ == "__main__":
    main()
