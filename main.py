from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
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


class Source(BaseModel):
    """Schema for a source of information used by the agent."""

    url: str = Field(description="The URL of the source.")


class AgentInput(BaseModel):
    """Schema for agent response with answer and sources."""

    answer: str = Field(description="The answer to the question.")
    sources: List[Source] = Field(
        default_factory=list,
        description="A list of sources used to answer the question.",
    )


llm = ChatOpenAI(model="gpt-5", temperature=0)
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentInput)


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
