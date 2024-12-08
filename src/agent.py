from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama

from .arxiv_search import search_arxiv
from .summarizer import summarize_text
from . import cfg

# Настройка модели ChatOllama
llama = ChatOllama(model="llama3.1:8b-instruct-q6_K")


# Инструмент для поиска статей
def arxiv_tool(query: str):
    results = search_arxiv(query)
    return "\n\n".join(
        [f"Title: {r['title']}\nAuthors: {', '.join(r['authors'])}\nSummary: {r['summary']}" for r in results]
    )


search_tool = Tool(
    name="Arxiv Search",
    func=arxiv_tool,
    description="Поиск статей на arxiv.org по ключевым словам."
)


# Инструмент для суммаризации текста
def summarization_tool(text: str):
    return summarize_text(text)


summarizer_tool = Tool(
    name="Summarizer",
    func=summarization_tool,
    description="Суммаризация текста статей."
)


def create_agent():
    """
    Создание ReAct-агента с инструментами поиска и суммаризации.
    """
    tools = [search_tool, summarizer_tool]
    agent = create_react_agent(llama, tools=tools, debug=cfg.DEBUG)
    return agent
