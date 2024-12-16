from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_community.retrievers.arxiv import ArxivRetriever

import json
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from langchain_core.language_models import BaseChatModel
from langchain.tools import BaseTool
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    """The state of the agent"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    model: BaseChatModel
    # tools: Sequence[BaseTool]

@tool("ArticleSummarizingTool")
def arxiv_search_tool(query: str, max_results: int = 2):
    """Search articles on Arxiv by keywords and returns summaries of each article.
    It takes keywords as query, max count of requested articles""" # and model from AgentState
    retriever = ArxivRetriever(
        load_max_docs=min(5, max_results), 
        get_full_documents=True, 
        doc_content_chars_max=None,
    )

    docs = retriever.invoke(query)

    sys_prompt = SystemMessage(
        """You are a helpful summarization AI assistant that takes text of Arxiv article and summarizes it into general overview including the most important points.
        The max summarization length is 1500 characters. Minimum summarization length is 400 characters. 
        Do not return anything except summary."""
    )
    summary = "Summaries:"

    # print("query: ", query, "num docs:", len(docs))
    for i, doc in enumerate(docs):
        inp = [sys_prompt] + [HumanMessage(doc.page_content)]
        sum_i = f"{doc.metadata['Title']} \n {model.invoke(inp).content}"
        summary += "\n\n" + sum_i
    
    return summary

# @tool("SummarizingTool") 
# def summarize_tool(text: str):
#     """Summarizes text using a language model"""
#     prompt = f"Summarize the following article: {text}"
#     summary = model(prompt)
#     return summary

@tool("SummarizingTool")
def summarize_tool(text: str):
    """Summarizes article's content into a general overview of the approaches"""
    prompt = f"Summarize the following content into a general overview of the approaches: {text}"
    summary = model.invoke(prompt).content
    return summary

def tool_node(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

def call_model(
        state: AgentState,
        config: RunnableConfig,
):
    system_prompt = SystemMessage(
        """You are a helpful AI assistant that takes a user input and summarize arxiv articles found by strictly keywords from input. 
        You can use tool you have for searching articles and summarizing them.
        You need to search relevant articles by given keywords and summarize approaches from all found articles and make general overview of the approaches
        When you are certain you've got enough article summaries comprise them into related work with references.
        Respond with plain text, do not include enumerates and any other paragraphs""" # , do not include enumerates and any other paragraphs beside Related Work.
        )
    response = model_with_tools.invoke([system_prompt] + state["messages"], config)
    return {"messages": response}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

def print_stream(stream):
    idx = 0
    for s in stream:
        for message in s["messages"][idx:]:
            idx += 1
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()


if __name__ == '__main__':

    model = ChatOllama(model="llama3.2:3b-instruct-fp16", num_ctx=32768)

    tools = [arxiv_search_tool, summarize_tool]
    model_with_tools = model.bind_tools(tools)

    tools_by_name = {tool.name: tool for tool in tools}

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )

    workflow.add_edge("tools", "agent")

    graph = workflow.compile()

    inputs = {"messages": [("user", "attention")]}
    print_stream(graph.stream(inputs, stream_mode="values"))