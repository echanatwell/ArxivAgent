from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

import json
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

import os
import fitz
import uuid

class AgentState(TypedDict):
    """The state of the agent"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    model: BaseChatModel

@tool("ArticleSummarizingTool")
def arxiv_search_tool(query: int):
    """Search articles on Arxiv by given query identifier and returns summaries of each article.
    It takes keywords as query, max count of requested articles."""

    sys_prompt = SystemMessage(
        """You are a helpful summarization AI assistant that takes text of Arxiv article and summarizes it into general overview including the most important points.
        Maximum length of summarization is 2000 words. Minimum summarization length is 1000 words. 
        """ # Do not return anything except summary.
    )
    summary = "Summaries:"

    folder_name = os.listdir("data")[query]
    print(folder_name)
    for article_name in os.listdir(os.path.join("data", folder_name, "refs")):
        pdf_path = os.path.join("data", folder_name, "refs", article_name)
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()

        inp = [sys_prompt] + [ToolMessage(text, tool_call_id=uuid.uuid4())]
        sum_i = model.invoke(inp).content
        summary += "\n\n" + sum_i
    
    return summary

@tool("GeneralSummarizingTool")
def summarize_tool(text: str):
    """Summarizes content from few articles and makes a general overview of the approaches from given summaries of set od articles."""
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
        """You are a helpful AI assistant that takes a user input and summarize arxiv articles found strictly by keywords from input. 
        You can use tools you have for searching articles and summarizing them.
        You need to, first, search relevant articles by given keywords to get summarization of each article and then make general overview of the approaches.
        When you are certain you've got enough article summaries comprise them into a general overview of the approaches with a detailed description of each of them.
        Your answer must have plain text formatting, do NOT include enumerates, lists or any headers.
        DO NOT change or reformat user input when pass it into tools.
        IMPORTANT: user input must be represented as integer query identifier for articles searching.""" 
        )
    # Use tools sequentially and Use the ONLY ONE tool per call, dont try to use a several at a time. 
    #     If want to use more than one tool choose only one and call other next time.
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
    m = None
    for s in stream:
        for message in s["messages"][idx:]:
            idx += 1
            if isinstance(message, tuple):
                print(message)
            else:
                m = message
                message.pretty_print()
    return m


if __name__ == "__main__":
    model = ChatOllama(model="qwen2.5:7b-instruct", num_ctx=32768)

    tools = [arxiv_search_tool]
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

    folders = os.listdir('data')
    n = 19
    folder_name = folders[n]

    inputs = {"messages": [("user", str(n))]} 
        
    ans = print_stream(graph.stream(inputs, stream_mode="values"))
    ans_text = ans.content

    with open(os.path.join("data", folder_name, "ans.txt"), "w") as f:
        f.write(ans_text)

    
    # for inp in range(len(folders)):
    #     inputs = {"messages": [("user", str(inp))]} 
        
    #     ans = print_stream(graph.stream(inputs, stream_mode="values"))
    #     ans_text = ans.content
    #     with open(os.path.join("data", folders[inp], "rw.txt"), "r") as f:
    #         ref = ""
    #         for line in f.readlines():
    #             ref += line
    #     sc = metric(ans_text, ref)

    #     scores.append((folders[inp], sc))

    # print("+_+_+_+_+_+_+_+_+_+_+_+")
    # print(ans.content)
    
    
