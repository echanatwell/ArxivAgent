from typing import Annotated, Sequence, TypedDict, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain.tools import BaseTool
import json 

class AgentState(TypedDict):
    """The state of the agent"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    model: BaseChatModel
    tools_by_name: Dict[str, BaseTool]


def tool_node(state: AgentState):
    sys_prompt = SystemMessage(
        """You are a helpful summarization AI assistant that takes text of Arxiv article and summarizes it into general overview including the most important points.
        The max summarization length is 1500 characters. Minimum summarization length is 400 characters. 
        Do not return anything except summary."""
    )
    summary = "Summaries:"

    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = state["tools_by_name"][tool_call["name"]].invoke(tool_call["args"])
        if tool_call["name"] == "ArticleSummarizingTool":
            for i, article_text in enumerate(tool_result.split("[ARTICLE_BREAK]")):
                if len(article_text.strip()) > 0:
                    inp = [sys_prompt] + [HumanMessage(article_text)]
                    sum_i = state['model'].invoke(inp).content
                    summary += "\n\n" + sum_i
            tool_result = summary
        print("EBANAYA DLINA", len(json.dumps(tool_result)))
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
        You need to, first, search relevant articles by given keywords to get summarization of each article and then make general overview of the approaches
        When you are certain you've got enough article summaries comprise them into related work with references.
        Respond with plain text, do not include enumerates and any other paragraphs"""
        )
    response = state["model"].invoke([system_prompt] + state["messages"], config)
    return {"messages": response}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"