from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from modules.tools import arxiv_search_tool
from modules.nodes import AgentState, call_model, tool_node, should_continue

import streamlit as st

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


if __name__ == '__main__':

    model = ChatOllama(model="qwen2.5:7b-instruct", num_ctx=16384) # 32768

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

    st.title("ArxivAgent app")

    with st.form("my_form"):
        text = st.text_area(
            "Enter keywords:",
            "aboba",
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            inputs = {
                "messages": [("user", text)],
                "model": model_with_tools,
                "tools_by_name": tools_by_name,
            }
            response = print_stream(graph.stream(inputs, stream_mode="values"))
            st.info(response.content)