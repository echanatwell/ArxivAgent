from langchain_core.tools import tool
from langchain_community.retrievers.arxiv import ArxivRetriever
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage

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
        # inp = [sys_prompt] + [HumanMessage(doc.page_content)]
        sum_i = f"{doc.metadata['Title']} \n {doc.page_content} \n\n [ARTICLE_BREAK] "
        summary += "\n\n" + sum_i
    
    return summary

@tool("SummarizingTool")
def summarize_tool(text: str):
    """Summarizes article's content into a general overview of the approaches"""
    prompt = f"Summarize the following content into a general overview of the approaches: {text}"
    summary = model.invoke(prompt).content
    return summary
