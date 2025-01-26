from langchain_core.tools import tool
from langchain_community.retrievers.arxiv import ArxivRetriever

from . import llm


@tool("ArticleSummarizingTool")
def arxiv_search_tool(query: str, max_results: int = 2):
    """Search articles on Arxiv by keywords and returns summaries of each article.
    It takes keywords as query, max count of requested articles"""
    retriever = ArxivRetriever(
        load_max_docs=min(5, max_results),
        get_full_documents=True,
        doc_content_chars_max=None,
    )

    docs = retriever.invoke(query)

    summary = ""

    for doc in docs:
        sum_i = f"{doc.metadata['Title']} \n {doc.page_content} \n\n [ARTICLE_BREAK] "
        summary += "\n\n" + sum_i

    return summary


@tool("QueryRewritingTool")
def query_rewriting_tool(query: str) -> str:
    """
    Rewrites a user query for improved search results. Tries not to alter query too much, but enough to get more relevant data.

    Args:
        query (str): The original user query.

    Returns:
        str: The rewritten query, optimized for better search results.
    """
    system_prompt = """You are a helpful assistant that optimizes user search queries for academic research. 
    Rewrite the query to make it more specific and accurate for searching scientific papers. 
    Ensure the rewritten query includes only relevant and necessary keywords. 
    Try making query shorter without loosing information."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Original query: {query}"}
    ]

    rewritten_query = llm.invoke(messages).content.strip()

    return rewritten_query
