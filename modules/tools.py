from langchain_core.tools import tool
from langchain_community.retrievers.arxiv import ArxivRetriever

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
