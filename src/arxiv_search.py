import arxiv

arciv_cli = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)


def search_arxiv(query: str, max_results: int = 5):
    """
    Поиск статей на arxiv.org по ключевым словам.
    """
    results = []
    for result in arciv_cli.results(arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )):
        results.append({
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "pdf_url": result.pdf_url,
        })
    return results
