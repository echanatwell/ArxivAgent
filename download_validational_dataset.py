import os
import re
import requests
import arxiv
import fitz  # PyMuPDF for PDF processing

# Create main output directory
os.makedirs("output", exist_ok=True)

def download_arxiv_pdf(paper_id, save_path):
    url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {save_path}")
    else:
        print(f"Failed to download: {url}")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_related_work(text):
    # Extract "Related Work" section
    sections = text.split("\n")
    related_work = []
    capture = False
    for line in sections:
        if "Related Work" in line:
            capture = True
        elif capture and (line.strip() == "" or "References" in line):
            break
        if capture:
            related_work.append(line)
    return "\n".join(related_work)

def extract_citations(text):
    # Extract citation numbers like [12] or [1, 2, 3]
    citations = re.findall(r"\[(\d+(?:,\s?\d+)*)\]", text)
    flattened_citations = set()
    for group in citations:
        flattened_citations.update(group.split(","))
    return sorted(flattened_citations, key=int)

def extract_references_section(text):
    # Extract the "References" section
    references_start = text.find("References")
    if references_start == -1:
        return ""
    return text[references_start:]

def find_reference_by_number(references_text, number):
    # Match references by their leading number (e.g., "12.")
    pattern = re.compile(rf"^{number}\.\s(.+)$", re.MULTILINE)
    match = pattern.search(references_text)
    if match:
        return match.group(1).strip()
    return None

def process_paper(result: arxiv.Result, output_dir):
    title = result.title.replace("/", "-").replace(" ", "_")
    paper_dir = os.path.join(output_dir, title)
    os.makedirs(paper_dir, exist_ok=True)

    # Step 1: Download PDF
    pdf_path = os.path.join(paper_dir, f"{title}.pdf")
    result.download_pdf(paper_dir, f"{title}.pdf")
    print(result.links)
    # Step 2: Extract "Related Work" section
    paper_text = extract_text_from_pdf(pdf_path)
    related_work = extract_related_work(paper_text)
    if not related_work.strip():
        print(f"No 'Related Work' section found in {title}, skipping.")
    else:
        with open(os.path.join(paper_dir, "rw.txt"), "w") as f:
            f.write(related_work)
        print(f"Saved Related Work section for {title}")
    for related_paper in result.links:
        if related_paper.rel == "related" and related_paper.content_type == "pdf":
            print("AAAAAAAAAAAAA", related_paper.href)
    # # Step 3: Extract and process citations
    # citations = extract_citations(related_work)
    # references_text = extract_references_section(paper_text)
    #
    # refs_dir = os.path.join(paper_dir, "refs")
    # os.makedirs(refs_dir, exist_ok=True)
    #
    # for citation in citations:
    #     reference = find_reference_by_number(references_text, citation.strip())
    #     if reference and "arxiv.org" in reference:
    #         arxiv_id_match = re.search(r"arxiv\.org/(abs|pdf)/([\w\.\-]+)", reference)
    #         if arxiv_id_match:
    #             arxiv_id = arxiv_id_match.group(2)
    #             citation_pdf_path = os.path.join(refs_dir, f"{arxiv_id}.pdf")
    #             download_arxiv_pdf(arxiv_id, citation_pdf_path)


def main():
    query = "machine learning"
    max_results = 100

    # Fetch random papers matching query
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    cli = arxiv.Client()

    for result in cli.results(search):
        print("processing", result.title)
        process_paper(result, "output")

if __name__ == "__main__":
    main()
