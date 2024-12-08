from langchain_ollama import ChatOllama

# Настройка модели ChatOllama
llama = ChatOllama(model="llama3.1:8b-instruct-q6_K")

def summarize_text(text: str, max_tokens: int = 150):
    """
    Суммаризация текста с помощью Ollama (Llama).
    """
    prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
    response = llama(prompt, max_tokens=max_tokens)
    return response
