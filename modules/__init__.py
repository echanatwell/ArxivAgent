from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b-instruct", num_ctx=16384)
