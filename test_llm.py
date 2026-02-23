from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="mistral")

response = llm.invoke("Explain Retrieval Augmented Generation (RAG) in simple words for a student.")

print(response)