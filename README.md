# RAG Chat System – Intelligent Q&A with PDF & Word Files

A Retrieval-Augmented Generation (RAG) based system that answers user questions using uploaded PDF and Word documents. This project combines advanced NLP techniques, a vector database (Milvus), and large language models (LLMs) to deliver highly accurate responses.

---

## Key Features

- Support for uploading and parsing multiple PDF and DOCX files
- Two processing modes: Simple RAG and Graph-based RAG
- Auto-delete or update of duplicate documents in the Milvus database
- Integration with a Milvus vector database for efficient semantic search
- Secure handling of sensitive information through environment variables (.env)
- Semantic search using advanced embeddings (mpnet, distilroberta, bert, etc.)
- Answer generation using LLMs via the OpenRouter API

---

## Prerequisites

- Python 3.8 or higher
- Milvus (Standalone or Docker)
- Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Environment Variables (.env)

Create a `.env` file in the root of your project to safely store sensitive information such as API keys.

```
OPENAI_API_KEY=your_openai_api_key
```

---

## Running the Application

1. Start Milvus (using Docker):

```bash
docker-compose up -d
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## Simple RAG vs. Graph RAG

| Feature                | Simple RAG               | Graph RAG                     |
|------------------------|--------------------------|-------------------------------|
| Processing Method      | Similarity search        | Graph traversal               |
| Performance on complex queries | Moderate              | High                          |
| Speed                  | Fast                     | Slower                        |
| Context-aware chaining | No                       | Yes                           |

---

## APIs Used

- [OpenRouter API](https://openrouter.ai)
- [Milvus Vector DB](https://milvus.io)

---

## Project Screenshot

Below is a screenshot from the project interface.  

![Project Screenshot](https://github.com/SepehrNorouzi7/RAGChat/blob/main/Screenshot.jpg)

---

## Useful Links

- [Milvus Documentation](https://milvus.io/docs)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain & RAG Concepts](https://github.com/hwchase17/langchain)
