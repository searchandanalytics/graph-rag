# GraphRAG with FalkorDB & Ollama

A powerful, locally-hosted Retrieval Augmented Generation (RAG) system that combines Knowledge Graphs with Vector Search. This project leverages **FalkorDB** for graph storage and **Ollama** for local LLM inference, ensuring data privacy and offline capability.

## 🚀 Features

*   **Hybrid Search**: Combines vector similarity search with graph traversal for richer context.
*   **Local Inference**: Uses Ollama (Llama 3, Mistral, etc.) for zero-cost, private text generation and embeddings.
*   **Knowledge Graph**: Automatically extracts entities and relationships from text to build a structured knowledge base.
*   **FastAPI Backend**: Robust, asynchronous API for ingestion and querying.
*   **FalkorDB Integration**: High-performance, low-latency graph database backed by Redis.

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.10+**
*   **Docker** (for running FalkorDB)
*   **FalkorDB**
*   **Ollama**

### 1. Install & Run Ollama
Download Ollama from [ollama.com](https://ollama.com).

Pull the necessary models:
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 2. Run FalkorDB
The easiest way to run FalkorDB is via Docker:

```bash
docker run -p 6379:6379 -it --rm falkordb/falkordb
```

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd graph-rag
    ```

2.  **Install dependencies**
    This project uses `poetry` or `pip`.

    **Using Poetry:**
    ```bash
    poetry install
    ```

    **Using Pip:**
    ```bash
    pip install .
    ```

3.  **Configuration**
    Create a `.env` file in the root directory:

    ```bash
    cp .env.example .env  # If example exists, otherwise create new
    ```

    Add the following configuration:
    ```ini
    FALKORDB_HOST=localhost
    FALKORDB_PORT=6379
    
    # Ollama Configuration
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_EMBEDDING_MODEL=nomic-embed-text
    OLLAMA_TEXT_MODEL=deepseek-r1:8b
    ```

## 🚀 Usage

Start the API server:

```bash
# Using Python directly
python -m uvicorn src.graph_rag.app:app --port 8081 --reload

# Using Poetry
poetry run uvicorn src.graph_rag.app:app --port 8081 --reload
```

The API will be available at `http://localhost:8081`.

## 📚 API Endpoints

### 1. Health Check
*   **GET** `/api/v1/health`
    *   Checks connectivity to FalkorDB and Ollama.

### 2. Ingest Data
*   **POST** `/api/v1/ingest`
    *   Ingests a single document.
    *   **Body**:
        ```json
        {
          "doc_id": "unique-id-1",
          "text": "Your document text here...",
          "metadata": {"source": "wiki"},
          "generate_embeddings": true
        }
        ```

*   **POST** `/api/v1/ingest/batch`
    *   Ingests multiple documents in bulk.

### 3. Query
*   **POST** `/api/v1/query`
    *   Performs a hybrid search and generates an answer.
    *   **Body**:
        ```json
        {
          "query": "How does GraphRAG work?",
          "top_k": 5,
          "include_sources": true
        }
        ```

### 4. Graph Inspection
*   **POST** `/api/v1/graph/query`
    *   Executes a raw Cypher query against the knowledge graph.
    *   **Body**:
        ```json
        {
          "cypher_query": "MATCH (e:Entity) RETURN e.name LIMIT 10"
        }
        ```

*   **GET** `/api/v1/graph/stats`
    *   Returns statistics about nodes and relationships in the graph.

## 🏗️ Project Structure

```
graph-rag/
├── src/
│   └── graph_rag/
│       ├── app.py             # FastAPI application entry point
│       ├── config.py          # Configuration management
│       ├── embeddings.py      # Ollama embedding utilities
│       ├── graph_builder.py   # Graph construction logic
│       ├── query_engine.py    # RAG query & hybrid search logic
│       └── advanced_features.py # Advanced analytics (Topic detection)
├── pyproject.toml             # Project dependencies
├── .env                       # Environment variables
└── README.md                  # Documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
