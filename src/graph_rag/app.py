# app.py (Updated with lifespan events)
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
import json
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize components at module level (will be set in lifespan)
query_engine = None
graph_builder = None

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    global query_engine, graph_builder
    
    print("Starting up GraphRAG application...")
    
    try:
        # Import inside lifespan to avoid circular imports
        from .query_engine import GraphRAGQueryEngine
        from .graph_builder import GraphBuilder
        
        print("Initializing query engine and graph builder...")
        query_engine = GraphRAGQueryEngine()
        graph_builder = GraphBuilder()
        
        # Create schema if not exists
        try:
            graph_builder.create_schema()
            print("Graph schema created successfully")
        except Exception as e:
            print(f"Schema may already exist: {e}")
        
        print("Application startup complete")
        
    except ImportError as e:
        print(f"Import error during startup: {e}")
    except Exception as e:
        print(f"Error during startup: {e}")
    
    yield  # Application runs here
    
    # Shutdown code
    print("Shutting down GraphRAG application...")
    # Cleanup if needed

# Create FastAPI app with lifespan
app = FastAPI(
    title="GraphRAG API with FalkorDB and Ollama",
    description="Intelligent Graph-based Retrieval Augmented Generation System",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    include_sources: bool = True

class DocumentRequest(BaseModel):
    text: str
    doc_id: str
    metadata: Optional[dict] = None
    generate_embeddings: bool = True

class BatchDocumentRequest(BaseModel):
    documents: List[DocumentRequest]

class GraphQueryRequest(BaseModel):
    cypher_query: str
    params: Optional[dict] = None

class HealthResponse(BaseModel):
    status: str
    falkordb: str
    ollama: str
    embedding_dimension: Optional[int] = None
    error: Optional[str] = None

# Helper function to check if components are initialized
def check_components():
    if query_engine is None or graph_builder is None:
        raise HTTPException(
            status_code=503,
            detail="Service components not initialized. Please try again in a moment."
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "GraphRAG API with FalkorDB and Ollama",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "ingest": "POST /api/v1/ingest",
            "query": "POST /api/v1/query",
            "health": "GET /api/v1/health",
            "stats": "GET /api/v1/graph/stats"
        }
    }

@app.post("/api/v1/ingest")
async def ingest_document(
    request: DocumentRequest,
    background_tasks: BackgroundTasks
):
    """Ingest a single document"""
    check_components()
    
    try:
        # Add document to graph
        graph_builder.add_document(
            doc_id=request.doc_id,
            text=request.text,
            metadata=request.metadata or {}
        )
        
        # Optionally generate embeddings in background
        if request.generate_embeddings:
            from .embeddings import OllamaEmbeddingService
            embedding_service = OllamaEmbeddingService()
            
            # Get chunks for this document
            result = graph_builder.graph.query("""
                MATCH (d:Document {id: $doc_id})-[:CONTAINS]->(c:Chunk)
                RETURN c.id as chunk_id, c.text as text
            """, {'doc_id': request.doc_id})
            
            # Store embeddings in background
            for record in result.result_set:
                background_tasks.add_task(
                    embedding_service.store_chunk_embedding,
                    record[0],  # chunk_id
                    record[1]   # text
                )
        
        return {
            "status": "success",
            "message": "Document ingested",
            "doc_id": request.doc_id,
            "embeddings_scheduled": request.generate_embeddings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting document: {str(e)}")

@app.post("/api/v1/ingest/batch")
async def ingest_batch(request: BatchDocumentRequest):
    """Ingest multiple documents"""
    check_components()
    
    results = []
    for doc_request in request.documents:
        try:
            graph_builder.add_document(
                doc_id=doc_request.doc_id,
                text=doc_request.text,
                metadata=doc_request.metadata or {}
            )
            results.append({
                "doc_id": doc_request.doc_id,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "doc_id": doc_request.doc_id,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "status": "completed",
        "results": results,
        "success_count": len([r for r in results if r["status"] == "success"])
    }

@app.post("/api/v1/query")
async def query(request: QueryRequest):
    """Query the GraphRAG system"""
    check_components()
    
    try:
        result = query_engine.query(
            query=request.query,
            top_k=request.top_k
        )
        
        # Remove sources if not requested
        if not request.include_sources:
            result.pop('sources', None)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/api/v1/graph/query")
async def graph_query(request: GraphQueryRequest):
    """Execute raw Cypher query on the graph"""
    check_components()
    
    try:
        result = graph_builder.graph.query(
            request.cypher_query,
            request.params or {}
        )
        
        # Convert result to serializable format
        records = []
        for record in result.result_set:
            serializable_record = []
            for item in record:
                if isinstance(item, (dict, list)):
                    serializable_record.append(item)
                else:
                    serializable_record.append(str(item))
            records.append(serializable_record)
        
        return {
            "query": request.cypher_query,
            "results": records,
            "count": len(records)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph query error: {str(e)}")

@app.get("/api/v1/graph/stats")
async def get_graph_stats():
    """Get graph statistics"""
    check_components()
    
    try:
        result = graph_builder.graph.query("""
            CALL db.labels() YIELD label
            WITH label
            CALL {
                WITH label
                MATCH (n)
                WHERE labels(n)[0] = label
                RETURN count(n) as count
            }
            RETURN label, count
            ORDER BY count DESC
        """)
        
        stats = {}
        for record in result.result_set:
            stats[record[0]] = record[1]
        
        # Get relationship counts
        rel_result = graph_builder.graph.query("""
            CALL db.relationshipTypes() YIELD relationshipType
            WITH relationshipType
            CALL {
                WITH relationshipType
                MATCH ()-[r]->()
                WHERE type(r) = relationshipType
                RETURN count(r) as count
            }
            RETURN relationshipType, count
            ORDER BY count DESC
        """)
        
        relationships = {}
        for record in rel_result.result_set:
            relationships[record[0]] = record[1]
        
        return {
            "node_counts": stats,
            "relationship_counts": relationships,
            "total_nodes": sum(stats.values()),
            "total_relationships": sum(relationships.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check if components are initialized
        if graph_builder is None:
            raise Exception("GraphBuilder service not initialized")

        # Test database connection
        graph_builder.graph.query("RETURN 1")
        
        # Test Ollama connection
        from .embeddings import OllamaEmbeddingService
        embedding_service = OllamaEmbeddingService()
        test_embedding = embedding_service.get_embedding("test")
        
        return HealthResponse(
            status="healthy",
            falkordb="connected",
            ollama="connected",
            embedding_dimension=len(test_embedding) if test_embedding else 0
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            falkordb="disconnected",
            ollama="error",
            error=str(e)
        )

@app.post("/api/v1/embeddings/generate")
async def generate_embeddings(background_tasks: BackgroundTasks):
    """Generate embeddings for all chunks without embeddings"""
    check_components()
    
    try:
        result = graph_builder.graph.query("""
            MATCH (c:Chunk)
            WHERE c.embedding IS NULL
            RETURN c.id as chunk_id, c.text as text
            LIMIT 1000
        """)
        
        chunk_count = len(result.result_set)
        
        if chunk_count > 0:
            from .embeddings import OllamaEmbeddingService
            embedding_service = OllamaEmbeddingService()
            
            for record in result.result_set:
                background_tasks.add_task(
                    embedding_service.store_chunk_embedding,
                    record[0],  # chunk_id
                    record[1]   # text
                )
        
        return {
            "status": "success",
            "message": f"Scheduled embeddings for {chunk_count} chunks"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scheduling embeddings: {str(e)}")

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    # For development with auto-reload, use:
    # uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    
    # For production (or if you want to run directly)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8081,
        reload=False  # Set to False when running directly
    )
