# query_engine.py
import numpy as np
import json
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from .embeddings import OllamaEmbeddingService
from .config import init_falkordb, Config
import ollama

class GraphRAGQueryEngine:
    def __init__(self):
        self.db = init_falkordb()
        self.graph = self.db.select_graph("knowledge_graph")
        self.embedding_service = OllamaEmbeddingService()
        self.config = Config()
        self.text_model = self.config.TEXT_MODEL
        self.client = ollama.Client(host=self.config.OLLAMA_BASE_URL)
        
    def hybrid_search(self, query: str, top_k: int = 5):
        """Combine vector and graph search"""
        
        # 1. Vector similarity search
        query_embedding = self.embedding_service.get_embedding(query)
        
        # Get all chunks with embeddings
        result = self.graph.query("""
            MATCH (c:Chunk)
            WHERE c.embedding IS NOT NULL
            RETURN c.id as id, c.text as text, c.embedding as embedding
            LIMIT 1000
        """)
        
        # Calculate similarities
        chunks = []
        for record in result.result_set:
            try:
                chunk_embedding = json.loads(record[2])
                similarity = cosine_similarity(
                    [query_embedding],
                    [chunk_embedding]
                )[0][0]
                chunks.append({
                    'id': record[0],
                    'text': record[1],
                    'similarity': similarity,
                    'type': 'vector'
                })
            except:
                continue
        
        # Sort by similarity
        vector_results = sorted(chunks, key=lambda x: x['similarity'], reverse=True)[:top_k]
        
        # 2. Graph traversal search
        # Extract query entities
        entities = self.embedding_service.extract_entities(query)
        entity_names = [e['name'] for e in entities.get('entities', [])]
        
        graph_results = []
        if entity_names:
            result = self.graph.query("""
                MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk)
                WHERE e.name IN $entities
                RETURN c.id as id, c.text as text,
                   COUNT(*) as relevance_score,
                   COLLECT(DISTINCT e.name) as matched_entities
                ORDER BY relevance_score DESC
                LIMIT $top_k
            """, {'entities': entity_names, 'top_k': top_k})
            
            for record in result.result_set:
                graph_results.append({
                    'id': record[0],
                    'text': record[1],
                    'relevance': record[2],
                    'entities': record[3],
                    'type': 'graph'
                })
        
        # 3. Combine results (simple fusion)
        combined_results = self.fuse_results(vector_results, graph_results)
        
        # 4. Expand with graph context
        expanded_context = self.expand_with_graph(combined_results)
        
        return expanded_context
    
    def fuse_results(self, vector_results, graph_results, alpha=0.7):
        """Fuse vector and graph search results"""
        # Create a map of chunks by ID
        chunk_map = {}
        
        # Add vector results with weight
        for chunk in vector_results:
            chunk_id = chunk['id']
            chunk_map[chunk_id] = {
                **chunk,
                'score': chunk.get('similarity', 0) * alpha
            }
        
        # Add or update with graph results
        for chunk in graph_results:
            chunk_id = chunk['id']
            if chunk_id in chunk_map:
                # Update score: combine vector and graph
                chunk_map[chunk_id]['score'] += chunk.get('relevance', 0) * (1 - alpha)
                chunk_map[chunk_id]['entities'] = chunk.get('entities', [])
                chunk_map[chunk_id]['type'] = 'hybrid'
            else:
                chunk_map[chunk_id] = {
                    **chunk,
                    'score': chunk.get('relevance', 0) * (1 - alpha)
                }
        
        # Sort by combined score
        sorted_chunks = sorted(chunk_map.values(), key=lambda x: x['score'], reverse=True)
        return sorted_chunks[:10]  # Return top 10 combined results
    
    def expand_with_graph(self, chunks):
        """Expand search results with related entities"""
        if not chunks:
            return chunks
            
        chunk_ids = [chunk['id'] for chunk in chunks[:5]]  # Use top 5 for expansion
        
        result = self.graph.query("""
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE c.id IN $chunk_ids
            WITH e, COUNT(*) as frequency
            ORDER BY frequency DESC
            LIMIT 5
            
            MATCH (e)<-[:MENTIONS]-(related:Chunk)
            WHERE NOT related.id IN $chunk_ids
            RETURN DISTINCT related.id as id, related.text as text,
                   COLLECT(DISTINCT e.name) as related_entities
            ORDER BY rand()  // Random sampling for diversity
            LIMIT 3
        """, {'chunk_ids': chunk_ids})
        
        expanded_chunks = chunks.copy()
        for record in result.result_set:
            expanded_chunks.append({
                'id': record[0],
                'text': record[1],
                'type': 'expanded',
                'related_entities': record[2]
            })
        
        return expanded_chunks
    
    def generate_answer(self, query: str, context: List[Dict]) -> Dict:
        """Generate answer using Ollama"""
        if not context:
            return {
                'answer': "I couldn't find relevant information in the knowledge base.",
                'sources': []
            }
        
        # Format context
        context_text = "\n\n".join([
            f"[Source {i+1} - {c.get('type', 'unknown')}]: {c['text']}" 
            for i, c in enumerate(context[:8])  # Limit context tokens
        ])
        
        prompt = f"""You are an AI assistant answering questions based on the provided context.
        
        Context Information:
        {context_text}
        
        User Question: {query}
        
        Instructions:
        1. Answer using ONLY the information from the context above
        2. If the context doesn't contain relevant information, say so
        3. Be specific and cite sources when possible (e.g., "According to Source 1...")
        4. Keep the answer concise but comprehensive
        
        Answer:"""
        
        try:
            response = self.client.chat(
                model=self.text_model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            return {
                'answer': response['message']['content'],
                'sources': [c['id'] for c in context],
                'context_count': len(context)
            }
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': [],
                'context_count': 0
            }
    
    def query(self, query: str, top_k: int = 5) -> Dict:
        """Complete query pipeline"""
        # Search for relevant context
        context = self.hybrid_search(query, top_k)
        
        # Generate answer
        result = self.generate_answer(query, context)
        
        # Add metadata
        result['query'] = query
        result['retrieved_chunks'] = len(context)
        
        return result
