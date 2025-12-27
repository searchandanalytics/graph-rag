# embeddings.py
import ollama
import json
import numpy as np
from typing import List, Dict
from .config import Config, init_falkordb

class OllamaEmbeddingService:
    def __init__(self):
        self.config = Config()
        self.embedding_model = self.config.EMBEDDING_MODEL
        self.text_model = self.config.TEXT_MODEL
        self.client = ollama.Client(host=self.config.OLLAMA_BASE_URL)
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding using Ollama"""
        try:
            # Ollama embedding API
            result = self.client.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            return result['embedding']
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 768  # Adjust size based on your model
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in batch"""
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def store_chunk_embedding(self, chunk_id: str, text: str):
        """Store embedding in FalkorDB graph"""
        embedding = self.get_embedding(text)
        
        # Store in FalkorDB
        db = init_falkordb()
        graph = db.select_graph("knowledge_graph")
        
        graph.query("""
            MATCH (c:Chunk {id: $chunk_id})
            SET c.embedding = $embedding
        """, {'chunk_id': chunk_id, 'embedding': json.dumps(embedding)})
    
    def extract_entities(self, text: str) -> Dict:
        """Extract entities from text using Ollama"""
        prompt = f"""
        Extract entities from the following text. Return JSON with entities list.
        For each entity, include name, type (PERSON, ORGANIZATION, LOCATION, etc.), and context.
        
        Text: {text}
        
        Return format:
        {{
            "entities": [
                {{"name": "Entity Name", "type": "TYPE", "context": "Description"}}
            ]
        }}
        """
        
        try:
            response = self.client.chat(
                model=self.text_model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json'
            )
            
            response_text = response['message']['content']
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group())
            else:
                try:
                    return json.loads(response_text)
                except:
                    return {"entities": []}
                
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return {"entities": []}
