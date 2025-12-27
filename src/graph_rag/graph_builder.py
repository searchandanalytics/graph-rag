# graph_builder.py
import json
from typing import List, Dict, Any
from falkordb import Graph
import hashlib
import ollama
from .config import Config, init_falkordb

class GraphBuilder:
    def __init__(self, graph_name: str = "knowledge_graph"):
        self.db = init_falkordb()
        self.graph = self.db.select_graph(graph_name)
        self.config = Config()
        self.text_model = self.config.TEXT_MODEL
        self.client = ollama.Client(host=self.config.OLLAMA_BASE_URL)
        
    def create_schema(self):
        """Create indices and constraints"""
        # Create constraints
        self.graph.query("""
            CREATE CONSTRAINT ON (d:Document) ASSERT d.id IS UNIQUE
        """)
        self.graph.query("""
            CREATE CONSTRAINT ON (c:Chunk) ASSERT c.id IS UNIQUE
        """)
        self.graph.query("""
            CREATE CONSTRAINT ON (e:Entity) ASSERT e.name IS UNIQUE
        """)
        
    def chunk_document(self, text: str, chunk_size: int = 500) -> List[Dict]:
        """Split document into chunks with overlap"""
        import re
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - 100):
            chunk_text = ' '.join(words[i:i + chunk_size])
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()
            chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'index': i // chunk_size
            })
        return chunks
    
    def extract_entities_with_ollama(self, text: str) -> List[Dict]:
        """Extract entities using Ollama"""
        prompt = f"""
        Extract entities from the following text and return them in JSON format.
        For each entity, include:
        - name: the entity name
        - type: one of PERSON, ORGANIZATION, LOCATION, EVENT, CONCEPT, PRODUCT, DATE, or OTHER
        - context: brief description of how it's mentioned
        
        Text: {text[:2000]}
        
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
                result = json.loads(json_match.group())
            else:
                try:
                    result = json.loads(response_text)
                except:
                    result = {"entities": []}
                
            return result
            
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return {"entities": []}
    
    def add_document(self, doc_id: str, text: str, metadata: Dict = None):
        """Add document to graph"""
        # Create document node
        self.graph.query("""
            MERGE (d:Document {id: $doc_id})
            SET d.text = $text,
                d.metadata = $metadata,
                d.created_at = timestamp()
        """, {'doc_id': doc_id, 'text': text, 'metadata': json.dumps(metadata or {})})
        
        # Create chunks
        chunks = self.chunk_document(text)
        for chunk in chunks:
            # Create chunk node
            self.graph.query("""
                MERGE (c:Chunk {id: $id})
                SET c.text = $text,
                    c.index = $index
            """, chunk)
            
            # Link document to chunk
            self.graph.query("""
                MATCH (d:Document {id: $doc_id})
                MATCH (c:Chunk {id: $chunk_id})
                MERGE (d)-[:CONTAINS {order: $index}]->(c)
            """, {'doc_id': doc_id, 'chunk_id': chunk['id'], 'index': chunk['index']})
            
            # Extract and link entities using Ollama
            entities_result = self.extract_entities_with_ollama(chunk['text'])
            for entity in entities_result.get('entities', []):
                self.graph.query("""
                    MERGE (e:Entity {name: $name, type: $type})
                    MERGE (c:Chunk {id: $chunk_id})
                    MERGE (c)-[:MENTIONS {context: $context}]->(e)
                """, {
                    'name': entity['name'],
                    'type': entity.get('type', 'UNKNOWN'),
                    'chunk_id': chunk['id'],
                    'context': entity.get('context', '')
                })
