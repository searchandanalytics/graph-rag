# advanced_features.py
import ollama
import json
from typing import List, Dict
from .config import Config

class AdvancedGraphFeatures:
    def __init__(self):
        self.config = Config()
        self.client = ollama.Client(host=self.config.OLLAMA_BASE_URL)
        self.text_model = self.config.TEXT_MODEL
        
    def detect_topics_with_ollama(self, chunks: List[Dict]) -> Dict:
        """Use Ollama to detect topics from chunks"""
        chunk_texts = "\n".join([f"Chunk {i}: {chunk['text']}" 
                                for i, chunk in enumerate(chunks[:10])])
        
        prompt = f"""
        Analyze the following text chunks and identify main topics/themes.
        
        Text Chunks:
        {chunk_texts}
        
        Identify 3-5 main topics. For each topic:
        1. Topic name
        2. Brief description
        3. Key entities involved
        4. Relevance score (1-10)
        
        Return in JSON format:
        {{
            "topics": [
                {{
                    "name": "Topic Name",
                    "description": "Brief description",
                    "key_entities": ["Entity1", "Entity2"],
                    "relevance": 8
                }}
            ]
        }}
        """
        
        try:
            response = self.client.chat(
                model=self.text_model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json'
            )
            
            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', response['message']['content'], re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                try:
                    return json.loads(response['message']['content'])
                except:
                    return {"topics": []}
                
        except Exception as e:
            print(f"Error detecting topics: {e}")
            return {"topics": []}
    
    def summarize_document_with_graph(self, doc_id: str) -> Dict:
        """Generate document summary using graph structure"""
        # Get document and its chunks
        from .config import init_falkordb
        db = init_falkordb()
        graph = db.select_graph("knowledge_graph")
        
        result = graph.query("""
            MATCH (d:Document {id: $doc_id})-[:CONTAINS]->(c:Chunk)
            OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
            RETURN d.text as doc_text, 
                   COLLECT(DISTINCT c.text) as chunks,
                   COLLECT(DISTINCT e.name) as entities
        """, {'doc_id': doc_id})
        
        if not result.result_set:
            return {"summary": "Document not found"}
        
        doc_text = result.result_set[0][0]
        chunks = result.result_set[0][1]
        entities = result.result_set[0][2]
        
        # Use Ollama to generate summary
        prompt = f"""
        Summarize the following document based on its content and key entities:
        
        Document: {doc_text[:3000]}  # Limit length
        
        Key Entities Mentioned: {', '.join(entities[:10])}
        
        Please provide:
        1. A concise summary (2-3 paragraphs)
        2. Key points (bullet points)
        3. Main topics/themes
        4. Important relationships between entities
        
        Structure the response in JSON format.
        """
        
        try:
            response = self.client.chat(
                model=self.text_model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json'
            )
            
            # Parse response
            import re
            content = response['message']['content']
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                summary = json.loads(json_match.group())
            else:
                try:
                    summary = json.loads(content)
                except:
                    summary = {"summary": content}
            
            # Store summary in graph
            graph.query("""
                MATCH (d:Document {id: $doc_id})
                SET d.summary = $summary,
                    d.entities = $entities,
                    d.summarized_at = timestamp()
            """, {
                'doc_id': doc_id,
                'summary': json.dumps(summary),
                'entities': json.dumps(entities)
            })
            
            return summary
            
        except Exception as e:
            return {"error": str(e), "summary": "Failed to generate summary"}
