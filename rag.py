import re
import os
import heapq
import numpy as np
import networkx as nx
from openai import OpenAI
from database import MilvusDB
from dotenv import load_dotenv

load_dotenv()

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks.
    
    Args:
        text (str): Input text.
        chunk_size (int): Chunk size in characters.
        overlap (int): Overlap between chunks.
        
    Returns:
        List[Dict]: List of chunks with metadata.
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append({
                "text": chunk,
                "index": len(chunks),
                "start_pos": i,
                "end_pos": i + len(chunk)
            })
    print(f"Created {len(chunks)} text chunks")
    return chunks

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.
    
    Returns:
        float: Cosine similarity value.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def build_knowledge_graph(chunks, embedding_func=None):
    """
    Build a knowledge graph from text chunks.
    
    Args:
        chunks (List[Dict]): List of text chunks.
        embedding_func (callable, optional): Function to compute embedding for a given text. 
            If None, an error will be raised.
    
    Returns:
        Tuple[nx.Graph, List[np.ndarray]]: Graph and a list of embeddings.
    """
    if embedding_func is None:
        raise ValueError("An embedding function must be provided to build the graph.")
    
    print("Building knowledge graph...")
    graph = nx.Graph()
    texts = [chunk["text"] for chunk in chunks]
    
    print("Creating embeddings for chunks...")
    embeddings = embedding_func(texts)
    
    print("Adding nodes to the graph...")
    for i, chunk in enumerate(chunks):
        concepts = re.findall(r'\b\w{4,}\b', chunk["text"])[:5]
        graph.add_node(i, text=chunk["text"], concepts=concepts, embedding=embeddings[i])
    
    print("Creating edges between nodes...")
    for i in range(len(chunks)):
        node_concepts = set(graph.nodes[i]["concepts"])
        for j in range(i + 1, len(chunks)):
            other_concepts = set(graph.nodes[j]["concepts"])
            shared_concepts = node_concepts.intersection(other_concepts)
            if shared_concepts:
                sim = cosine_similarity(embeddings[i], embeddings[j])
                concept_score = len(shared_concepts) / min(len(node_concepts) or 1, len(other_concepts) or 1)
                edge_weight = 0.7 * sim + 0.3 * concept_score
                if edge_weight > 0.6:
                    graph.add_edge(i, j, weight=edge_weight, similarity=sim, shared_concepts=list(shared_concepts))
    print(f"Knowledge graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph, embeddings

def traverse_graph(query, graph, embeddings, top_k=5, max_depth=3):
    """
    Traverse the knowledge graph to find relevant information for the query.
    
    Args:
        query (str): The user query.
        graph (nx.Graph): The knowledge graph.
        embeddings (List): List of node embeddings.
        top_k (int): Number of initial nodes to consider.
        max_depth (int): Maximum traversal depth.
    
    Returns:
        Tuple[List[Dict], List]: Relevant chunks and the traversal path.
    """
    print(f"Traversing graph for query: {query}")
    query_embedding = embeddings[0] * 0 
    query_embedding = embeddings[0]  
    
    scores = []
    for i, emb in enumerate(embeddings):
        sim = cosine_similarity(query_embedding, emb)
        scores.append((i, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    starting_nodes = [node for node, _ in scores[:top_k]]
    print(f"Starting traversal from {len(starting_nodes)} nodes")
    
    visited = set()
    traversal_path = []
    results = []
    queue = []
    for node in starting_nodes:
        # Use negative similarity for max-heap
        heapq.heappush(queue, (-scores[node][1], node))
    
    while queue and len(results) < (top_k * 3):
        _, node = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        traversal_path.append(node)
        results.append({
            "text": graph.nodes[node]["text"],
            "concepts": graph.nodes[node]["concepts"],
            "node_id": node
        })
        if len(traversal_path) < max_depth:
            neighbors = [(neighbor, graph[node][neighbor]["weight"]) 
                         for neighbor in graph.neighbors(node) if neighbor not in visited]
            for neighbor, weight in sorted(neighbors, key=lambda x: x[1], reverse=True):
                heapq.heappush(queue, (-weight, neighbor))
    
    print(f"Graph traversal found {len(results)} relevant chunks")
    return results, traversal_path

def generate_response(query, context_chunks, call_openai_func=None):
    """
    Generate a response using the retrieved context chunks.
    
    Args:
        query (str): The user's query.
        context_chunks (List[Dict]): Relevant text chunks.
        call_openai_func (callable, optional): Function to call the language model.
        
    Returns:
        str: The generated response.
    """
    context_texts = [chunk["text"] for chunk in context_chunks]
    combined_context = "\n\n---\n\n".join(context_texts)
    max_context = 14000
    if len(combined_context) > max_context:
        combined_context = combined_context[:max_context] + "... [truncated]"
    
    prompt = f"Context:\n{combined_context}\n\nQuestion: {query}"
    
    # If a language-model call function is provided, use it; otherwise, raise an error.
    if call_openai_func is None:
        raise ValueError("A function for calling the language model must be provided.")
    
    response = call_openai_func(prompt)
    return response

class RAGEngine:
    def __init__(self, embedding_manager, model):
        """
        Initialize the RAG engine with an embedding manager and LLM model.
        
        Args:
            embedding_manager: An instance that manages embeddings.
            model (str): The selected LLM model.
        """
        self.models_map = {
            "llama3": "meta-llama/llama-3.3-70b-instruct:free",
            "gemini": "google/gemini-2.5-pro-exp-03-25:free",
            "deepseek": "deepseek/deepseek-r1-zero:free"
        }
        self.selected_model = model 

        # Initialize the OpenAI client with your credentials
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.embedding_manager = embedding_manager
        self.embedding_model = embedding_manager.embedding_model

        # Initialize Milvus vector database
        self.milvus_db = MilvusDB(embedding_dim=768)

        # Initialize an empty knowledge graph (for graph-based RAG)
        self.knowledge_graph = nx.DiGraph()

        # Container to hold full document texts (populated via file upload)
        self.documents = []
    
    def _call_openai(self, prompt):
        """
        Send request to the OpenAI API.
        """
        completion = self.client.chat.completions.create(
            model=self.models_map[self.selected_model],
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content

    def simple_rag(self, query):
        """
        Simple RAG implementation using semantic search on text chunks.
        
        This method concatenates all uploaded document texts, splits them into chunks, computes
        embeddings for each chunk, performs cosine similarity with the query embedding, selects the top chunks,
        and finally generates the answer using the language model.
        """
        if not self.documents:
            return {"result": "No documents available for simple RAG."}
        
        # Concatenate documents and split into chunks
        full_text = " ".join(self.documents)
        chunks = chunk_text(full_text, chunk_size=1000, overlap=200)
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        # Compute embeddings for each chunk using the engine's embedding manager
        embeddings = self.embedding_manager.get_embeddings(chunk_texts)
        # Compute query embedding
        query_embedding = self.embedding_manager.get_embeddings([query])[0]
        
        # Compute cosine similarity for each chunk
        similarities = []
        for i, emb in enumerate(embeddings):
            sim = cosine_similarity(query_embedding, emb)
            similarities.append((i, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 2 relevant chunks
        top_indices = [idx for idx, _ in similarities[:2]]
        top_chunks = [chunk_texts[i] for i in top_indices]
        
        # Create context string
        context = "\n\n---\n\n".join(top_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        # Generate response using the OpenAI call function
        result = self._call_openai(prompt)
        return {"result": result}

    def graph_rag(self, query):
        """
        Graph-based RAG implementation.
        
        This method concatenates document texts, splits them into chunks, builds a knowledge graph from the chunks,
        traverses the graph to retrieve relevant chunks, and finally generates a response based on these chunks.
        """
        if not self.documents:
            return {"result": "No documents available for Graph RAG."}
        
        full_text = " ".join(self.documents)
        chunks = chunk_text(full_text, chunk_size=1000, overlap=200)
        
        # Build the knowledge graph using the embedding manager's get_embeddings function as the embedding function
        graph, embeddings = build_knowledge_graph(chunks, embedding_func=self.embedding_manager.get_embeddings)
        
        # Traverse the graph to find the most relevant chunks for the query
        relevant_chunks, traversal_path = traverse_graph(query, graph, embeddings, top_k=3)
        
        # Generate a response using the retrieved chunks
        response = generate_response(query, relevant_chunks, call_openai_func=self._call_openai)
        return {"query": query, "result": response}


    def select_rag_method(self, query):
        """
        Selects the appropriate RAG method based on rule-based keywords.
        
        If the query contains graph-related terms, graph_rag is used.
        Otherwise, simple_rag is applied.
        """
        graph_keywords = ["connection", "graph", "structure", "diagram", "ارتباط", "گراف", "ساختار", "نمودار"]
        
        if any(keyword in query for keyword in graph_keywords):
            return self.graph_rag(query)
        else:
            return self.simple_rag(query)
