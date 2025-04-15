from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import numpy as np

class MilvusDB:
    def __init__(self, embedding_dim: int = 768, host: str = "127.0.0.1", port: str = "19530"):
        # Connect to the Milvus Standalone instance
        connections.connect(alias="default", host=host, port=port)

        self.collection_name = "rag_docs"
        self.embedding_dim = embedding_dim

        # Check if the collection exists; if not, create it
        if not utility.has_collection(self.collection_name):
            self._create_collection()

        # Load the collection
        self.collection = Collection(self.collection_name)
        self.collection.load()

    def _create_collection(self):
        """Create a collection with a predefined schema"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        ]
        schema = CollectionSchema(fields, description="RAG Documents Collection")
        collection = Collection(name=self.collection_name, schema=schema)

        # Create an index on the embedding field
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 256}
        }
        collection.create_index(field_name="embedding", index_params=index_params)

    def insert_document(self, text: str, embedding: list, doc_id: int):
        # Check for duplicate document ID
        existing = self.collection.query(f"id == {doc_id}")
        if existing:
            # If document exists, delete the existing document before inserting the new one.
            print(f"Document with id {doc_id} already exists. Replacing it.")
            self.collection.delete(expr=f"id == {doc_id}")
        
            # Convert the embedding to float32 and insert the document
            embedding = np.array(embedding, dtype=np.float32).tolist()
            self.collection.insert([[doc_id], [text], [embedding]])

    def search_similar(self, query_embedding: list, top_k: int = 3):
        """Perform a vector search in Milvus"""
        query_embedding = np.array(query_embedding, dtype=np.float32).tolist()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )
        return [hit.entity.text for hit in results[0]]

    def close(self):
        """Disconnect from Milvus"""
        connections.disconnect("default")
