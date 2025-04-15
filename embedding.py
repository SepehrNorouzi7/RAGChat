from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingManager:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        """Manages embedding models"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        # Available embedding models
        self.models = {
            "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "distilroberta": "sentence-transformers/all-distilroberta-v1",
            "bert": "sentence-transformers/all-mpnet-base-v2",
        }

        # Select the model based on input
        if model_name in self.models:
            self.embedding_model = HuggingFaceEmbeddings(model_name=self.models[model_name])
        else:
            raise ValueError(f"Model {model_name} not found. Choose from {list(self.models.keys())}")

    def get_embeddings(self, texts: list) -> list:
        """Converts text into embedding vectors"""
        return self.embedding_model.embed_documents(texts)

    def process_text(self, text):
        """Preprocesses and splits the text into chunks"""
        cleaned_text = self._clean_text(text)
        return self.text_splitter.split_text(cleaned_text)

    def _clean_text(self, text):
        """Cleans the text: replaces Arabic characters with Persian equivalents"""
        return text.replace('ي', 'ی').replace('ك', 'ک').strip()
