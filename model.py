import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class ChatBot:
    def __init__(self, model_name="all-MiniLM-L6-v2", index_file="data/faiss.index"):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_file)
        self.questions = np.load("data/questions.npy", allow_pickle=True)
        self.answers = np.load("data/answers.npy", allow_pickle=True)

    def search(self, query, top_k=1):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "matched_question": self.questions[idx],
                "answer": self.answers[idx],
                "score": float(distances[0][i])
            })
        return results
