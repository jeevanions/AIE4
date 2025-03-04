import numpy as np
from enum import Enum
from collections import defaultdict
from typing import List, Tuple, Callable
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio
import PyPDF2


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

def euclidean_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the Euclidean distance between two vectors."""
    return np.sqrt(np.sum((vector_a - vector_b) ** 2))

def manhattan_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the Manhattan distance between two vectors."""
    return np.sum(np.abs(vector_a - vector_b))

def minkowski_distance(vector_a: np.array, vector_b: np.array, p: float) -> float:
    """
    Computes the Minkowski distance between two vectors.

    Parameters:
    vector_a (np.array): First vector.
    vector_b (np.array): Second vector.
    p (float): The order of the norm. For example, p=1 gives Manhattan distance, p=2 gives Euclidean distance.

    Returns:
    float: Minkowski distance between vector_a and vector_b.
    """
    # Ensure the input vectors are NumPy arrays
    vector_a = np.asarray(vector_a)
    vector_b = np.asarray(vector_b)

    # Compute Minkowski distance
    distance = np.sum(np.abs(vector_a - vector_b) ** p) ** (1 / p)
    return distance

class DistanceMeasure(Enum):
    COSINE_SIMILARITY = cosine_similarity
    EUCLIDEAN_DISTANCE = euclidean_distance
    MANHATTAN_DISTANCE = manhattan_distance
    MINKOWSKI_DISTANCE = minkowski_distance



class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, key: str, vector: np.array) -> None:
        self.vectors[key] = vector

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
        p: float = None
    ) -> List[Tuple[str, float]]:
        if p:
            scores = [
                (key, distance_measure(query_vector, vector,p))
                for key, vector in self.vectors.items()
            ]
        else:
            scores = [
                (key, distance_measure(query_vector, vector))
                for key, vector in self.vectors.items()
            ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
        p: float = None
    ) -> List[Tuple[str, float]]:
        results = None
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure,p)
        return [result[0] for result in results] if return_as_text else results

    def retrieve_from_key(self, key: str) -> np.array:
        return self.vectors.get(key, None)

    async def abuild_from_list(self, list_of_text: List[str]) -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for text, embedding in zip(list_of_text, embeddings):
            self.insert(text, np.array(embedding))
        return self

    async def build_from_pdf(self, pdf_path: str, chunk_size: int = 250) -> "VectorDatabase":
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        text_chunks = []
        print('Number of pages:', len(pdf_reader.pages))
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            # Split text into chunks of specified size
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                text_chunks.append(chunk)

        print('Number of chunks:', len(text_chunks))
        # Generate embeddings and store them
        for index, chunk in enumerate(text_chunks):
            embedding = self.embedding_model.get_embedding(chunk)
            self.insert(chunk, np.array(embedding))
            if index % 1000 == 0:
                print(f"Processed {index} chunks.")

        return self


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
