from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class TextsSimilarity:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the ResponseEvaluator with a SentenceTransformer model.

        Args:
            model_name (str): The name of the SentenceTransformer model to use. Default is 'all-MiniLM-L6-v2'.
        """
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts1: Union[str, List[str]], texts2: Union[str, List[str]]) -> List[float]:
        """
        Evaluate the similarity between pairs of texts using SentenceTransformer embeddings.

        Args:
            texts1 (Union[str, List[str]]): First text or list of texts.
            texts2 (Union[str, List[str]]): Second text or list of texts. Must match the size of texts1.

        Returns:
            List[float]: List of cosine similarity scores for each pair of texts.
        """
        # Ensure inputs are lists
        if isinstance(texts1, str):
            texts1 = [texts1]
        if isinstance(texts2, str):
            texts2 = [texts2]

        if len(texts1) != len(texts2):
            raise ValueError("texts1 and texts2 must have the same number of elements")

        # Compute embeddings
        embeddings1 = self.model.encode(texts1, convert_to_tensor=True).cpu().numpy()
        embeddings2 = self.model.encode(texts2, convert_to_tensor=True).cpu().numpy()

        # Compute cosine similarity for each pair
        similarities = [
            np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            for e1, e2 in zip(embeddings1, embeddings2)
        ]

        return similarities


# Example usage
if __name__ == "__main__":
    evaluator = TextsSimilarity()
    texts1 = ["This is a sample response.", "Another example response."]
    texts2 = ["This is a sample answer.", "A different example response."]
    similarities = evaluator(texts1, texts2)
    print("Similarity scores:", similarities)
