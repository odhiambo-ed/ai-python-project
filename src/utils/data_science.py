import numpy as np
import pandas as pd
from typing import Tuple, List

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit vectors."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform basic dataset analysis."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "description": df.describe().to_dict(),
    }