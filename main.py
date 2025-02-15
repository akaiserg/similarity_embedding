from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from Levenshtein import distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Union
import numpy as np
from collections import Counter

class TextComparison:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        # Initialize for semantic embedding
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Initialize for TF-IDF
        self.tfidf = TfidfVectorizer()
    
    def text_to_vector(self, text: str) -> np.ndarray:
        """
        Convert text to a word frequency vector
        """
        words = text.lower().split()
        word_counts = Counter(words)
        all_words = sorted(list(set(words)))
        return np.array([word_counts[word] for word in all_words])
    
    def cosine_similarity_custom(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity using word frequency vectors
        """
        # Get all unique words from both texts
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        all_words = sorted(list(words1.union(words2)))
        
        # Create word frequency vectors
        vector1 = np.array([text1.lower().split().count(word) for word in all_words])
        vector2 = np.array([text2.lower().split().count(word) for word in all_words])
        
        # Calculate cosine similarity
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

    def string_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate string-based similarity using Levenshtein distance
        Returns normalized similarity score (0-1)
        """
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0
        return 1 - (distance(text1, text2) / max_len)

    def token_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate token-based similarity using TF-IDF and cosine similarity
        """
        # Fit and transform the texts
        tfidf_matrix = self.tfidf.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        similarity = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
        return float(similarity)

    def embedding_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using transformer embeddings
        """
        # Tokenize texts
        encoded = self.tokenizer([text1, text2], padding=True, truncation=True, return_tensors='pt')
        
        # Get embeddings
        with torch.no_grad():
            model_output = self.model(**encoded)
        
        # Mean pooling
        attention_mask = encoded['attention_mask']
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
        return similarity.item()

    def choose_comparison_method(self, text1: str, text2: str, criteria: Dict[str, bool]) -> Dict[str, Union[float, str]]:
        """
        Choose and apply the appropriate comparison method based on criteria
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            criteria: Dictionary with boolean flags for:
                     - exact_match: Need exact string matching
                     - speed_critical: Need fast comparison
                     - semantic_understanding: Need meaning-based comparison
                     - use_cosine: Use cosine similarity
        
        Returns:
            Dictionary with similarity score and method used
        """
        # Input validation
        if not isinstance(text1, str) or not isinstance(text2, str):
            raise ValueError("Both texts must be strings")
        
        # Preprocess texts
        text1 = text1.strip().lower()
        text2 = text2.strip().lower()
        
        # Choose method based on criteria
        if criteria.get('use_cosine'):
            similarity = self.cosine_similarity_custom(text1, text2)
            method = 'cosine_similarity'
        elif criteria.get('exact_match'):
            similarity = self.string_similarity(text1, text2)
            method = 'string_similarity'
        elif criteria.get('speed_critical'):
            similarity = self.token_similarity(text1, text2)
            method = 'token_similarity'
        elif criteria.get('semantic_understanding'):
            similarity = self.embedding_similarity(text1, text2)
            method = 'embedding_similarity'
        else:
            # Default to token similarity as a balanced approach
            similarity = self.token_similarity(text1, text2)
            method = 'token_similarity'
            
        return {
            'similarity': similarity,
            'method': method
        }

# Example usage
if __name__ == "__main__":
    # Initialize the comparison class
    comparator = TextComparison()
    
    # Example texts
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "A fast brown fox leaps over a sleepy canine"
    
    # Try different criteria
    criteria_examples = [
        {'exact_match': True},
        {'speed_critical': True},
        {'semantic_understanding': True},
        {'use_cosine': True}
    ]
    
    # Compare using different methods
    for criteria in criteria_examples:
        result = comparator.choose_comparison_method(text1, text2, criteria)
        print(f"\nCriteria: {criteria}")
        print(f"Method used: {result['method']}")
        print(f"Similarity score: {result['similarity']:.4f}")