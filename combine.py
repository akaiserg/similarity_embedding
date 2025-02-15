from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from Levenshtein import distance
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Union, List
import numpy as np
from collections import Counter

class TextComparison:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
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
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        all_words = sorted(list(words1.union(words2)))
        
        vector1 = np.array([text1.lower().split().count(word) for word in all_words])
        vector2 = np.array([text2.lower().split().count(word) for word in all_words])
        
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

    def levenshtein_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate string-based similarity using Levenshtein distance
        Returns normalized similarity score (0-1)
        """
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0
        return 1 - (distance(text1, text2) / max_len)

    def tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate token-based similarity using TF-IDF and cosine similarity
        """
        tfidf_matrix = self.tfidf.fit_transform([text1, text2])
        similarity = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
        return float(similarity)

    def embedding_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using transformer embeddings
        """
        encoded = self.tokenizer([text1, text2], padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            model_output = self.model(**encoded)
        
        attention_mask = encoded['attention_mask']
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
        return similarity.item()

    def combine_methods(self, text1: str, text2: str, methods: List[str], weights: List[float] = None) -> Dict[str, Union[float, dict]]:
        """
        Combine multiple comparison methods with optional weights
        """
        available_methods = {
            'levenshtein': self.levenshtein_similarity,
            'tfidf': self.tfidf_similarity,
            'semantic': self.embedding_similarity,
            'cosine': self.cosine_similarity_custom
        }
        
        # Validate methods
        for method in methods:
            if method not in available_methods:
                raise ValueError(f"Invalid method. Available methods: {list(available_methods.keys())}")
        
        # Set default weights if none provided
        if weights is None:
            weights = [1/len(methods)] * len(methods)
        
        if len(weights) != len(methods):
            raise ValueError("Number of weights must match number of methods")
        
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("Weights must sum to 1")
        
        # Calculate individual scores
        scores = {}
        for method, weight in zip(methods, weights):
            scores[method] = available_methods[method](text1, text2)
        
        # Calculate weighted combination
        combined_score = sum(scores[method] * weight 
                           for method, weight in zip(methods, weights))
        
        return {
            'combined_similarity': combined_score,
            'individual_scores': scores,
            'weights_used': dict(zip(methods, weights))
        }

    def adaptive_combine(self, text1: str, text2: str) -> Dict[str, Union[float, dict]]:
        """
        Adaptively combine methods based on text characteristics
        """
        # Analyze text characteristics
        len_diff = abs(len(text1) - len(text2)) / max(len(text1), len(text2))
        word_overlap = len(set(text1.split()) & set(text2.split())) / len(set(text1.split()) | set(text2.split()))
        
        # Adjust weights based on characteristics
        semantic_weight = 0.4
        levenshtein_weight = 0.2
        tfidf_weight = 0.2
        cosine_weight = 0.2
        
        if len_diff > 0.5:
            semantic_weight += 0.2
            levenshtein_weight -= 0.1
            tfidf_weight -= 0.05
            cosine_weight -= 0.05
        
        if word_overlap < 0.2:
            semantic_weight += 0.1
            tfidf_weight -= 0.05
            cosine_weight -= 0.05
            
        total = semantic_weight + levenshtein_weight + tfidf_weight + cosine_weight
        weights = [semantic_weight/total, levenshtein_weight/total, 
                  tfidf_weight/total, cosine_weight/total]
        
        return self.combine_methods(
            text1, text2,
            methods=['semantic', 'levenshtein', 'tfidf', 'cosine'],
            weights=weights
        )

# Example usage
if __name__ == "__main__":
    comparator = TextComparison()
    
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "A fast brown fox leaps over a sleepy canine"
    
    # Example 1: Equal weights for two methods
    result = comparator.combine_methods(
        text1, text2,
        methods=['semantic', 'levenshtein'],
        weights=[0.5, 0.5]
    )
    print("\nEqual weights combination:")
    print(f"Combined score: {result['combined_similarity']:.4f}")
    print("Individual scores:", result['individual_scores'])
    
    # Example 2: Custom weights for three methods
    result = comparator.combine_methods(
        text1, text2,
        methods=['semantic', 'tfidf', 'cosine'],
        weights=[0.6, 0.2, 0.2]
    )
    print("\nCustom weights combination:")
    print(f"Combined score: {result['combined_similarity']:.4f}")
    print("Individual scores:", result['individual_scores'])
    
    # Example 3: Adaptive combination
    result = comparator.adaptive_combine(text1, text2)
    print("\nAdaptive combination:")
    print(f"Combined score: {result['combined_similarity']:.4f}")
    print("Individual scores:", result['individual_scores'])
    print("Weights used:", result['weights_used'])