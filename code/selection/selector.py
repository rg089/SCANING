from .utils import load_config
from .metrics import similarity, diversity
from sentence_transformers import SentenceTransformer


class Selector():
    def __init__(self, sim_model_path) -> None:
        self.config = load_config()
        self.weights = self.config['weights']
        self.metrics = list(self.weights.keys())
        self.sim_model = SentenceTransformer(sim_model_path)
        print(f"[INFO] Semantic Similarity Model has been loaded.")

    def score(self, original, candidates):
        similarity_scores = similarity(self.sim_model, original, candidates)
        diversity_scores = diversity(original, candidates)
        overall_scores = [self.weights['similarity'] * s + \
        self.weights['diversity'] * d for s, d in zip(similarity_scores, diversity_scores)]
        return similarity_scores, diversity_scores, overall_scores

    def select(self, original, candidates, top_n=-1, **kwargs):
        """
        will return the top_n candidates with the highest scores
        the return format will be a list of tuples 
        (candidate, overall_score, similarity_score, diversity_score) sorted with
        highest score coming in first
        """
        scores = []
        similarity_scores, diversity_scores, overall_scores = self.score(original, candidates)
        for candidate, score, sim_score, div_score in \
            zip(candidates, overall_scores, similarity_scores, diversity_scores):
            scores.append((candidate, score, sim_score, div_score)) 
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        if top_n > 0: return scores[:top_n]
        return scores
        
        
        
