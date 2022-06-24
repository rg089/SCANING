from selection.metrics import diversity_score
from .utils import load_config
from .metrics import similarity, diversity, numeracy
from sentence_transformers import SentenceTransformer
import numpy as np


class Scorer():
    def __init__(self, sim_model_path, select=True) -> None:
        self.config = load_config()
        self.weights = self.config['weights']
        self.metrics = list(self.weights.keys())
        self.sim_model1 = SentenceTransformer(sim_model_path)
        self.sim_model2 = SentenceTransformer("all-MiniLM-L12-v1")
        self.select = select


    def weighted_gm(self, sim, div, num, weights):
        eps = 1e-9
        return np.exp((weights[0]*np.log(sim+eps) + weights[1]*np.log(div+eps) + weights[2]*np.log(num+eps))/sum(weights))


    def score(self, original, candidate):
        if self.select:
            similarity_score = similarity(self.sim_model1, self.sim_model2, original, candidate, select=True)[0]
            bleu, wpd, diversity_score = diversity(original, [candidate])[0]
            numeracy_score = numeracy(original, [candidate])[0]
            overall_score = self.weights['similarity'] * similarity_score + \
            self.weights['diversity'] * diversity_score + self.weights['numeracy'] * numeracy_score
            return similarity_score, bleu, wpd, diversity_score, numeracy_score, overall_score
        else:
            weights = [2, 1, 1]
            similarity_score = similarity(self.sim_model1, self.sim_model2, original, candidate, select=False)[0]
            bleu, wpd, diversity_score = diversity(original, [candidate])[0]
            numeracy_score = numeracy(original, [candidate])[0]
            overall_score = self.weighted_gm(similarity_score, diversity_score, numeracy_score, weights)
            return similarity_score, bleu, wpd, diversity_score, numeracy_score, overall_score
