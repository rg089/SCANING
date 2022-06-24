from .corruption import Corruption
from utils import untokenize
import random


class CompleteShuffle(Corruption):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:     
        return 'Complete Shuffling'
        
    def __call__(self, text, doc, frac=0.2, **kwargs):
        """
        perc represents the fraction of times to shuffle
        """
        tokens = [str(token) for token in doc]
        random.shuffle(tokens)
        text = untokenize(tokens)
        return text


complete_shuffle = CompleteShuffle()