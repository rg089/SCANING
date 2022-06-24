from .utils import sample_span
from .corruption import Corruption
import numpy as np
from utils import untokenize
import random


class RandomShuffle(Corruption):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:     
        return 'Random Shuffling'

    def shuffle(self, tokens, doc):
        span_len = np.random.randint(2, 5) # shuffling either 2, 3 or 4 words
        from_, to = sample_span(tokens, doc, span_len) # Including both from and to
        if from_ < 0:
            return tokens
        sub_array = tokens[from_:to+1]
        random.shuffle(sub_array)
        new_tokens = tokens[:from_] + sub_array + tokens[to+1:]
        return new_tokens
        
    def __call__(self, text, doc, frac=0.2, **kwargs):
        """
        perc represents the fraction of times to shuffle
        """
        tokens = [token for token in doc]
        num_tokens = len(tokens)
        num_shuffles = int(num_tokens * frac)
        for _ in range(num_shuffles):
            tokens = self.shuffle(tokens, doc)
        tokens = list(map(lambda x: x.orth_ if type(x)!=str else x, tokens))
        text = untokenize(tokens)
        return text


random_shuffle = RandomShuffle()