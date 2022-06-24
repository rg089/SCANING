from .corruption import Corruption
from utils import untokenize
import random
from .utils import common_words


class CommonWordInsertion(Corruption):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return "Common Word Insertion"

    def __call__(self, text, doc=None, frac=0.2, **kwargs):
        """
        frac represents the fraction of the words to check for synonyms
        Insertion of any synonym is always done within a few tokens of the original word
        """

        tokens = [token for token in doc]
        num_possible = len(tokens)
        num_insertions = int(frac * num_possible)

        chosen_idxs = random.sample(range(num_possible), num_insertions)
        chosen_words = random.choices(common_words, k=num_insertions)
        
        selected_insertions = [(idx, word) for idx, word in zip(chosen_idxs, chosen_words)]

        for idx, word in selected_insertions:
            insert_position = idx + random.randint(-2, 2)
            insert_position = min(max(0, insert_position), len(tokens))
            tokens.insert(insert_position, word) 

        tokens = [str(token) for token in tokens]
        text = untokenize(tokens)
        return text