from .corruption import Corruption
from utils import untokenize
from .utils import sample_position


class RandomDeletion(Corruption):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return "Random Deletion"

    def delete(self, tokens, doc):
        position = sample_position(tokens, doc)
        if position < 0: return tokens
        tokens.pop(position)
        return tokens

    def __call__(self, text, doc, frac=0.2, **kwargs):
        """
        Randomly delete words from the text keeping in mind the leniency constraints
        """
        tokens = [token for token in doc]
        num_tokens = len(tokens)
        num_deletions = int(num_tokens * frac)
        for _ in range(num_deletions):
            tokens = self.delete(tokens, doc)
        tokens = list(map(lambda x: x.orth_ if type(x)!=str else x, tokens))
        text = untokenize(tokens)
        return text

random_deletion = RandomDeletion()