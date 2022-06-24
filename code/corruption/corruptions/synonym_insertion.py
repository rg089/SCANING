from .corruption import Corruption
from utils import untokenize
from .utils import generate_synonym_parallel, is_part_of
import random
from joblib import Parallel, delayed


class SynonymInsertion(Corruption):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return "Synonym Insertion"

    def __call__(self, text, doc=None, frac=0.2, **kwargs):
        """
        frac represents the fraction of the words to check for synonyms
        Insertion of any synonym is always done within a few tokens of the original word
        """

        tokens = [token for token in doc]
        num_possible = len(tokens)
        num_insertions = int(frac * num_possible)

        chosen_idxs = random.sample(range(num_possible), num_insertions)
        chosen_idxs = [chosen_idx for chosen_idx in chosen_idxs if not is_part_of(str(tokens[chosen_idx]), doc.ents)]
        # selected_insertions = [(idx, generate_synonym(str(tokens[idx]), tokens[idx].pos_)) for idx in chosen_idxs] # Do parallelized version of this

        selected_insertions = Parallel(n_jobs=1)(delayed(generate_synonym_parallel)(str(tokens[idx]), tokens[idx].pos_, idx) for idx in chosen_idxs)
        selected_insertions = [item for item in selected_insertions if item[1] is not None]

        for idx, synonym in selected_insertions:
            insert_position = idx + random.randint(-2, 2)
            insert_position = min(max(0, insert_position), len(tokens))
            # Note: Due to the dynamic insertion, the position may not always be within 
            # a couple of tokens, but this will help the model learn more
            tokens.insert(insert_position, synonym) 

            swap = random.random() < 0.3 # 30% chance of swapping the word with the synonym
            if swap:
                tokens[insert_position] = tokens[idx]
                tokens[idx] = synonym

        tokens = [str(token) for token in tokens]
        text = untokenize(tokens)
        return text