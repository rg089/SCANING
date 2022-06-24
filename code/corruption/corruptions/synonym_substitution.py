from .corruption import Corruption
from .utils import generate_synonym_parallel, is_part_of, get_common_graph
from utils import untokenize
import random
from joblib import Parallel, delayed


class SynonymSubstitution(Corruption):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return 'Synonym Substitution'

    def __call__(self, text, doc=None, frac=0.4, **kwargs):
        """
        frac represents the fraction of the total words to substitute for synonyms
        """
        train = kwargs.get('train', False)
        tokens = [token for token in doc]
        possible_indices = [idx for idx in range(len(tokens)) if not is_part_of(str(tokens[idx]), doc.ents)]

        mapper = get_common_graph(doc)

        num_possible = len(possible_indices)
        num_substitutions = int(frac * num_possible)
        obj_indices = []
        
        if not train:
            for idx, token in enumerate(tokens):
                if "obj" in token.dep_ and idx in possible_indices:
                    obj_indices.append(idx)

            if num_substitutions > len(obj_indices): # If the number of substitutions is greater than the number of objects, we need to randomly select some
                num_substitutions -= len(obj_indices)
                left = [idx for idx in possible_indices if idx not in obj_indices] # The indices left
                chosen_idxs = obj_indices
                if len(left) >= num_substitutions: # If there are enough indices left, we can just randomly select else it will throw an error
                    chosen_idxs += random.sample(left, num_substitutions)
                else:
                    chosen_idxs += left

            else:
                chosen_idxs = random.sample(obj_indices, num_substitutions)

        else:
            chosen_idxs = random.sample(possible_indices, num_substitutions)

        selected_substitutions = Parallel(n_jobs=1)(delayed(generate_synonym_parallel)(tokens[idx], tokens[idx].pos_, idx) for idx in chosen_idxs)
        selected_substitutions = [item for item in selected_substitutions if item[1] is not None]

        replaced = set([])
        for idx, synonym in selected_substitutions:
            if idx in replaced: continue

            token = tokens[idx]
            tokens[idx] = synonym
            replaced.add(idx)

            if type(token) != str and token.lemma_ in mapper:
                for idx_ in mapper[token.lemma_]:
                    tokens[idx_] = synonym
                    replaced.add(idx_)

        tokens = [str(token) for token in tokens]
        text = untokenize(tokens)
        return text


synonym_substitutor = SynonymSubstitution()