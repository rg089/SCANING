import random
import spacy
spacy.prefer_gpu()
import numpy as np
import os
import benepar

from .corruptions import Active2Passive, sentence_permutation, random_shuffle, \
random_deletion, templatization, same_sentence, synonym_substitutor, word_inserter, \
complete_shuffle, phrase_shuffle, permute_shuffle
from .preprocessing import preprocess
from .utils import load_corruptions, jitter
from utils import load_config


class Corrupter(object):
    def __init__(self, fname='corruptions.json', train=True):
        spacy_model = load_config()['spacy']
        self.nlp = spacy.load(f'en_core_web_{spacy_model}')
        self.nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
        self.active2passive = Active2Passive() # This is required to convert the output
        self.train = train
        self.corruption_combinations, self.probabilities = load_corruptions(fname, train=train)
        assert len(self.corruption_combinations) == len(self.probabilities)
        self.distance = 0.1 if train else 0.02
        print(f"[INFO] CORRUPTING FOR {'TRAIN' if train else 'INFERENCE'}: ")

    def corrupt(self, text, num_augs=5, verbose=False, **kwargs):
        text = preprocess(text)
        doc = self.nlp(text)
        logger = kwargs.get('logger', None)
        kwargs['train'] = self.train
        assert num_augs <= len(self.corruption_combinations)
        if num_augs == -1:
            num_augs = len(self.corruption_combinations)

        chosen_idxs = np.random.choice(len(self.corruption_combinations), num_augs, p=self.probabilities, replace=False)
        corruptions = [self.corruption_combinations[idx] for idx in chosen_idxs]

        corrupted = []
        logs = [] # Adding a separate logging list (will be a list of lists, with each inner list representing a corruption combination)

        for i in range(num_augs): # Looping over the different corruption combinations
            current_corruptions = corruptions[i]
            num_corruptions = len(current_corruptions) // 2
            new_text = text
            new_doc = doc
            current = [] # Used for logging
            current_corruptions_log = [] # Separate logger

            for j in range(num_corruptions): # Looping over the corruptions in a single combination
                corruption_func, frac = current_corruptions[j], current_corruptions[j+num_corruptions]

                frac = jitter(frac, self.distance) # Jittering the fraction
                if type(corruption_func) == str:
                    corruption_func = eval(corruption_func)

                new_text = corruption_func(new_text, doc=new_doc, frac=frac, **kwargs)
                if verbose:
                    print(f"Corruption {j+1}: {corruption_func}, Output: {new_text}")

                current.append(f"{corruption_func}: ({frac})")
                current_corruptions_log.append(str(corruption_func))

                new_doc = self.nlp(new_text)

            corrupted.append(new_text)
            
            current_corruptions_log  = sorted(current_corruptions_log)
            current_corruptions_log_str = ' + '.join(current_corruptions_log)
            logs.append(current_corruptions_log_str)

            if logger is not None: logger.log_current(current)
            if verbose: print()

        return corrupted, logs