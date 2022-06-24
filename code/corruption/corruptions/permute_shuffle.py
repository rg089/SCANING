import re
from .corruption import Corruption
from nltk.tokenize import sent_tokenize, word_tokenize
from .sentence_permutation import sentence_permutation
from .shuffle_phrases import phrase_shuffle
import numpy as np
from selection.metrics import fluency


class PermuteShuffle(Corruption):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return 'Sentence Permutation + Phrase Shuffle'
         
    def __call__(self, text, doc=None, frac=0.4, **kwargs):
        train = kwargs.get("train", False)
        sentences = list(doc.sents)
        num_sentences = len(sentences)
        if num_sentences ==0: return text

        num_to_change = max(1, round(num_sentences * frac)) # Permute at least 1 sentence
        indices = np.random.choice(num_sentences, num_to_change, replace=False) # Indices of sentences to permute.

        new_sentences = []
        for idx, sentence in enumerate(sentences):
            if idx in indices:
                new_sentence1 = str(sentence_permutation.permute_sentence(sentence, doc, train=train))
                new_sentence2 = str(phrase_shuffle.shuffle(sentence))
                sentences = [new_sentence1, new_sentence2]
                new_sentence = fluency.most_fluent([new_sentence1, new_sentence2], k=1, normalize=True)[0][1]
            else:
                new_sentence = str(sentence)
            new_sentences.append(new_sentence)
        return ' '.join(new_sentences)

         
permute_shuffle = PermuteShuffle()