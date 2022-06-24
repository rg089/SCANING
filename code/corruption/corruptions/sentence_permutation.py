import re
from .corruption import Corruption
from nltk.tokenize import sent_tokenize, word_tokenize
from utils import untokenize
from .utils import is_part_of
import numpy as np
import random, string
from selection.metrics import fluency


class SentencePermutation(Corruption):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return 'Sentence Permutation'

    def permute_sentence_index(self, words, doc, permutation_point):
        punct = words[-1] in string.punctuation
        start_word = words[0]
        if not re.findall(r'[A-Z]+\d+', start_word): # The start word is not templatized as we don't want to lower a templated word.
            entities = [str(token) for token in doc.ents]
            if not is_part_of(start_word, set(entities)): # Don't lower the start word if it is a named entity.
                words[0] = words[0].lower() # Make sure the first word is lowercase as it gets permuted to the middle
        if punct:
            words = words[permutation_point+1:-1] + words[:permutation_point+1] + words[-1:] # Permute the sentence, keeping the punctuation intact.
        else:
            words = words[permutation_point+1:] + words[:permutation_point+1]
        if not words[0].isupper(): words[0] = words[0].capitalize() # Capitalize the first word to mak sure the model doesnt just copy.
        return untokenize(words)

    def permute_sentence_train(self, sentence, doc):
        words = [str(token) for token in sentence]
        num_words = len(words)
        low = 2; high = num_words - 2
        if low >= high: return str(sentence)
        permutation_point = np.random.randint(low, high) # Indice after which the sentence is permuted. Only permuted near the middle.

        return self.permute_sentence_index(words, doc, permutation_point)

    def permute_sentences_inference(self, sentence, doc):
        words = [str(token) for token in sentence]
        prepositions = [] # indices after which prepositions occur

        for idx, token in enumerate(sentence):
            if token.pos_ == "ADP" or token.pos_ == "SCONJ":
                prepositions.append(max(0, idx-1))

        if len(prepositions) > 0:
            chosen_idx = random.sample(prepositions, 1)[0]
            output = self.permute_sentence_index(words, doc, chosen_idx)
        else:
            possibilities = []
            for i in range(0, len(words)-2):
                sent = self.permute_sentence_index(words, doc, i)
                possibilities.append(sent)

            if len(possibilities) > 0:
                output = fluency.most_fluent(possibilities, k=1, normalize=True)[0][1]
            else:
                output = untokenize(words)

        return output

    def permute_sentence(self, sentence, doc, train=True):
        if train:
            return self.permute_sentence_train(sentence, doc)
        else:
            return self.permute_sentences_inference(sentence, doc)
         
    def __call__(self, text, doc=None, frac=0.4, **kwargs):
        train = kwargs.get("train", True)
        sentences = list(doc.sents)
        num_sentences = len(sentences)
        num_to_permute = max(1, round(num_sentences * frac)) # Permute at least 1 sentence
        indices = np.random.choice(num_sentences, num_to_permute, replace=False) # Indices of sentences to permute.
        new_sentences = []
        for idx, sentence in enumerate(sentences):
            if idx in indices:
                new_sentence = self.permute_sentence(sentence, doc, train=train)
            else:
                new_sentence = str(sentence)
            new_sentences.append(new_sentence)
        return ' '.join(new_sentences)
            
sentence_permutation = SentencePermutation()