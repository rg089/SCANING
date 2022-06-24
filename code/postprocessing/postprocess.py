from simplejson import load
from .utils import load_common_data
import spacy
import numpy as np
from utils import load_config


class PostProcessor:
    def __init__(self):
        spacy_model = load_config()["spacy"]
        self.nlp = spacy.load(f"en_core_web_{spacy_model}")
        self.mn, self.fn, self.places = load_common_data()

    def sample_element(self, ent, wordlist):
        n = len(wordlist)
        idx = np.random.randint(n)
        while wordlist[idx].lower() == ent.lower():
            idx = np.random.randint(n)
        return wordlist[idx]

    def find_replacement(self, ent, wordlist):
        """
        checks wheteher ent is present in the wordlist.
        if it is, returns a random element from the wordlist, else returns None
        """
        words = ent.split()
        for word in wordlist:
            if word.lower() == words[0].lower():
                return self.sample_element(words[0], wordlist)
        return None

    def replace_entity(self, ent, type_, replacement):
        ent = str(ent); type_ = str(type_)
        if ent in replacement:  # If replacement is already cached in memory
            return ent, replacement[ent]
        else:
            if type_ == "PERSON":
                rep = self.find_replacement(ent, self.mn)
                if rep is None: rep = self.find_replacement(ent, self.fn) 
                ent = ent.split()[0]
                if rep is None: rep = ent
                replacement[ent] = rep

            elif type_ == "GPE":
                rep = self.find_replacement(ent, self.places)
                if rep is None: rep = ent
                replacement[ent] = rep

            else: 
                rep = ent
                
            return ent, rep

    def process(self, sentence):
        replacements = {}
        sent = sentence

        doc = self.nlp(sentence)
        for ent in doc.ents:
            ent, rep = self.replace_entity(ent.text, ent.label_, replacements)
            sent = sent.replace(ent, rep)

        return sent