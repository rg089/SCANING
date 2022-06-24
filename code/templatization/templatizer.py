import numpy as np
import re
import spacy
from utils import untokenize, load_config
import random
from nltk.stem import PorterStemmer


def is_part_of(word_, wordlist):
    word_ = str(word_)
    words = word_.split()
    for word in wordlist:
        # print(word, words)
        try:
            if word.split()[0].lower() == words[0].lower(): return word
        except:
            print(word, words)
    return ""


def is_part_of_2(token, wordlist):
    token1 = str(token).lower()
    for word in wordlist:
        word = str(word).lower()
        if len(word) <= 2 or len(token1) <= 2:
            if token1 == word: return True
        elif token1 in word or word in token1:
            return True
    return False


class Templatizer:
    def __init__(self):
        spacy_model = load_config()["spacy"]
        self.nlp = spacy.load(f"en_core_web_{spacy_model}")
        self.ps = PorterStemmer()

    def mask_nums(self, sent, nums):
        numbers = re.findall("\d+\.?\d*", sent)
        for num in numbers:
            if num not in nums:
                rep = f"NUM{len(nums)+1}"
                nums[num] = rep
            else:
                rep = nums[num]
            sent = re.sub(f"(^|\W)({num})($|\W)", f" {rep} ", sent)
        sent = re.sub("\s+", " ", sent).strip()
        return sent


    def mask_ne(self, sent, doc, persons, places):
        for ent in doc.ents:
            if ent.label_ not in ["PERSON", "GPE"]: continue
            if ent.label_ == "PERSON": 
                rep = is_part_of(ent, persons.keys())
                if rep == "": 
                    rep = f"PERSON{len(persons)+1}"
                    persons[str(ent)] = rep
                else:
                    rep = persons[rep]
            else:
                rep = is_part_of(ent, places.keys())
                if rep == "": 
                    rep = f"PLACE{len(places)+1}"
                    places[str(ent)] = rep
                else:
                    rep = places[rep]
            sent = sent.replace(str(ent), rep)
        return sent


    def mask_pos(self, sent, doc, pos, tags, frac=1, keep_words=[], coreference=True, preserve_last_n=0, 
                    objects_always=False):
        """
        frac denotes the fraction of eligible tokens to be masked if objects_always is False
        if objects_always is True, all objects are masked and frac represents the fraction of remaining tokens to be masked
        """
        tokens = []
        for token in doc:
            tokens.append(str(token))

        new_tokens = tokens.copy()
        change_idxs = []
        change_idxs_objects = []

        # Find the indices of tokens that can't be masked
        for idx, token in enumerate(doc):
            if idx >= len(tokens) - preserve_last_n: break

            stemmed = self.ps.stem(str(token))
            if is_part_of_2(stemmed, keep_words): continue
            
            tag = token.pos_

            if objects_always:
                if "obj" in token.dep_ and tag not in tags["neg"]:
                    change_idxs_objects.append(idx)
                    continue

            if tag == "PUNCT": continue
            if len(tags["pos"]) > 0 and len(tags["neg"]) > 0: raise ValueError("Cannot have both pos and neg tags")
            if len(tags["pos"]) == 0 and tag in tags["neg"]: continue
            if len(tags["pos"]) > 0 and tags["pos"][0].lower() != "all" and \
                tag not in tags["pos"]: continue # If the tag is not the one we want

            change_idxs.append(idx) # The token at this index can be masked

        
        change_idxs_final = change_idxs_objects
        total_changes = round((len(change_idxs_objects) + len(change_idxs)) * frac)
        # print(len(change_idxs_objects), len(change_idxs), total_changes)

        if total_changes <= len(change_idxs_objects):
            change_idxs_final = random.sample(change_idxs_objects, k=total_changes)
        else:
            change_idxs_final = change_idxs_objects
            changes_left = total_changes - len(change_idxs_objects)
            change_idxs_final +=  random.sample(change_idxs, k=changes_left)
        
        change_idxs_final = set(change_idxs_final)

        # Mask the tokens
        for idx, token in enumerate(doc):
            if idx not in change_idxs_final: continue

            tag = token.pos_
            if tag in pos: 
                rep = is_part_of(str(token), pos[tag].keys()) # If that tag has been encountered before
            else: 
                rep = ""
                pos[tag] = {}

            if rep == "":  # The token hasn't been seen before
                if coreference: rep = f"{tag}{len(pos[tag])+1}"
                else: rep = tag
                pos[tag][str(token)] = rep
            else: 
                rep = pos[tag][rep]

            new_tokens[idx] = rep

        sent = untokenize(new_tokens)
        return sent


    def templatize(self, sentence, mapper= {"person": {}, "place": {}, "pos": {}, "num": {}}, 
                    tags={"pos": ["ALL"], "neg": []}, frac=1, keep_words=[], coreference=True,
                    preserve_last_n=0, objects_always=False):
        """
        returns the templatized sentence and the template mapping dictionary
        mapper: {"person": {}, "place": {}, "pos": {}, "num": {}}
        keep_words: a list of words that should not be masked (case insensitive)
        """
        persons, places, nums, pos = mapper['person'], mapper['place'], mapper['num'], mapper['pos']
        sent = sentence
        doc = self.nlp(sent)

        # sent = self.mask_nums(sent, nums)
        # sent = self.mask_ne(sent, doc, persons, places)
        sent = self.mask_pos(sent, doc, pos, tags, frac=frac, keep_words=keep_words, coreference=coreference,
                            preserve_last_n=preserve_last_n, objects_always=objects_always)

        return sent, mapper