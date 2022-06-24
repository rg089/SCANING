from pickle import NONE
import re, random, os
import numpy as np
import json
from nltk.corpus import wordnet as wn
import spacy
spacy.prefer_gpu()
from gensim.models import KeyedVectors
from similarity.normalized_levenshtein import NormalizedLevenshtein
import random
from utils import load_config
import wordtodigits
import string
from nltk.stem import PorterStemmer


# Print path of current file
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
preserve_last_n_words = 4
units = json.load(open(os.path.join(CURRENT_PATH, "..", "helper", "UnitList.json"))) + ["not"]
stemmed_units = json.load(open(os.path.join(CURRENT_PATH, "..", "helper", "UnitStemmed.json")))
pos_tags_preserve = set(["VERB", "ADJ", "NUM", "PROPN", "ADV", "PUNCT"])  # POS tags to preserve
common_words = json.load(open(os.path.join(CURRENT_PATH, "..", "helper", "common_words.json")))
nl = NormalizedLevenshtein()
ps = PorterStemmer()

spacy_model = load_config()['spacy']
nlp = spacy.load(f'en_core_web_{spacy_model}')

glove_datapath = os.path.join(CURRENT_PATH, "..", "..", "data", "glove.6B.300d.txt")
print("[INFO] Loading GloVe model for synonyms...")
glove_model = KeyedVectors.load_word2vec_format(glove_datapath, binary=False, no_header=True)

ppdb = json.load(open(os.path.join(CURRENT_PATH, "..", "..", "data", "ppdb-xl-lexical-processed.json"), "r"))


def is_part_of(token, wordlist):
    token1 = str(token).lower()
    for word in wordlist:
        word = str(word).lower()
        if len(word) <= 2 or len(token1) <= 2:
            if token1 == word: return True
        elif token1 in word or word in token1:
            return True
    return False
    

def check_ner_pos(token, doc):
    """
    returns true if the token is in the pos/ner list that cannot be replaced
    """
    ne = doc.ents
    if is_part_of(token, ne): return True
    # if type(token) == str: return False # Only possible if the token is a mask
    if token.pos_ in pos_tags_preserve: return True
    return False


def isChangeable(token, doc=None, check_pos=True):
    """
    returns a boolean indicationg whether the token can be changed (not numeric or unit)
    """
    token1 = str(token)
    token1 = wordtodigits.convert(token1)
    stemmed = ps.stem(token1)
    if re.search("\d+", token1) or is_part_of(stemmed, stemmed_units) or is_part_of(token1, units):
        return False
    if re.search(r"[A-Z]+\d+", token1) is not None: return False
    if check_pos and check_ner_pos(token, doc): return False
    return True


def sample_span(tokens, doc, span_len):
    candidates = []
    for idx, token in enumerate(tokens):
        if idx + span_len - 1 >= len(tokens) - preserve_last_n_words: continue # Preserving the last few words, so that the qn doesnt change
        text = ' '.join(list(map(lambda x: str(x), tokens[idx:idx+span_len])))
        if idx + span_len - 1 < len(tokens) and re.search("\d+", text) is None:
            candidates.append((idx, idx+span_len-1))
    if len(candidates) <= 0:
        return -1, -1
    return random.choice(candidates)


def sample_position(tokens, doc):
    candidates = [idx for idx in range(len(tokens)-preserve_last_n_words)]
    if len(candidates) == 0:
        return -1

    pointer = 0
    choices = np.random.choice(candidates, 2*len(candidates))
    idx = choices[pointer]
    while (pointer < len(choices)-1 and not isChangeable(tokens[idx], doc)):
        pointer += 1
        idx = choices[pointer]
    if isChangeable(tokens[idx], doc): return idx
    else: return -1


def map_wordnet_tags(tag):
    pos_wordnet_dict = {
        "VERB": "v",
        "NOUN": "n",
        "ADV": "r",
        "ADJ": "s"}
    return pos_wordnet_dict.get(tag, None)


def filter_same(word, similar, nl=NormalizedLevenshtein()):
    output = []
    for w, score in similar:
        if w.lower() == word.lower(): continue
        if w.lower() in word.lower(): continue
        if word.lower() in w.lower(): continue

        dist = nl.distance(word, w)
        if dist < 0.2: continue
        output.append((w, score))
        
    return output


def keep_same_pos(word, pos, similar):
    output = []
    for w, score in similar:
        pos_ = [token.pos_ for token in nlp(w)][0]
        if pos_ == pos: output.append((w, score))
    return output


def get_k_most_similar(word, pos=None):
    try:
        similar = glove_model.most_similar(word, topn=10)
        similar_filtered = filter_same(word, similar)
        similar_pos = similar_filtered

        if pos is not None:
            similar_pos = keep_same_pos(word, pos, similar_filtered)

        if len(similar_pos) == 0:
            if len(similar_filtered) == 0:
                similar_pos = similar
            else:
                similar_pos = similar_filtered

        if similar_pos[0][1] < 0.5: return None # No good synonym found
        k = [i for i in range(len(similar_pos)) if similar_pos[i][1] < 0.5] # Get the indexes where the score is less than 0.45

        if len(k) == 0: 
            return similar_pos
        return similar_pos[:k[0]]

    except:
        return None


def get_common_graph(doc):
    """
    returns a dictionary of type {entity: [idx1, idx2....]}
    """
    mapper = {}
    for idx, token in enumerate(doc):
        if token.pos_ not in ["PROPN", "NOUN", "VERB"]:
            continue
        lemmatized = token.lemma_
        if lemmatized not in mapper:
            mapper[lemmatized] = []
        mapper[lemmatized].append(idx)
    return mapper

        
def get_synonym_glove(word, pos=None):
    word_ = word.lower()
    capitalized = word[0].isupper()
    similar = get_k_most_similar(word_, pos)
    if similar is None: return None
    choice = random.choice(similar)[0]
    if capitalized: return choice.capitalize()
    return choice


def get_synonym_ppdb(word, pos=None):
    word_ = word.lower()
    capitalized = word[0].isupper()
    stemmed = ps.stem(word_)

    word = ppdb.get(word_, None)
    if word is None: return None

    similar_list = word["synonyms"]

    similar = similar_list[:len(similar_list)//3+1]
    similar = [x for x in similar if nl.distance(word_, x) > 0.2 and ps.stem(x) != stemmed]

    if len(similar) == 0: return None

    choice = random.choice(similar)
    if capitalized: return choice.capitalize()
    return choice

    
def generate_synonym_model(word, model='glove', pos=None):
    """
    Generate a synonym for a word using WordNet or PPDB.
    If synonym is not found, return None.
    """
    if word in string.punctuation: return None
    if not isChangeable(word, check_pos=False): return None

    if model == 'wordnet':
        if pos is None: return None
        wordnet_pos = map_wordnet_tags(pos)
        if wordnet_pos is None:
            return None # If tag is unknown, then return None

        synsets = wn.synsets(word, pos=wordnet_pos)
        if len(synsets) > 0:
            synsets = [syn.name().split(".")[0] for syn in synsets]
            synsets = [syn for syn in synsets if syn.lower() != word.lower()]
            synsets = list(set(synsets)) # Remove duplicates
            if len(synsets) > 0:
                syn = random.choice(synsets[:2])
                syn = syn.replace("_", " ")
                return syn

    elif model == 'ppdb':
        return get_synonym_ppdb(word, pos)

    elif model == 'glove':
        return get_synonym_glove(word, pos)

    return None


def generate_synonym(word, pos=None):
    """
    Generate a synonym for a word using PPDB, GloVe and WordNet.
    If synonym is not found, return None.
    """
    synonym = generate_synonym_model(word, 'ppdb', pos)
    if synonym is not None: return synonym

    synonym = generate_synonym_model(word, 'glove', pos)
    if synonym is not None: return synonym

    synonym = generate_synonym_model(word, 'wordnet', pos)
    if synonym is not None: return synonym

    return None

def generate_synonym_parallel(token, pos=None, idx=0):
    """
    Generate a synonym for a word using PPDB, GloVe and WordNet.
    If synonym is not found, return None.
    Also returns the index speciified for parallelism
    """
    word = str(token)

    synonym = generate_synonym_model(word, 'ppdb', pos)
    if synonym is not None: return idx, synonym

    synonym = generate_synonym_model(word, 'glove', pos)
    if synonym is not None: return idx, synonym

    synonym = generate_synonym_model(word, 'wordnet', pos)
    if synonym is not None: return idx, synonym

    return None, None