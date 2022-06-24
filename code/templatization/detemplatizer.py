from nltk.tokenize import word_tokenize
from utils import untokenize
import re
from copy import deepcopy


def reverse_mapper(mapper):
    rev_map = mapper.copy()
    pos = mapper['pos']
    pos_ = rev_map['pos']
    for tag in pos:
        tag_dict = pos[tag].copy()
        tag_dict = {v:k for k,v in tag_dict.items()}
        pos_[tag] = tag_dict
    return rev_map


def fill(template, mapper):
    tokens = word_tokenize(template)
    pos = mapper['pos']
    for idx, token in enumerate(tokens):
        tag_l = re.findall(r"[A-Z]+", token)
        if not tag_l: continue
        tag = tag_l[0]
        
        if tag not in pos: continue
        if token not in pos[tag]: continue
            
        rep = pos[tag][token]
        tokens[idx] = rep
        
    filled = untokenize(tokens)
    return filled


def convert2clean(template):
    tokens = word_tokenize(template)
    for idx, token in enumerate(tokens):
        match = re.findall(r"[A-Z]+\d{1,2}", token)
        if match and match[0] == token:
            tokens[idx] = re.findall(r"[A-Z]+", token)[0]
            
    cleaned = untokenize(tokens)
    return cleaned


def detemplatize(template, mapper_):
    mapper = deepcopy(mapper_)
    rev_mapper = reverse_mapper(mapper)
    filled = fill(template, rev_mapper)
    cleaned = convert2clean(filled)
    return cleaned