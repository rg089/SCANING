from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from paraphrase_metrics import metrics as pm
import spacy
import numpy as np


nlp = spacy.load("en_core_web_md")


def get_wpd(ref, hyp):
    # sent_ref = list(ref.sents)
    # sent_hyp = list(hyp.sents)

    # if len(sent_ref) != len(sent_hyp):
    return pm.wpd(ref, hyp)

    # wpds = []
    # for ref_sent, hyp_sent in zip(sent_ref, sent_hyp):
    #     wpds.append(pm.wpd(ref_sent, hyp_sent))

    # return np.mean(wpds)


def diversity_single(inp, can, use_wpd=True, reduce='mean', w1=1, w2=1):
    """
    w1 is for the bleu diversity and w2 is for the wpd
    """
    smoothie = SmoothingFunction().method4

    inp = inp.lower()
    can = can.lower()

    eps = 1e-9
    ref = nlp(inp); hyp = nlp(can)
    if use_wpd:
        wpd = get_wpd(ref, hyp)

    ref = [str(token) for token in ref]
    hyp = [str(token) for token in hyp]

    bleu_div = 1-sentence_bleu([ref], hyp, weights=(0.45, 0.35, 0.2), smoothing_function=smoothie)
    if not use_wpd:
        return bleu_div

    reduced = 0
    if reduce == 'mean': # Take the weighted arithmetic mean
        reduced = (w1*bleu_div + w2*wpd)/(w1+w2)
    elif reduce == 'gmean': # Take the geometric mean
        reduced = np.exp(w1*np.log(bleu_div+eps) + w2*np.log(wpd+eps)/(w1+w2))
    elif reduce == 'hmean': # Take the harmonic mean
        reduced = ((w1+w2)*bleu_div*wpd+eps)/(w1*wpd+w2*bleu_div+eps)
    else:
        raise NotImplementedError(f"Reduce method {reduce} not implemented")

    return (bleu_div, wpd, reduced)

def diversity(original, candidates, use_wpd=True, reduce='mean', w1=0.6, w2=0.4):
    return [diversity_single(original, can, use_wpd=use_wpd, reduce=reduce, w1=w1, w2=w2) for can in candidates]