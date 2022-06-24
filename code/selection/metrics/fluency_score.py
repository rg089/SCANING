import encodings
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn
import math
from torch.nn import DataParallel
import re
import random, os
import numpy as np


"""
Code inspired by the lm-scorer library
"""

def seed_everything(seed: int):
    print(f"[INFO] Setting seed to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Fluency:
    def __init__(self, model_name='distilgpt2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, 
                    add_special_tokens=False)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
        self.tokenizer.pad_token = "<|pad|>"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[INFO] Initializing {model_name} for fluency...")
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

        # if torch.cuda.device_count() > 1:
        #     print("[INFO] Using %d GPUs..." % torch.cuda.device_count())
        #     self.model = DataParallel(self.model)

    def tokenize(self, texts):
        texts = [text + self.tokenizer.eos_token for text in texts]
        encoding = self.tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        input_ids = encoding['input_ids']
        nopad_mask = input_ids != self.tokenizer.pad_token_id

        return encoding, input_ids, nopad_mask

    def get_token_log_probs(self, texts):
        encoding, input_ids, nopad_mask = self.tokenize(texts)
        outputs = []

        with torch.no_grad():
            logits = self.model(**encoding)[0]

        for sent_index in range(len(texts)):
            sent_nopad_mask = nopad_mask[sent_index]
            # print(self.tokenizer.tokenize(self.tokenizer.decode(input_ids[sent_index][sent_nopad_mask], skip_special_tokens=False)))
            sent_ids = input_ids[sent_index, sent_nopad_mask][1:] # Not including BOS
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, :] # Right shifting to calculate the probabilities correctly
            sent_logits[:, self.tokenizer.pad_token_id] = float("-inf") # Setting the logits of generating the pad token to -infinity
            # so it doesn't contribute to the sum of the exponent of the logits

            sent_ids_scores = sent_logits.gather(1, sent_ids.unsqueeze(1)).squeeze(1)
            sent_log_probs = sent_ids_scores - sent_logits.logsumexp(1) # Log Probabilities of each token

            sent_log_probs = sent_log_probs.double()
            outputs.append(sent_log_probs)

        return outputs

    def score(self, texts, reduce="gmean", normalize=True):
        """
        Get the fluency scores of the texts
        normalize=True: normalize the scores
        """
        outputs = self.get_token_log_probs(texts)
        scores = []

        for output in outputs:
            log_probs = output
            tlen = log_probs.shape[0]

            if reduce == "prod":
                score = log_probs.sum()
            elif reduce == "mean":
                score = log_probs.logsumexp(0) - math.log(tlen)
            elif reduce == "gmean":
                score = log_probs.mean(0)
            elif reduce == "hmean":
                score = log_probs.neg().logsumexp(0).neg() + math.log(tlen)
            else:
                raise ValueError("Unrecognized scoring strategy: %s" % reduce)
            
            score = score.exp()
            scores.append(score.item())
        
        if normalize:
            scores = [score / sum(scores) for score in scores]

        return scores

    def most_fluent(self, texts, k=1, reduce='mean', normalize=True, lower=True, remove_punct=False):
        """
        Get the k most fluent texts
        returns: list of tuples (index, text, score)
        """
        texts_modified = texts.copy()
        if lower:
            texts_modified = [text.lower() for text in texts_modified]
        if remove_punct:
            texts_modified = [re.sub(r"[,.;@#?!&$]+\ *", " ", text) for text in texts_modified]
        scores = self.score(texts_modified, reduce=reduce, normalize=normalize)
        indices_sorted = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        texts_sorted = [texts[i] for i in indices_sorted]
        scores_sorted = [scores[i] for i in indices_sorted]

        return list(zip(indices_sorted, texts_sorted, scores_sorted))


# seed_everything(6)
fluency = Fluency(model_name="gpt2")