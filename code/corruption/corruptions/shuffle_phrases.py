from .utils import sample_span
from .corruption import Corruption
import numpy as np
from utils import untokenize
import random
import spacy, benepar
from selection.metrics import fluency
from .random_shuffling import random_shuffle

class PhraseShuffle(Corruption):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:     
        return 'Phrase Shuffling'

    def format_properly(self, tokens):
        tokens = [str(token) for token in tokens]
        tokens[0] = tokens[0].capitalize() if not tokens[0].isupper() else tokens[0]

        sent = untokenize(tokens)
        return sent    

    def swap_preposition(self, sent):
        """
        Given the tree, make PP the first child of its parent node using recursion
        returns a list of children and a boolean indicating whether found
        """
        children = []
        found_idx = -1
        found = False

        if len(list(sent._.children)) == 0: # Base case for recursion
            return [sent], False

        for idx, child in enumerate(sent._.children):
            if "PP" in child._.labels and not found:
                found_idx = idx
                found = True

            children.append(child)

        if found_idx != -1:
            child = children[found_idx]
            del children[found_idx]
            children.insert(0, child)


        else:
            children = []
            for child in sent._.children:
                if not found:
                    child_1, found = self.swap_preposition(child)
                    children += child_1
                else:
                    children.append(child)

        return children, found

    def swap_cc(self, sent):
        """
        Given the tree, swap the nodes occuring before and after CC
        returns a list of children and a boolean indicating whether found
        """
        children = []
        found_idx = -1
        found = False

        if len(list(sent._.children)) == 0: # Base case for recursion
            return [sent], False

        for idx, child in enumerate(sent._.children):
            labels = child._.labels
            if ("CC" in labels or (len(labels) == 0 and "CC" in child[0].pos_)) and not found: # added the or part due to a bug in benepar
                found_idx = idx
                found = True

            children.append(child)

        if found_idx != -1:
            if found_idx > 0 and found_idx < len(children)-1:
                child_1 = children[found_idx-1]
                child_2 = children[found_idx+1]
                children[found_idx+1], children[found_idx-1] = child_1, child_2
            else:
                found = False
                found_idx = -1


        if not found:
            children = []
            for child in sent._.children:
                if not found:
                    child_1, found = self.swap_cc(child)
                    children += child_1
                else:
                    children.append(child)

        return children, found

    def shuffle_verbs(self, sent):
        """
        if multiple verbs found, shuffles them
        returns a list of tokens and a boolean indicating whether found
        """
        found = False
        num_verbs = sum(1 for token in sent if token.pos_ == "VERB")
        tokens = [token for token in sent]
        
        if num_verbs < 2:
            return tokens, found
        
        first_idx = -1; second_idx = -1
        for idx, token in enumerate(tokens):
            if token.pos_ == "VERB" and first_idx == -1:
                first_idx = idx
            elif token.pos_ == "VERB" and first_idx != -1:
                second_idx = idx
                break

        tokens = tokens[:first_idx] + tokens[second_idx:] + tokens[first_idx:second_idx]
        found = True

        return tokens, found

    def swap_p1_p2(self, sent, p1="NP", p2="VP", diff=2):
        """
        Given the tree, swap p1 and p2 sibling nodes if present
        diff represents the maximum difference between the number of children of p1 and p2
        returns a list of children and a boolean indicating whether found
        """
        children = []
        p1_idx, p2_idx = -1, -1
        num_c1 = 0; num_c2 = 0
        found = False

        if len(list(sent._.children)) == 0: # Base case for recursion
            return [sent], False

        for idx, child in enumerate(sent._.children):
            if p1 in child._.labels and p1_idx == -1:
                p1_idx = idx
                num_c1 = len(list(child._.children))

            elif p2 in child._.labels and p2_idx == -1:
                p2_idx = idx
                num_c2 = len(list(child._.children))

            children.append(child)
        
        if p1_idx != -1 and p2_idx != -1 and abs(num_c1 - num_c2) <= diff:
            children[p1_idx], children[p2_idx] = children[p2_idx], children[p1_idx]
            found = True

        if not found:
            children = []
            for child in sent._.children:
                if not found:
                    child_1, found = self.swap_p1_p2(child, p1, p2)
                    children += child_1
                else:
                    children.append(child)

        return children, found

    def swap_np_vp(self, sent, diff=3):
        return self.swap_p1_p2(sent, p1="NP", p2="VP", diff=diff)


    def shuffle(self, sent):
        possibilities = []
        
        # Swap prepositions
        children, found = self.swap_preposition(sent)
        if found:
            output = self.format_properly(children)
            possibilities.append(output)

        # Swap CC
        children, found = self.swap_cc(sent)
        if found:
            output = self.format_properly(children)
            possibilities.append(output)

        # Swap verbs
        tokens, found = self.shuffle_verbs(sent)
        if found:
            output = self.format_properly(tokens)
            possibilities.append(output)

        if len(possibilities) > 1:
            fluent = fluency.most_fluent(possibilities, k=1, normalize=True)
            output = fluent[0][1]
            return output
        elif len(possibilities) == 1:
            output = possibilities[0]
            return output

        # Swap NP and VP
        children, found = self.swap_np_vp(sent, diff=2)
        if not found:
            children, found = self.swap_np_vp(sent, diff=3)
            if not found:
                children, found = self.swap_np_vp(sent, diff=4)

        if found:
            output = self.format_properly(children)
            possibilities.append(output)

        if len(possibilities) == 0:
            return str(sent)
        else:
            output = possibilities[0]

        return output

        
    def __call__(self, text, doc, frac=0.5, **kwargs):
        """
        frac represents the fraction of sentences to shuffle
        """
        sents = list(doc.sents)
        num_sents = len(sents)
        num_shuffles = max(1, round(num_sents * frac))

        shuffle_idxs = random.sample(range(num_sents), num_shuffles)

        output = []
        shuffled = False
        for idx, sent in enumerate(sents):
            if idx in shuffle_idxs:
                sent_ = self.shuffle(sent)
                sent_ = str(sent_)

                if sent_.lower() != str(sent).lower(): # Checking whether shuffling has happened
                    shuffled = True

                output.append(sent_)
            else:
                sent = str(sent)
                output.append(sent)

        if not shuffled: # If not shuffled even once
            return random_shuffle(text, doc, frac=0.3, **kwargs)

        text = untokenize(output)
        return text


phrase_shuffle = PhraseShuffle()