from .corruption import Corruption
import random
from .common_word_insertion import CommonWordInsertion
from .synonym_insertion import SynonymInsertion


class WordInsertion(Corruption):
    def __init__(self) -> None:
        super().__init__()
        self.synonym_insertion = SynonymInsertion()
        self.common_word_insertion = CommonWordInsertion()

    def __str__(self) -> str:
        return "Word Insertion"

    def __call__(self, text, doc=None, frac=0.2, **kwargs):
        """
        Insertion of either a synonym or a common word
        """

        if random.random() < 0.6:
            text = self.synonym_insertion(text, doc, frac, **kwargs)
        else:
            text = self.common_word_insertion(text, doc, frac, **kwargs)
        
        return text


word_inserter = WordInsertion()