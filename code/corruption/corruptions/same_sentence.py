from .corruption import Corruption


class SameSentence(Corruption):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return 'Same Sentence'

    def __call__(self, text, doc=None, frac=0.4, **kwargs):
        return text
            

same_sentence = SameSentence()