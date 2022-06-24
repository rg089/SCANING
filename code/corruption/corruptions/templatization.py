from re import S
from templatization import Templatizer
from .corruption import Corruption
from .utils import pos_tags_preserve, stemmed_units


class Templatize(Corruption):
    def __init__(self) -> None:
        super().__init__()
        self.templatizer = Templatizer()
        self.tags = {"pos": [], "neg": list(pos_tags_preserve)}
        self.keep_words = stemmed_units
        self.coreference = True


    def __str__(self) -> str:
        return "Templatization"

    def __call__(self, text, doc=None, frac=0.5, **kwargs) -> str:
        train = kwargs.get("train", False)
        objects_always = False if train else True
        preserve_last_n = kwargs.get("preserve_last_n", 5)
        mapper= {"person": {}, "place": {}, "pos": {}, "num": {}}
        sent, _ = self.templatizer.templatize(text, mapper=mapper, tags=self.tags, frac=frac,
                        keep_words=self.keep_words, coreference=self.coreference, preserve_last_n=preserve_last_n,
                        objects_always=objects_always)
        return sent


templatization = Templatize()