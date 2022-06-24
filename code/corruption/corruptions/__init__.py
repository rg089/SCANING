from .active_to_passive import Active2Passive
from .sentence_permutation import sentence_permutation
from .random_shuffling import random_shuffle
from .random_deletion import random_deletion
from .templatization import templatization
from .same_sentence import same_sentence
from .synonym_substitution import synonym_substitutor
from .word_insertion import word_inserter
from .complete_shuffle import complete_shuffle
from .shuffle_phrases import phrase_shuffle
from .permute_shuffle import permute_shuffle



import nltk
import spacy
import benepar
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
benepar.download('benepar_en3')