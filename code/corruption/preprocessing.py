"""
contraction remover code (from myself) at: https://gist.github.com/rg089/77f5c4ea78e708f174a2a769b4a394f1
"""

import requests
import re


def preprocess(text):
    # Remove contractions
    text = re.sub("[\n\t]", " ", text)
    text = remove_contractions(text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub("\s+", " ", text)
    return text


contractions = requests.get("https://git.io/JzOrp").json()
CONTR = re.compile(f"({'|'.join(contractions.keys())})")
replacer = lambda x: contractions[x.group(0)]
remove_contractions = lambda x: CONTR.sub(replacer, x, re.IGNORECASE|re.DOTALL)
