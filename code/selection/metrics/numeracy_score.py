import wordtodigits as wtd
import re
import string
from collections import Counter


def get_number_list(s):
    s = wtd.convert(s)
    nums = re.findall('(\d+\.\s\d*|\d+)', s) # Adding adjustment to regex as wtd converts decimals like 15.2 to 15. 2 for some reason
    nums = [num.strip(".,").replace(" ", "") for num in nums]
    nums = [float(num) for num in nums]
    return nums

def cntFrequency(lst1, lst2):
    dct=dict(Counter(lst1))
    sub_dct={k:dct.get(k,0) for k in lst2}
    return sum(sub_dct.values())

def number_match(s1, s2):
    s1 = get_number_list(s1)
    s2 = get_number_list(s2)
    if len(s1) == 0 and len(s2) == 0:
        return 1
    intersection = min(cntFrequency(s1, s2), cntFrequency(s2, s1)) # Can't use set intersection due to duplicacy cases
    max_nums = max(len(s1), len(s2)) # Not using union as it penalizes hallucination more than deletion
    iou = intersection/max_nums
    return iou**3


def numeracy(original, candidates=[]):
    scores = [number_match(original, candidate) for candidate in candidates]
    return scores