"""
some of the code adapted from: https://github.com/PrithivirajDamodaran/Styleformer
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from .corruption import Corruption
from nltk.tokenize import sent_tokenize
from torch.nn import DataParallel

class StyleChange():
    def __init__(self, num_beams=5, max_length=32, quality_filter=0.95):
        super().__init__()
        m_name = "prithivida/active_to_passive_styletransfer"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading Styleformer on {self.device}....")
        print(f"[INFO] Styleformer Loaded!")
        self.tokenizer = AutoTokenizer.from_pretrained(m_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(m_name).to(self.device)
        # if torch.cuda.device_count() > 1:
        #     print("[INFO] Using %d GPUs..." % torch.cuda.device_count())
        #     self.model = DataParallel(self.model)
        self.max_output = num_beams
        self.num_beams = num_beams
        self.max_length = max_length
        self.quality_filter = quality_filter
        self.model.eval()

    def generate(self, sentence: str):
        with torch.no_grad():
            ctf_prefix = "transfer Active to Passive: "
            sentence = ctf_prefix + sentence
            input_ids = self.tokenizer.encode(sentence, return_tensors="pt")
            input_ids = input_ids.to(self.device)

            preds = self.model.generate(
                input_ids,
                num_beams=3,
                max_length=128, 
                early_stopping=True,
                num_return_sequences=1
            )

            return self.tokenizer.decode(preds[0], skip_special_tokens=True).strip()


class Gramformer:
    def __init__(self):    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")      
        self.device = device
        self.backup = False 

        self.backup = True
        name = "zuu/grammar-error-correcter"
        self.correction_tokenizer = AutoTokenizer.from_pretrained(name)
        self.correction_model     = AutoModelForSeq2SeqLM.from_pretrained(name).to(device)
        self.correction_model.eval()
        print("[INFO] Grammar Corrector Loaded!")

        # if torch.cuda.device_count() > 1:
        #     print("[INFO] Using %d GPUs..." % torch.cuda.device_count())
        #     self.correction_model = DataParallel(self.correction_model)

    def correct(self, input_sentence, max_candidates=1):
        with torch.no_grad():
            if not self.backup: # Gramformer is available
                correction_prefix = "gec: "
                input_sentence = correction_prefix + input_sentence
                input_ids = self.correction_tokenizer.encode(input_sentence, return_tensors='pt')
                input_ids = input_ids.to(self.device)

                preds = self.correction_model.generate(
                    input_ids,
                    num_beams=3,
                    max_length=128, 
                    early_stopping=True,
                    num_return_sequences=max_candidates)

                corrected = [self.correction_tokenizer.decode(pred, skip_special_tokens=True).strip() for pred in preds]
                return corrected[0]
            else:
                tokenized = self.correction_tokenizer([input_sentence], truncation=True, padding='max_length', 
                            max_length=64, return_tensors='pt').to(self.device)

                preds = self.correction_model.generate(
                    **tokenized,
                    num_beams=5,
                    max_length=128, 
                    early_stopping=True,
                    num_return_sequences=max_candidates)

                corrected = [self.correction_tokenizer.decode(pred, skip_special_tokens=True).strip() for pred in preds]
                return corrected[0]



class Active2Passive(Corruption):
    def __init__(self) -> None:
        super().__init__()
        self.style_change = StyleChange()
        self.grammar_corrector = Gramformer()
        self.backup = self.grammar_corrector.backup

    def __str__(self) -> str:
        return "Active2Passive"

    def post_process(self, text):
        # Split the text into sentences and capitalize the first letter
        # of each sentence.
        sentences = sent_tokenize(text)
        sentences = [sentence.capitalize() for sentence in sentences]
        return " ".join(sentences)

    def __call__(self, text, doc=None, frac=0, **kwargs):
        sentences = sent_tokenize(text)
        transferred = []
        for sentence in sentences:
            transferred.append(self.style_change.generate(sentence))

        if self.backup: # The backup model needds smaller sentences
            corrected = []
            for sentence in transferred:
                corrected.append(self.grammar_corrector.correct(sentence))
            corrected = " ".join(corrected)
        else:
            transferred = " ".join(transferred)
            corrected = self.grammar_corrector.correct(transferred)

        processed = self.post_process(corrected)
        return processed