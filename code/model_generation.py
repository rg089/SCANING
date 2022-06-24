import pandas as pd
import numpy as np
import string
import re
import json
import spacy
import random, math
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
import os
import argparse
import torch


def paraphrase(test_samples, prefix, model, tokenizer, num_return_sequences=1, sample=False):
    with torch.no_grad():
        if type(test_samples) == str:
            test_samples = prefix + test_samples
        else:
            for i in range(len(test_samples)):
                test_samples[i] = prefix[i] + test_samples[i]

        inputs = tokenizer(test_samples, truncation=True, padding="max_length", max_length=max_input_length,
                            return_tensors="pt")
        
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        if sample:
            outputs = model.generate(input_ids, attention_mask=attention_mask, do_sample=True, top_p=0.90, top_k=50,
                            max_length=max_target_length, num_return_sequences=num_return_sequences)
        else:
            outputs = model.generate(input_ids, attention_mask=attention_mask, num_beams=num_beams, max_length=max_target_length,
                                     num_return_sequences=num_return_sequences, diversity_penalty = diversity_penalty,
                                    num_beam_groups = num_beam_groups)

        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_str


def add_paraphrased(data, paraphrased, df, start, end, num_return_sequences):
    """
    add paraphrased outputs to df
    if num_return_sequences > 1, then duplicate the row num_return_sequences times
    """
    pointer = 0 # pointer to the index of the paraphrases
    for i in range(start, end+1):
        for j in range(num_return_sequences):
            values = df.loc[i].values.tolist()
            values.append(paraphrased[pointer])
            data.append(values)
            pointer += 1
    
    return data


def generate_paraphrases(df, model, tokenizer, batch_size=32, num_return_sequences=1, 
                        input_col='corruption', prefix_col='prefix', output_col='bart_v5', 
                        output_path='results.csv', sample=False):

    data = []
    df = df.reset_index(drop=True)
    columns = df.columns.tolist() + [output_col]
    for idx in range(0, len(df), batch_size):
        end = min(idx+batch_size-1, len(df)-1)
        corrupted = df.loc[idx:end, input_col].tolist()
        prefixes = df.loc[idx:end, prefix_col].tolist()
        paraphrased = paraphrase(corrupted, prefixes, model, tokenizer, num_return_sequences=num_return_sequences, 
                        sample=sample)
        # df.loc[idx:end, output_col] = paraphrased
        data = add_paraphrased(data, paraphrased, df, idx, end, num_return_sequences)

        if (idx)%(batch_size*2) == 0:
            print(f"[INFO] {idx+batch_size} SAMPLES DONE!")
            new_df = pd.DataFrame(data, columns=columns)
            new_df.to_csv(output_path, index=False)

    new_df = pd.DataFrame(data, columns=columns)
    new_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_path", "-m", default="bart", type=str, required=True)
    ap.add_argument("--input_path", "-i", default="input.csv", type=str, required=True)
    ap.add_argument("--output_path", "-o", default="output.csv", type=str, required=True)
    ap.add_argument("--batch_size", "-b", default=32, type=int, required=False)
    ap.add_argument("--num_return_sequences", "-n", default=3, type=int, required=False)
    ap.add_argument("--sample", "-s", action="store_true")
    ap.add_argument("--max_input_length", "-mi", default=256, type=int, required=False)
    ap.add_argument("--max_target_length", "-mt", default=256, type=int, required=False)
    ap.add_argument("--input_col", "-ic", default="corruption", type=str, required=False)
    ap.add_argument("--prefix_col", "-pc", default="prefix", type=str, required=False)
    ap.add_argument("--output_col", "-oc", default="bart_v5", type=str, required=False)
    ap.add_argument("--num_beam_groups", "-nbg", default=3, type=int, required=False)

    args = ap.parse_args()

    model_path = args.model_path
    input_path = args.input_path
    output_path = args.output_path
    batch_size = args.batch_size
    num_return_sequences = args.num_return_sequences
    sample = args.sample
    max_input_length = args.max_input_length
    max_target_length = args.max_target_length
    input_col = args.input_col
    prefix_col = args.prefix_col
    output_col = args.output_col
    num_beam_groups = args.num_beam_groups
    num_beams = num_beam_groups * 2
    diversity_penalty = 100.0

    df = pd.read_csv(input_path)

    # load model
    print(f"[INFO] Loading model from {model_path}....")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"[INFO] Model loaded!\n")

    print(f"[INFO] Generating paraphrases with batch_size {batch_size}....")
    generate_paraphrases(df, model, tokenizer, batch_size=batch_size, num_return_sequences=num_return_sequences,
                        input_col=input_col, prefix_col=prefix_col, output_col=output_col, 
                        output_path=output_path, sample=sample)
    print(f"[INFO] Process finished!")

    """
    To run this script, use the following command:
    python model_generation.py -m bart -i input.csv -o output.csv -b 32 -n 1 -ic corruption -pc prefix -oc bart_v5 -nbg 3
    """