import json
import argparse
import re
import pandas as pd
import random
# from corruption.preprocessing import preprocess


def convert_log_to_prompt(log, prefix, converter, shuffle=False):
    """
    Convert log to prompt.
    """
    prompts = set([])
    corruptions = log.split("+")

    for corruption in corruptions:
        corruption = corruption.strip().lower()

        for corruption_1, prompt in converter.items():
            if corruption_1 in corruption:
                prompts.add(prompt)

    prompts = sorted(list(prompts))
    if shuffle: random.shuffle(list(prompts))

    prompt = " ".join(prompts)

    prefix = prefix.strip(" :")
    prompt = f"{prefix} {prompt} : "
    prompt = re.sub(r"\s+", " ", prompt)

    return prompt


def process_prompt(df, converter, original_col="original", prefix_col="prefix", log_col="log", prompt_col="prompt"):
    """
    Process prompt.
    """
    # df[original_col] = df[original_col].apply(lambda x: preprocess(x))
    df[prompt_col] = df.apply(
        lambda row: convert_log_to_prompt(row[log_col], row[prefix_col], converter), axis=1
    )

    return df


if __name__ == "__main__":
    converter = json.load(open("corruption/helper/corruption_prompt.json"))

    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--input", required=True, help="path to input file", default="data/train.csv")
    ap.add_argument("-o", "--output", required=True, help="path to output file", default="data/train_prompt.csv")
    ap.add_argument("-pref", "--prefix_col", required=True, help="prefix column", default="prefix")
    ap.add_argument("-log", "--log_col", required=True, help="log column", default="log")
    ap.add_argument("-prompt", "--prompt_col", required=True, help="prompt column", default="prompt")

    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = process_prompt(df, converter, prefix_col=args.prefix_col, log_col=args.log_col, prompt_col=args.prompt_col)
    df.to_csv(args.output, index=False)

    """
    To run this script, use the following command:
    python add_prompts_train.py --input data/aqua_train_v5_phase2_corrupted.csv --output data/aqua_train_v5_phase2_corrupted.csv --prefix_col prefix --log_col log --prompt_col prompt
    """