import pandas as pd
import argparse
import json, re


def process_prompt(prompt):
    prompt = prompt.strip().strip(":")
    prompt = "paraphrase " + prompt + " : "
    prompt = re.sub(r'\s+', ' ', prompt)
    return prompt


def add_prompt(df, prompt_list=[], prompt_col='prompt'):
    assert prompt_col not in df.columns

    data = []
    columns = df.columns.tolist() + [prompt_col]

    for idx, row in df.iterrows():
        for prompt in prompt_list:
            new_row = row.values.tolist() + [prompt]
            data.append(new_row)

    df_new = pd.DataFrame(data, columns=columns)
    return df_new
            

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_file", type=str, default="../../data/bart_v4.4/test.csv", help="input file")
    ap.add_argument("--prompt_file", type=str, default="../../data/bart_v4.4/prompt.json", help="prompt file")
    ap.add_argument("--output_file", type=str, default="../../data/bart_v4.4/test_selected.csv", help="output file")
    ap.add_argument("--use_prompts", action="store_true", help="use prompts")

    args = ap.parse_args()

    df = pd.read_csv(args.input_file)
    
    if args.use_prompts:
        with open(args.prompt_file, 'r') as f:
            prompt_list = json.load(f)
        prompt_list = [process_prompt(prompt) for prompt in prompt_list]
    else:
        prompt_list = ["", "passive"]
        prompt_list = [process_prompt(prompt) for prompt in prompt_list]

    df_new = add_prompt(df, prompt_list)
    df_new.to_csv(args.output_file, index=False)

    print("[INFO] Done!")

    """
    To use this file, use the command:
    python add_prompts_test_selected.py --input_file data/aqua_test.csv --prompt_file data/prompt.json --output_file data/aqua_test_selected.csv --use_prompts
    """