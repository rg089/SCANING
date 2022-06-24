from soupsieve import select
from selection import Scorer
import pandas as pd
import argparse as ap


def add_scores(df, scorer, original_col="target", candidate_col="bart_v4.4"):
    assert original_col in df.columns and candidate_col in df.columns\

    output_cols = ["similarity", "bleu", "wpd", "diversity", "numeracy", "overall"]
    for idx, row in df.iterrows():
        original = row[original_col]
        candidate = row[candidate_col]
        similarity_score, bleu, wpd, diversity_score, numeracy_score, overall_score = scorer.score(original, candidate)
        try:
            df.loc[idx, output_cols] = [similarity_score, bleu, wpd, diversity_score, numeracy_score, overall_score]
        except:
            print(idx, original, candidate, similarity_score, diversity_score, overall_score)
            raise Exception

        if (idx+1) % 200 == 0:
            print(f"[INFO] {idx+1} SAMPLES SCORED!")
            
    return df

if __name__ == "__main__":
    parser = ap.ArgumentParser()

    parser.add_argument("--sim_model_path", type=str, default="ParaQDv3.1")
    parser.add_argument("--input_file", type=str, default="../../data/bart_v4.4/test.csv")
    parser.add_argument("--output_file", type=str, default="../../data/bart_v4.4/test_scores.csv")
    parser.add_argument("--original_col", type=str, default="target")
    parser.add_argument("--candidate_col", type=str, default="bart_v4.4")
    parser.add_argument("--select", action="store_true", help="Scoring for selection")
    parser.add_argument("--final", action="store_true", help="Scoring for final")

    args = parser.parse_args()

    assert args.select or args.final, "Please specify whether scoring for selection or final"
    assert not (args.select and args.final), "Please specify only one of --select or --final"

    print("[INFO] Start Scoring...")

    scorer = Scorer(args.sim_model_path, select=args.select)

    df = pd.read_csv(args.input_file)
    df = add_scores(df, scorer, original_col=args.original_col, candidate_col=args.candidate_col)
    df.to_csv(args.output_file, index=False)

    print("[INFO] Scoring Complete!")

    """
    To run this file, use the command:
    python score.py --sim_model_path ParaQDv3.1 --input_file ../../data/bart_v4.4/test.csv --output_file ../../data/bart_v4.4/test_scores.csv --original_col target --candidate_col bart_v4.4 --select
    """