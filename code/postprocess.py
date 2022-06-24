import argparse as ap
import pandas as pd
from postprocessing import PostProcessor


def process_col(df, processor, col_name, output_col):
    df[output_col] = df[col_name].apply(lambda x: processor.process(x))
    return df


if __name__ == "__main__":
    parser = ap.ArgumentParser()

    parser.add_argument("--input_file", type=str, default="../../data/bart_v4.4/test.csv")
    parser.add_argument("--output_file", type=str, default="../../data/bart_v4.4/test_selected.csv")
    parser.add_argument("--input_col", type=str, default="bart_complete_v1")
    parser.add_argument("--output_col", type=str, default="bart_complete_v1_processed")

    args = parser.parse_args()

    processor = PostProcessor()

    df = pd.read_csv(args.input_file)
    df = process_col(df, processor, args.input_col, args.output_col)
    df.to_csv(args.output_file, index=False)

    """
    To run this script, use the command:
    python postprocess.py --input_file ../../data/bart_v4.4/test.csv --output_file ../../data/bart_v4.4/test_scores.csv --input_col bart_complete_v1 --output_col bart_complete_v1_processed
    """
