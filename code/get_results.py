import pandas as pd
import argparse as ap
import os

columns = ["similarity", "bleu", "wpd", "diversity", "numeracy", "overall"]
weights = {"similarity": 0.35, "diversity": 0.35, "numeracy": 0.3}

eps = 1e-9

def weighted_hm(sim, div, num):
    """
    calculates the weighted harmonic mean
    """
    numer = sum(list(weights.values()))
    denom = (weights['similarity']/(sim+eps)) + (weights['diversity']/(div+eps)) + (weights['numeracy']/(num+eps))
    return numer/denom


def process_df(df):
    """
    processes the dataframe
    """
    main_col = "original" if "original" in df.columns else "question"
    df = df.loc[df.groupby(main_col)["overall"].idxmax()]
    return df


def get_results_df(df, method_name='reconstructor v10', out_path='data/aqua_results.csv', add=True):
    """
    returns the results dataframe
    """
    global columns
    df = process_df(df)

    columns_updated = []
    for col in columns: 
        columns_updated.append(col + '_mean')
        columns_updated.append(col + '_std')

    columns = ["name"] + columns
    columns_updated = ["name"] + columns_updated

    if os.path.exists(out_path) and add:
        df_results = pd.read_csv(out_path)
    else:
        df_results = pd.DataFrame(columns=columns_updated)

    row = [method_name]
    for col in columns[1:]:
        row.append(round(df[col].mean(), 2))
        row.append(round(df[col].std(), 2))

    print(row)
    new_df = pd.DataFrame([row], columns=columns_updated)
    df_results = pd.concat([df_results, new_df], axis=0)

    df_results.to_csv(out_path, index=False)
    return df_results


if __name__ == "__main__":
    ap = ap.ArgumentParser()

    ap.add_argument("--input_file", type=str, default="../../data/aqua_test.csv", help="input file")
    ap.add_argument("--output_file", type=str, default="../../data/aqua_results.csv", help="output file")
    ap.add_argument("--method_name", type=str, default="reconstructor v10", help="method name")
    ap.add_argument("--add", action="store_true", help="add to existing results")

    args = ap.parse_args()

    df = pd.read_csv(args.input_file)
    df_results = get_results_df(df, method_name=args.method_name, out_path=args.output_file, add=args.add)

    print("[INFO] Done!")


    """
    To run this script, use the following command:
    python get_results.py --input_file data/aqua_test.csv --output_file data/aqua_results.csv --method_name "reconstructor v10" --add
    """


