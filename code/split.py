import pandas as pd
import argparse


def split_files(input_path, split_sizes=[], num_splits=-1):
    """
    Split a file into multiple files.
    :param input_file: The file to split.
    :param output_file: The output file.
    :param split_sizes: The sizes of the splits.
    :param num_splits: The number of splits.
    :return:
    """
    assert len(split_sizes) == 0 and num_splits > 0 or len(split_sizes) > 0

    df = pd.read_csv(input_path)
    n = len(df)

    if sum(split_sizes) != n:
        split_sizes = [n // num_splits] * num_splits
        if n % num_splits > 0:
            split_sizes[-1] += n % num_splits

    assert sum(split_sizes) == n

    for i, size in enumerate(split_sizes):
        start = sum(split_sizes[:i])
        end = start + size

        current = df.iloc[start:end]
        path = input_path.replace('.csv', f'_{i+1}.csv')

        current.to_csv(path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, help='The path to the input file.')
    parser.add_argument('--split_sizes', nargs='*', default=[100, 100], help='The sizes of the splits.')
    parser.add_argument('--num_splits', type=int, default=-1, help='The number of splits.')

    args = parser.parse_args()

    split_files(args.input_path, args.split_sizes, args.num_splits)

    """
    To run this script, use the following command:
    python split.py --input_path input_file.csv --split_sizes 100 100 --num_splits 2
    """