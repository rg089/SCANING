import pandas as pd
import argparse


def combine_files(output_path, num_files=5):
    """
    Combine multiple files into a single file.
    :param input_file: The input file.
    :param num_files: The number of files.
    :return:
    """
    df = pd.DataFrame()

    for i in range(num_files):
        path = output_path.replace('.csv', f'_{i+1}.csv')
        current = pd.read_csv(path)
        df = pd.concat([df, current], axis=0)

    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_path', type=str, help='The path to the output file.')
    parser.add_argument('--num_files', type=int, default=5, help='The number of files.')

    args = parser.parse_args()

    combine_files(args.output_path, args.num_files)

    """
    To run this script, use the following command:
    python combine.py --output_path output_file.csv --num_files 5
    """