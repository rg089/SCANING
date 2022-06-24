import pandas as pd
import numpy as np

from corruption import Corrupter, CorruptionLogger
from utils import find_ops
import argparse as ap
import timeit
from datetime import datetime
import spacy
spacy.prefer_gpu()


def corrupt_single(corrupter, question, num_corruptions=2, convert_to_passive=[False, True, False], verbose=False, 
                logger=None, train=True, support_set=False, **kwargs):
    """
    :param question: the input question
    :param num_corruptions: number of corruptions to apply
    :param convert_to_passive: list of booleans, whether to convert to passive or not
    :param verbose: whether to print the question after each corruption
    :return: a list of the corruptions and reconstructions
    """
    assert len(convert_to_passive) == num_corruptions
    passive_qn = ""
    if sum(convert_to_passive) > 0: passive_qn = corrupter.active2passive(question)
    
    qns = [question for i in range(num_corruptions)]
    corrupted_qns, logs_single = corrupter.corrupt(question, num_augs=num_corruptions, verbose=verbose, logger=logger)

    if not train and not support_set: # If inference, then try to passivize temp+/subst if possible
        if sum(convert_to_passive) > 0: # Only do this if there is one passive corruption
            found_idx = find_ops(logs_single, ops=["templatization", "substitution"])
            if found_idx != -1:
                convert_to_passive = [False for _ in range(num_corruptions)]
                convert_to_passive[found_idx] = True

    if not train and support_set: # If support set, changing one corruption to the original question as the operation is passivization only
        if sum(convert_to_passive) > 0:
            idx = [i for i, passive in enumerate(convert_to_passive) if passive == True][0]
            corrupted_qns[idx] = question
            logs_single[idx] = "Passivization"

    reconstructed_qns = [passive_qn if is_passive else question for is_passive in convert_to_passive]
    prefixes = ["paraphrase: " if not is_passive else "paraphrase passive: " for is_passive in convert_to_passive]
    logger.combine_with_current([f"Prefix: {prefix}" for prefix in prefixes])

    return list(zip(qns, corrupted_qns, logs_single, prefixes, reconstructed_qns))
    

def corrupt(corrupter, input_path, out_path=None, input_column="question", output_columns=["original", "corruption", "log", "prefix", "target"], num_corruptions=2, prob_passive=0.2,
             verbose=False, print_every=200, save_every=200, train=True, support_set=False, log_path="corruption_logs.json", **kwargs):
    """
    given a dataframe, creates a new dataframe with 'output_columns' and saves it in 'out_path'          
    """
    assert len(output_columns) in [3, 4, 5], "output_columns must be original, corruption, log (optional), target and prefix (optional)"
    assert not train or not support_set, "support_set is only supported for inference"

    df = pd.read_csv(input_path)
    num_questions = len(df[input_column])

    if num_corruptions == -1:
        num_corruptions = 10 # CHANGE THIS IF ADDING CORRUPTIONS

    num_output_questions = num_questions * num_corruptions
    convert_to_passive_complete = np.random.choice([True, False], size=num_output_questions, p=[prob_passive, 1-prob_passive]) # useful for training corruptions only
    num_passive = round(num_questions*prob_passive)
    convert_to_passive_idxs = np.random.choice(range(num_questions), size=num_passive, replace=False)
    logger = CorruptionLogger()
    logger.add_output_file(out_path)

    data = []
    questions = df[input_column].values.tolist()
    start = timeit.default_timer()

    for i, question in enumerate(questions):
        if train:
            convert_to_passive = convert_to_passive_complete[i*num_corruptions:(i+1)*num_corruptions]
        else:
            convert_to_passive = [False for _ in range(num_corruptions)]
            if i in convert_to_passive_idxs:
                passive_idx = np.random.choice(range(num_corruptions))
                convert_to_passive[passive_idx] = True # During inference, set at max one corruption to be passive

        info_single = corrupt_single(corrupter, question, num_corruptions=num_corruptions, 
                convert_to_passive=convert_to_passive, verbose=verbose, logger=logger, train=train, support_set=support_set, **kwargs)

        data.extend(info_single)
        logger.combine_with_current([f"Passive Target: {convert_to_passive_i}" for convert_to_passive_i in convert_to_passive])
        logger.update_log()

        if (i+1) % print_every == 0:
            stop = timeit.default_timer()
            execution_time = int(stop - start) # Logging the time taken
            minutes = execution_time // 60
            seconds = execution_time % 60
            print(f"\n[INFO] {i+1} Samples Completed!")
            print(f"[INFO] Execution Time: {minutes}m {seconds}s")
            start = timeit.default_timer()

        if (i+1) % save_every == 0:
            df_out = pd.DataFrame(data, columns=output_columns)
            if out_path is not None: df_out.to_csv(out_path, index=False)
            logger.save_logs(log_path)

    df_out = pd.DataFrame(data, columns=output_columns)
    if out_path is not None: df_out.to_csv(out_path, index=False)
    print("\n[INFO] Process Complete!\n")

    logger.save_logs(log_path)
    return df_out


if __name__ == "__main__":
    parser = ap.ArgumentParser(description="corrupts a dataset")
    parser.add_argument("--input_path", "-i", type=str, help="path to the input dataset")
    parser.add_argument("--output_path", "-o", type=str, help="path to the output dataset")
    parser.add_argument("--num_corruptions", "-n", type=int, default=3, help="number of corruptions to apply per sample")
    parser.add_argument("--prob_passive", "-p", type=float, default=0.2, help="probability of converting to passive")
    parser.add_argument("--input_column", "-ic", type=str, default="question", help="column name of the input column")
    parser.add_argument("--print_every", "-pe", type=int, default=200, help="print every X samples")
    parser.add_argument("--save_every", "-se", type=int, default=200, help="save every X samples")
    parser.add_argument("--preserve_last_n", "-pln", type=int, default=4, help="preserve the last X tokens")
    parser.add_argument("--train", "-t", action="store_true", help="whether to use the corruptions for training.")
    parser.add_argument("--verbose", "-v", action="store_true", help="whether to print the question after each corruption")
    parser.add_argument("--log_path", "-lp", type=str, default="logs/corruption_logs.json", help="path to save the json corruption log")
    parser.add_argument("--corruption_file", "-cf", type=str, default="corruptions.json", help="path to the json file containing the corruptions")
    parser.add_argument("--support-set", "-ss", action="store_true", help="whether to use the support set")


    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    num_corruptions = args.num_corruptions
    prob_passive = args.prob_passive
    input_column = args.input_column
    print_every = args.print_every
    save_every = args.save_every
    # Additional keyword arguments for specific corruptions
    preserve_last_n = args.preserve_last_n
    train = args.train
    support_set = args.support_set
    verbose = args.verbose
    log_path = args.log_path
    # Add current date to log_path
    log_path = log_path.split(".")[0] + f"_{datetime.now().strftime('%d-%m-%y_%H:%M:%S')}.json"
    
    corrupter = Corrupter(fname=args.corruption_file, train=train)

    corrupt(corrupter, input_path=input_path, out_path=output_path, num_corruptions=num_corruptions, 
            prob_passive=prob_passive, input_column=input_column, print_every=print_every, 
            save_every=save_every, preserve_last_n=preserve_last_n, train=train, support_set=support_set, verbose=verbose, log_path=log_path)

    """
    To run, use the following command:
    python corrupt.py -i data/corruption_dummy.csv -o data/corruption_dummy_output.csv -n 2 -p 0.2 -ic question -pe 200 -se 200 -pln 4 -t
    -lp logs/corruption_logs.json

    python corrupt.py -i data/corruption_dummy.csv -o data/corruption_dummy_output.csv -n 8 -p 1 -ic question -pe 200 -se 200 -pln 4 -lp logs/corruption_logs.json -cf corruptions_v7.json
    python corrupt.py -i data/corruption_dummy.csv -o data/corruption_dummy_output.csv --support-set -n 6 -p 1 -ic question -pe 200 -se 200 -pln 4 -lp logs/corruption_logs_support_trial.json -cf corruptions_phase2_support.json

    python corrupt.py -i data/aqua_train_v4_phase1.csv -o data/aqua_train_v4_phase1_corrupted.csv -n 3 -p 0.25 -ic question -pe 400 -se 1000 -pln 5 -t -lp logs/corruption_logs_v4_phase1.json -cf corruptions.json
    """