import pandas as pd
import argparse as ap
from selection.metrics import diversity as diversity_metric
from joblib import Parallel, delayed


substitution_prompts = ["template", "substitute", "replace"]
shuffling_prompts = ["shuffle", "shuffling", "permute"]
other_prompts = ["fix", "simplify"]


def filter_possible(samples, sim_thresh=0.9, bleu_thresh=0.2, wpd_thresh=0.15, diversity_thresh=0.2, numeracy_thresh=0.6):
    """
    given a list of tuples (candidates, bleu, wpd, diversity, similarity, numeracy, overall, logs, prefixes, prompts),
    filters the list according to the given thresholds and returns a list of tuples
    """
    filtered_samples = []

    for sample in samples:
        candidate, bleu, wpd, diversity, similarity, numeracy, overall, log, prefix, prompt = sample
        if similarity >= sim_thresh and numeracy >= numeracy_thresh and (bleu >= bleu_thresh or wpd >= wpd_thresh or diversity >= diversity_thresh): # First criteria satisfied
            
            if prompt_consistency_filtering:
                has_substitution = any([substitution_prompt in prompt for substitution_prompt in substitution_prompts])
                if has_substitution and bleu < max(0, bleu_thresh-0.05): # Relaxing the threshold a little
                    continue

                has_shuffling = any([shuffling_prompt in prompt for shuffling_prompt in shuffling_prompts])
                if has_shuffling and wpd < max(0, wpd_thresh-0.1):
                    continue

                has_other = any([other_prompt in prompt for other_prompt in other_prompts])
                if has_other and diversity < max(0, diversity_thresh-0.1):
                    continue

            filtered_samples.append(sample) # 2nd criteria satisfied

    return filtered_samples


def adapted_mmr(samples, top_n=2, lam=0.6):
    """
    given a list of tuples (candidates, bleu, wpd, diversity, similarity, numeracy, overall, logs, prefixes, prompts),
    returns a selected list of tuples (candidates, log, prefix, prompt)  
    """
    if len(samples) <= top_n:
        final = []
        for idx in range(len(samples)):
            candidate, _, _, _, _, _, _, log, prefix, prompt = samples[idx]
            final.append((candidate, log, prefix, prompt))

        return final

    selected_idxs = []
    candidate_idxs = [i for i in range(len(samples))]

    for i in range(top_n):
        scores = []

        for candidate_idx in candidate_idxs:
            candidate, _, _, _, _, _, overall, _, _, _ = samples[candidate_idx]
            
            inter_sample_div = []
            for selected_idx in selected_idxs:
                selected_candidate, _, _, _, _, _, _, _, _, _ = samples[selected_idx]
                inter_sample_div.append(diversity_metric(candidate, [selected_candidate])[0][-1]) # Using the overall diversity


            min_inter_sample_div = min(inter_sample_div) if len(inter_sample_div) > 0 else 0
            # print(inter_sample_div, min_inter_sample_div, overall, lam)
            score = lam * overall + (1-lam) * min_inter_sample_div

            scores.append((candidate_idx, score))

        selected_idx = max(scores, key=lambda x: x[1])[0]
        selected_idxs.append(selected_idx)
        candidate_idxs.remove(selected_idx)

    final = []
    for selected_idx in selected_idxs:
        candidate, _, _, _, _, _, _, log, prefix, prompt = samples[selected_idx]
        final.append((candidate, log, prefix, prompt))

    return final


def select_n_parallel(df, i, num_corruptions, original_col, candidate_col, overall_col, log_col, prefix_col, 
                prompt_col, sim_col, bleu_col, wpd_col, diversity_col, numeracy_col, sim_thresh=0.9, bleu_thresh=0.2, 
                wpd_thresh=0.15, diversity_thresh=0.2, top_n=2, lam=0.6):
    if (i+1) % 200 == 0:
        print(f"[INFO] {i+1} SAMPLES COMPLETED!")

    data = df.iloc[i*num_corruptions:(i+1)*num_corruptions, :]
    assert data[original_col].nunique() in [1,2]

    original = data[original_col].values[0]
    candidates = data.loc[:, candidate_col].values.tolist()
    bleu = data.loc[:, bleu_col].values.tolist()
    wpd = data.loc[:, wpd_col].values.tolist()
    diversity = data.loc[:, diversity_col].values.tolist()
    numeracy = data.loc[:, numeracy_col].values.tolist()
    similarity = data.loc[:, sim_col].values.tolist()
    overall = data.loc[:, overall_col].values.tolist()
    logs = data.loc[:, log_col].values.tolist()
    prefixes = data.loc[:, prefix_col].values.tolist()
    prompts = data.loc[:, prompt_col].values.tolist()

    scored_candidates = list(zip(candidates, bleu, wpd, diversity, similarity, numeracy, overall, logs, prefixes, prompts))
    scored_candidates = filter_possible(scored_candidates, sim_thresh=sim_thresh, bleu_thresh=bleu_thresh, wpd_thresh=wpd_thresh, 
                                        diversity_thresh=diversity_thresh)
    scored_candidates = adapted_mmr(scored_candidates, top_n=top_n, lam=lam)
    return [(original, candidate, log, prefix, prompt) for (candidate, log, prefix, prompt) in scored_candidates]


def select_parallel(df, top_n=2, original_col="original", candidate_col="bart_v5.2", overall_col="overall", log_col="log", 
            prefix_col="prefix", prompt_col="prompt", num_corruptions=12, sim_col='similarity', bleu_col='bleu', wpd_col='wpd', 
            diversity_col='diversity', numeracy_col='numeracy', sim_thresh=0.9, bleu_thresh=0.2, wpd_thresh=0.15, 
            diversity_thresh=0.2, lam=0.6):

    output_cols = ["original", "log", "prefix", "prompt", "paraphrase"]
    data = []

    num_original = len(df) // num_corruptions

    scored_candidates = Parallel(n_jobs=1)(delayed(select_n_parallel)(df, i, num_corruptions, original_col, candidate_col, 
                            overall_col, log_col, prefix_col, prompt_col, sim_col, bleu_col, wpd_col, diversity_col, numeracy_col, 
                            sim_thresh=sim_thresh, bleu_thresh=bleu_thresh, wpd_thresh=wpd_thresh, diversity_thresh=diversity_thresh, 
                            top_n=top_n, lam=lam) for i in range(num_original))

    for sample in scored_candidates:
        for i, (original, candidate, log, prefix, prompt) in enumerate(sample):
            data.append([original, log, prefix, prompt, candidate])

    df_selected = pd.DataFrame(data, columns=output_cols)
    return df_selected


if __name__ == "__main__":
    parser = ap.ArgumentParser()

    parser.add_argument("--sim_model_path", type=str, default="ParaQDv3.1")
    parser.add_argument("--input_file", type=str, default="../../data/bart_v4.4/test.csv")
    parser.add_argument("--output_file", type=str, default="../../data/bart_v4.4/test_selected.csv")
    parser.add_argument("--original_col", type=str, default="original")
    parser.add_argument("--candidate_col", type=str, default="bart_v4.4")
    parser.add_argument("--overall_col", type=str, default="overall")
    parser.add_argument("--log_col", type=str, default="log")
    parser.add_argument("--prefix_col", type=str, default="prefix")
    parser.add_argument("--prompt_col", type=str, default="prompt")
    parser.add_argument("--top_n", type=int, default=2)
    parser.add_argument("--num_corruptions", type=int, default=12)
    parser.add_argument("--sim_thresh", type=float, default=0.9)
    parser.add_argument("--bleu_thresh", type=float, default=0.2)
    parser.add_argument("--wpd_thresh", type=float, default=0.15)
    parser.add_argument("--diversity_thresh", type=float, default=0.2)
    parser.add_argument("--numeracy_thresh", type=float, default=0.6)
    parser.add_argument("--lam", type=float, default=0.75)
    parser.add_argument("--sim_col", type=str, default="similarity")
    parser.add_argument("--bleu_col", type=str, default="bleu")
    parser.add_argument("--wpd_col", type=str, default="wpd")
    parser.add_argument("--diversity_col", type=str, default="diversity")
    parser.add_argument("--numeracy_col", type=str, default="numeracy")
    parser.add_argument("--prompt_consistency_filtering", "-pcf", action="store_true")


    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    prompt_consistency_filtering = args.prompt_consistency_filtering
    if prompt_consistency_filtering: print("[INFO] Using Prompt Consistency Filtering")

    print("[INFO] Start selecting...")
    df = select_parallel(df, top_n=args.top_n, original_col=args.original_col, candidate_col=args.candidate_col, overall_col=args.overall_col, 
                        log_col=args.log_col, prefix_col=args.prefix_col, prompt_col=args.prompt_col, num_corruptions=args.num_corruptions,
                        sim_col=args.sim_col, bleu_col=args.bleu_col, wpd_col=args.wpd_col, diversity_col=args.diversity_col, numeracy_col=args.numeracy_col,
                        sim_thresh=args.sim_thresh, bleu_thresh=args.bleu_thresh, wpd_thresh=args.wpd_thresh, diversity_thresh=args.diversity_thresh, lam=args.lam)

    df.to_csv(args.output_file, index=False)
    print("[INFO] Finish selecting.")

    """
    To run this file, use the command:
    python selecting.py --input_file data/aqua_train_v7_phase2_corrupted_scores.csv --output_file data/aqua_train_v7_phase2_train_new2.csv --original_col original --candidate_col bart_v7 --top_n 2 --num_corruptions 12 --sim_thresh 0.9 \
    --bleu_thresh 0.2 --wpd_thresh 0.15 --diversity_thresh 0.2 --numeracy_thresh 0.6 --lam 0.7 --prompt_consistency_filtering
    """