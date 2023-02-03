# SCANING

The source code and data for our paper '*Coherence and Diversity through Noise*: Self-Supervised Paraphrase Generation via Structure-Aware Denoising'.

## Instructions to run the code:

### Setup:
- Install the requirements
```
$ pip install -r requirements.txt
```
- Add the semantic similarity model in the code folder, and download glove.6B.300d.txt into `code/data`.

### NOTE:
For all the commands below, please replace `${X=Z}` with the appropriate arguments. Here, `X` represents the meaning of the argument, and `Z` (wherever specified) represents the default value used in the experiments.

### 1. Generating the noised training set:
```
$ CUDA_VISIBLE_DEVICES=${GPU_ID} python corrupt.py \
    -i ${TRAIN_DATA}\
    -o ${OUTPUT_DATA}\
    -n ${NUM_CORRUPTIONS=2}\
    -p ${PASSIVE_FRAC=0.3}\
    -ic ${INPUT_COL=question}\
    -pe ${PRINT_EVERY=100}\
    -se ${SAVE_EVERY=1000}\
    -pln ${PRESERVE_LAST_N=4}\
    -t\
    -lp ${LOG_PATH}\
    -cf ${CORRUPTION_FILE=corruptions_v10.json}
```

Do this for both the training and validation set.

### 2. Learning $\Delta$:
```
$ CUDA_VISIBLE_DEVICES=${GPU_ID} python train_generation.py\
    --train_path ${TRAIN_NOISE_DATA}\
    --val_path ${VAL_NOISE_DATA}\
    --model_path ${MODEL=facebook/bart-base}\
    --tokenizer_path ${MODEL=facebook/bart-base}\
    --metric_name rouge\
    --save_path ${SAVE_PATH}\
    --save_model_path ${SAVE_MODEL_PATH}\
    --lr ${LR=8e-5}\
    --batch_size ${BATCH_SIZE=32}\
    --epochs ${EPOCHS=15}\
    --max_input_length ${MAX_LENGTH=256}\
    --max_target_length ${MAX_LENGTH=256}\
    --prefix_col ${PREFIX_COL=prefix}\
    --input_col ${INPUT_COL=corruption}\
    --output_col ${OUTPUT_COL=target}
```

### 3. Generating the inference noised set:

```
$ CUDA_VISIBLE_DEVICES=${GPU_ID} python corrupt.py \
    -i ${TRAIN_DATA}\
    -o ${OUTPUT_DATA}\
    -n ${NUM_CORRUPTIONS=4}\
    -p ${PASSIVE_FRAC=0.3}\
    -ic ${INPUT_COL=question}\
    -pe ${PRINT_EVERY=100}\
    -se ${SAVE_EVERY=1000}\
    -pln ${PRESERVE_LAST_N=4}\
    -lp ${LOG_PATH}\
    -cf ${CORRUPTION_FILE=corruptions_v10.json}

$ python add_prompts_train.py \
    --input ${OUTPUT_DATA}\
    --output ${OUTPUT_DATA}\
    --prefix_col "prefix"\
    --log_col "log"\
    --prompt_col "prompt"
```

### 4. Running $\Delta$ on the inference noise:

```
$ CUDA_VISIBLE_DEVICES=${GPU_ID} python model_generation.py \
    -m ${SAVED_DENOISER_PATH} \
    -i ${OUTPUT_DATA}\
    -o ${OUTPUT_DATA_GENERATED}\
    -b ${BATCH_SIZE=32}\
    -n 3\
    -ic ${INPUT=corruption}\
    -pc ${PREFIX=prefix}\
    -oc ${OUTPUT=denoiser_generated}\
    -nbg ${NUM_BEAM_GROUPS=3}
```

### 5. Scoring, Filtering and Selecting:

```
$ CUDA_VISIBLE_DEVICES=${GPU_ID} python score.py \
    --sim_model_path ${SIM_MODEL=ParaQD_v3.1} \
    --input_file ${OUTPUT_DATA_GENERATED}\
    --output_file ${OUTPUT_DATA_SCORED}\
    --original_col ${ORIGINAL_COL=original}\
    --candidate_col ${CANDIDATE_COL=denoiser_generated}

$ python selecting.py \
    --input_file ${OUTPUT_DATA_SCORED}\
    --output_file ${PHASE2_TRAIN}\
    --original_col ${ORIGINAL_COL=original}\
    --candidate_col ${CANDIDATE_COL=denoiser_generated}\
    --top_n ${TOP_N=2}\
    --num_corruptions ${NUM_CORR=12}\
    --sim_thresh ${SIM_THRESH=0.9}\
    --bleu_thresh ${BLEU_THRESH=0.2}\
    --wpd_thresh ${WPD_THRESH=0.15}\
    --diversity_thresh ${DIVERSITY_THRESH=0.2}\
    --lam ${LAMBDA=0.65}\
    --prompt_consistency_filtering
```
### 6. Generating validation set for $\Psi$

- Running 3,4,5 on the training file generates the training set for $\Psi$
- Run 3,4,5 again on the original validation file to generate the validation set for $\Psi$

### 7. Training $\Psi$:

```
$ CUDA_VISIBLE_DEVICES=${GPU_ID} python train_generation.py \
    --train_path ${PHASE2_TRAIN}\
    --val_path ${PHASE2_VAL}\
    --model_path {MODEL=facebook/bart-base} \
    --tokenizer_path {MODEL=facebook/bart-base} \
    --metric_name rouge \
    --save_path ${SAVE_PATH}\
    --save_model_path ${SAVE_MODEL_PATH}\
    --lr ${LR=8e-5}\
    --batch_size ${TRAIN_BATCH_SIZE=32}\
    --epochs ${EPOCHS=15}\
    --max_input_length ${MAX_LENGTH=256}\
    --max_target_length ${MAX_LENGTH=256}\
    --prefix_col ${PREFIX=prompt}\
    --input_col ${ORIGINAL_COL=original}\
    --output_col ${OUTPUT_COL=paraphrase}
```

### 8. Evaluating $\Delta$:

Repeat 3 and 4 on the test set. Then, run:

```
$ CUDA_VISIBLE_DEVICES=${GPU_ID} python score.py \
    --sim_model_path ${SIM_MODEL=ParaQD_v3.1} \
    --input_file ${CORRUPTED_GENERATED}\
    --output_file ${CORRUPTED_SCORES}\
    --original_col ${ORIGINAL_COL=original}\
    --candidate_col ${CANDIDATE_COL=denoiser_generated}\
    --final

$ python get_results.py\
     --input_file ${CORRUPTED_SCORES} \
     --output_file ${RESULTS} \
     --method_name "denoiser" \
     --add
```

### Evaluating $\Psi$:

On the test file, run the following commands:
```
$ python add_prompts_test_selected.py \
    --input_file ${TEST_FILE}\
     --prompt_file ${PROMPT_FILE=corruption/helper/test_prompts.json}\
     --output_file ${PHASE2_PROMPTS}\
     --use_prompts

$ CUDA_VISIBLE_DEVICES=${GPU_ID} python model_generation.py \
    -m  models/reconstruction_bart_v${VERSION}_phase2_model \
    -i ${PHASE2_PROMPTS}\
    -o ${PHASE2_GENERATED}\
    -b ${BATCH_SIZE}\
    -n ${NUM=3}\
    -ic ${INPUT=question}\
    -pc ${PREFIX=prompt}\
    -oc ${OUTPUT=paraphraser_output}\
    -nbg 3

$ CUDA_VISIBLE_DEVICES=${GPU_ID} python score.py \
    --sim_model_path ${SIM_MODEL=ParaQD_v3.1} \
    --input_file ${PHASE2_GENERATED}\
    --output_file ${PHASE2_SCORES}\
    --original_col ${ORIGINAL=question}\
    --candidate_col ${CANDIDATE=paraphraser_output}\
    --final

$ python get_results.py\
     --input_file ${PHASE2_SCORES} \
     --output_file ${RESULTS} \
     --method_name "paraphraser" \
     --add
```

Thanks!
