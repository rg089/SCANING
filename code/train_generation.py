import transformers
import numpy as np

import torch
import datetime
import os

import nltk 
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import wandb
import argparse
wandb.login()


CONFIG = {}


def load_tools(model_path, **kwargs):
    """
    load the model and tokenizer from the path
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading model: {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    if "special_tokens" in kwargs:
        tokenizer.add_tokens(kwargs["special_tokens"], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        print(f"[INFO] Added special tokens: {kwargs['special_tokens']}")

    print(f"[INFO] Model loaded.")

    CONFIG["tokenizer"] = tokenizer
    CONFIG["model"] = model
    return model, tokenizer
    

def preprocess_function(examples):
    tokenizer = CONFIG["tokenizer"]
    prefix_col = CONFIG["prefix_col"]; input_col = CONFIG["input_col"]; output_col = CONFIG["output_col"]
    inputs = [prefix + doc for prefix, doc in zip(examples[prefix_col], examples[input_col])]
    model_inputs = tokenizer(inputs, max_length=CONFIG["max_input_length"], truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples[output_col], max_length=CONFIG["max_target_length"], truncation=True)

    labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def get_dataset(train_path, val_path, metric_name, **kwargs):
    """
    load the dataset from the path, tokenize it and return the dataset
    """
    data_files = {"train": train_path, "validation": val_path}
    raw_datasets = load_dataset("csv", data_files=data_files)
    metric = load_metric(metric_name)
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, 
                        remove_columns=raw_datasets["train"].column_names)

    CONFIG["metric"] = metric
    CONFIG["tokenized_datasets"] = tokenized_datasets
    return tokenized_datasets, metric


def compute_metrics(eval_pred):
    tokenizer = CONFIG["tokenizer"]
    metric = CONFIG["metric"]

    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


def train_model(model, tokenizer, tokenized_datasets):
    save_path = CONFIG["save_path"]
    lr = CONFIG["lr"]
    batch_size = CONFIG["batch_size"]
    epochs = CONFIG["epochs"]
    log_path = CONFIG["log_path"]


    args = Seq2SeqTrainingArguments(
                    output_dir=save_path,
                    learning_rate=lr,
                    do_train = True,
                    do_eval = True,
                    evaluation_strategy="steps",
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    weight_decay=0.01,
                    save_total_limit=3,
                    load_best_model_at_end=True,
                    num_train_epochs=epochs,
                    predict_with_generate=True,
                    # fp16=True,
                    logging_dir=log_path,
                    logging_steps=500,
                    save_steps=1000,
                    metric_for_best_model = 'rougeLsum',
                    greater_is_better = True,
                    report_to = "wandb",
                )

    # Reporting to wandb
    wandb_run = wandb.init(
    project="reconstruction_v3",
    config={ "per_device_train_batch_size": batch_size, "learning_rate": lr})

    now = datetime.datetime.now()
    current_time = now.strftime("%d.%b.%Y-%-I:%M:%S%p")

    run_name = save_path.rstrip("/").split("/")[-1] + "-" + current_time 
    wandb_run.name = run_name
    print(f"[INFO] W&B run name: {wandb_run.name}")

    # Defining the data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, 
                                       model=model,
                                       label_pad_token_id = -100, # if padding max_length is set
                                       pad_to_multiple_of= 8,
                                       )
    
    # Defining the trainer
    trainer = Seq2SeqTrainer(model=model,
                            args=args,
                            data_collator=data_collator,
                            train_dataset=tokenized_datasets["train"],
                            eval_dataset=tokenized_datasets["validation"],
                            tokenizer=tokenizer,
                            compute_metrics=compute_metrics,)

    
    # Initial Evaluation
    trainer.evaluate()

    print(f"[INFO] Training started....")
    # Training
    if not CONFIG["resume_training"]:
        trainer.train()
    else:
        trainer.train(CONFIG["model_path"])

    # Final Evaluation
    trainer.evaluate()

    # Saving the model
    save_model_path = CONFIG["save_model_path"]
    trainer.save_model(save_model_path)
    tokenizer.save_pretrained(save_model_path)
    print("[INFO] Process Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train.csv", help="Path to the training data")
    parser.add_argument("--val_path", type=str, default="data/validation.csv", help="Path to the validation data")
    parser.add_argument("--model_path", type=str, default="facebook/bart-large", help="Path to the model")
    parser.add_argument("--tokenizer_path", type=str, default="facebook/bart-large", help="Path to the tokenizer")
    parser.add_argument("--metric_name", type=str, default="rouge", help="Metric to use for evaluation")
    parser.add_argument("--save_path", type=str, default="data/saved_models/", help="Path to save the model")
    parser.add_argument("--save_model_path", type=str, default="data/saved_models/model.bin", help="Path to save the model")
    parser.add_argument("--log_path", type=str, default="data/logs/", help="Path to save the logs")
    parser.add_argument("--lr", type=float, default=7e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--max_input_length", type=int, default=512, help="Maximum input length")
    parser.add_argument("--max_target_length", type=int, default=512, help="Maximum target length")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens")
    parser.add_argument("--prefix_col", type=str, default="prefix", help="Prefix column for the input")
    parser.add_argument("--extra_tokens", nargs="*", default=['INTJ', 'X', 'VERB', 'NUM', 'SCONJ', 'NOUN', 'AUX', 'PUNCT', 'PROPN', 'SYM', 'SPACE', 'CCONJ', 'ADP', 'PRON', 'ADJ', 'DET', 'PART', 'ADV'], help="Extra tokens to add to the input")
    parser.add_argument("--input_col", type=str, default="corruption", help="Input column name")
    parser.add_argument("--output_col", type=str, default="target", help="Target column name")
    parser.add_argument("--resume-training", action="store_true", help="Resume training")

    args = parser.parse_args()

    CONFIG = {
        "train_path": args.train_path,
        "val_path": args.val_path,
        "tokenizer_path": args.tokenizer_path,
        "model_path": args.model_path,
        "metric_name": args.metric_name,
        "save_path": args.save_path,
        "save_model_path": args.save_model_path,
        "log_path": args.log_path,
        "input_col":args.input_col,
        "output_col":args.output_col,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_input_length": args.max_input_length,
        "max_target_length": args.max_target_length,
        "max_tokens": args.max_tokens,
        "prefix_col": args.prefix_col,
        "extra_tokens": args.extra_tokens,
        "resume_training": args.resume_training
    }

    special_tokens = CONFIG["extra_tokens"]
    if special_tokens == []:
        model, tokenizer = load_tools(CONFIG["model_path"])
    else:
        model, tokenizer = load_tools(CONFIG["model_path"], special_tokens=special_tokens)

    tokenized_dataset, metric = get_dataset(CONFIG["train_path"], CONFIG["val_path"], CONFIG["metric_name"])

    train_model(model, tokenizer, tokenized_dataset)


    """
    To run this script, use the command:

    python train_reconstructor.py --train_path data/aqua_train_v4_phase1_corrupted.csv --val_path data/aqua_val_corrupted_v4.csv --model_path facebook/bart-large --tokenizer_path facebook/bart-large --metric_name rouge --save_path models/reconstruction_bart_v4_phase1 --save_model_path models/reconstruction_bart_v5_phase1_model --log_path logs/reconstruction_bart_v5_phase1 --lr 7e-5 --batch_size 32 --epochs 10 --max_input_length 256 --max_target_length 256 --prefix_col prefix --input_col corruption --output_col target
    """
