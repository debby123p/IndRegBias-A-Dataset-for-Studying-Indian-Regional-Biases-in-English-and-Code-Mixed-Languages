import os
import torch
from datasets import load_dataset, Dataset, ClassLabel
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback, 
    set_seed
)
import pandas as pd
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch.utils.data import WeightedRandomSampler
from scipy.special import softmax
import gc
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "" # Target GPU node 
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3" # Model ID 

DATASET_PATH = "" # Dataset file path
OUTPUT_ROOT_DIR = "" # Output Directory 

id2label = {0: "Mild", 1: "Moderate", 2: "Severe"}
label2id = {"Mild": 0, "Moderate": 1, "Severe": 2}
label_names = ["Mild", "Moderate", "Severe"]
label_ids = [0, 1, 2] 

SEED = 42
NUM_FOLDS = 5

set_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    prompt_template = "Analyze the following text and classify its regional bias as Mild, Moderate, or Severe.\n\nText: {}\n\nClassification:"
    full_texts = [prompt_template.format(str(c).strip()) for c in examples['comment']]
    return tokenizer(full_texts, truncation=True, max_length=512, padding="max_length")

def get_model():
    # Loads the model.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=3, id2label=id2label, label2id=label2id,
        dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    
    peft_config = LoraConfig(
        lora_alpha=32, lora_dropout=0.1, r=16, bias="none", task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, peft_config)

def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}

class WeightedTrainer(Trainer):
    def _get_train_sampler(self, dataset=None):
        return self.custom_sampler

def main():
    if not os.path.exists(OUTPUT_ROOT_DIR):
        os.makedirs(OUTPUT_ROOT_DIR)

    # Load Data
    print("Loading full dataset...")
    df_full = pd.read_csv(DATASET_PATH)
    df_full.rename(columns={'level-2': 'label'}, inplace=True)
    df_full['label'] = df_full['label'].astype(int) - 1  
    df_full.dropna(subset=['comment'], inplace=True)
    df_full['comment'] = df_full['comment'].astype(str).str.strip()
    df_full = df_full.reset_index(drop=True)

    # Create Folds
    print(f"Creating {NUM_FOLDS}-Fold Stratified Split...")
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    splits = list(skf.split(df_full, df_full['label']))
    
    cv_results_data = []
    global_predictions_map = {}

    # Loop through Folds
    for fold_idx in range(NUM_FOLDS):
        current_fold = fold_idx + 1
        print(f"\n\n=============================================")
        print(f"   PROCESSING FOLD {current_fold} / {NUM_FOLDS}")
        print(f"=============================================")
        
        fold_dir = os.path.join(OUTPUT_ROOT_DIR, f"fold_{current_fold}")
        os.makedirs(fold_dir, exist_ok=True)

        test_indices = splits[fold_idx][1] 
 
        val_fold_idx = (fold_idx + 1) % NUM_FOLDS
        val_indices = splits[val_fold_idx][1]

        all_indices = np.arange(len(df_full))
        exclude_indices = np.concatenate([test_indices, val_indices])
        train_indices = np.setdiff1d(all_indices, exclude_indices)
   
        train_df = df_full.iloc[train_indices].copy()
        val_df = df_full.iloc[val_indices].copy()
        test_df = df_full.iloc[test_indices].copy()
        
        print(f"Sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Save Train/Val/Test Datasets
        train_df.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(fold_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(fold_dir, "test.csv"), index=False)

        # Create HF Datasets
        ds_train = Dataset.from_pandas(train_df)
        ds_val = Dataset.from_pandas(val_df)
        ds_test = Dataset.from_pandas(test_df)
        
        for ds in [ds_train, ds_val, ds_test]:
            ds = ds.cast_column('label', ClassLabel(num_classes=3, names=label_names))
        
        tokenized_train = ds_train.map(tokenize_function, batched=True)
        tokenized_val = ds_val.map(tokenize_function, batched=True)
        tokenized_test = ds_test.map(tokenize_function, batched=True)
        
        for ds in [tokenized_train, tokenized_val, tokenized_test]:
            ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        # Weighted Sampler Setup
        train_labels = train_df['label']
        class_counts = train_labels.value_counts().sort_index()
        class_weights = 1.0 / class_counts
        weights = train_labels.map(class_weights).values
        sampler_weights = torch.DoubleTensor(weights)
        custom_sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights), replacement=True)

        # Initialize Model
        model = get_model()

        # Training Arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(fold_dir, "checkpoints"),
            num_train_epochs=10,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            optim="adamw_8bit",
            logging_steps=50,
            learning_rate=2e-5, 
            fp16=False,
            bf16=True,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="cosine",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=1,
            report_to="none"
        )

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)],
        )
        trainer.custom_sampler = custom_sampler
        
        # Training
        print(f"Starting training for Fold {current_fold}.")
        trainer.train()

        # Predict
        print(f"Predicting Fold {current_fold} Test Set.")
        output = trainer.predict(tokenized_test)
        probs = softmax(output.predictions, axis=1)
        pred_labels = np.argmax(output.predictions, axis=1)
        
        # Save Test Results
        df_res = test_df.copy()
        df_res['predicted_label'] = pred_labels + 1
        for i, name in enumerate(label_names):
            df_res[f'prob_{name}'] = probs[:, i]
        df_res.to_csv(os.path.join(fold_dir, "test_predictions.csv"), index=False)
        
        # Classification Report
        report = classification_report(output.label_ids, pred_labels, target_names=label_names, labels=label_ids)
        with open(os.path.join(fold_dir, "test_classification_report.txt"), "w") as f:
            f.write(report)

        for local_idx, global_idx in enumerate(test_indices):
            row_probs = probs[local_idx]
            global_predictions_map[global_idx] = row_probs

        cv_results_data.append({
            "fold": current_fold,
            "test_f1": output.metrics['test_f1'],
            "test_acc": output.metrics['test_accuracy']
        })
  
        
        # Cleanup
        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Final Aggregation
    print("\n\n=============================================")
    print(f"   FINAL EVALUATION (Combined Folds 1-5)")
    print(f"=============================================")
    final_probs = []
    final_preds = []
    
    sorted_indices = sorted(global_predictions_map.keys())

    if len(sorted_indices) != len(df_full):
        print(f"Warning: Discrepancy in processed counts. Processed {len(sorted_indices)} vs Total {len(df_full)}")
    
    for idx in sorted_indices:
        p = global_predictions_map[idx]
        final_probs.append(p)
        final_preds.append(np.argmax(p))

    final_probs = np.array(final_probs)
    final_preds = np.array(final_preds)

    df_final = df_full.iloc[sorted_indices].copy()
    df_final['predicted_label'] = final_preds + 1
    for i, name in enumerate(label_names):
        df_final[f'prob_{name}'] = final_probs[:, i]

    # Final Metrics
    report = classification_report(df_final['label'], final_preds, target_names=label_names, labels=label_ids)
    print(report)

    # Save Final Files
    df_final.to_csv(os.path.join(OUTPUT_ROOT_DIR, "full_dataset_cv_predictions.csv"), index=False)
    with open(os.path.join(OUTPUT_ROOT_DIR, "full_dataset_classification_report.txt"), "w") as f:
        f.write(report)

    # Summary Metrics CSV
    res_df = pd.DataFrame(cv_results_data).sort_values("fold")
    avg_row = res_df.mean(numeric_only=True)
    avg_row['fold'] = 'Average'
    res_df = pd.concat([res_df, pd.DataFrame([avg_row])], ignore_index=True)
    res_df.to_csv(os.path.join(OUTPUT_ROOT_DIR, "cv_metrics_summary.csv"), index=False)

    print("\n5-Fold Cross-Validation Complete!")

if __name__ == "__main__":
    main()
