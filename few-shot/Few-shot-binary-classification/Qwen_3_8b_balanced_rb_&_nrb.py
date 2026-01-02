import os
import pandas as pd
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import sys

BATCH_SIZE = 16
TARGET_GPU = "" # Target GPU node
MODEL_ID = "Qwen/Qwen3-8B" # Model ID
BASE_DATA_PATH = "" # Base Data Path
OUTPUT_DIR = "" # Output Directory

# Paths for the specific few-shot learning examples
NON_REGIONAL_BIAS_EXAMPLES_PATH = os.path.join(BASE_DATA_PATH, "")
REGIONAL_BIAS_EXAMPLES_PATH = os.path.join(BASE_DATA_PATH, "")

# Inference File
INFERENCE_FILENAME = "" 
INFERENCE_FILE_PATH = os.path.join(BASE_DATA_PATH, INFERENCE_FILENAME)

# Saving the results
RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, "classification_results_final.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "classification_report.txt")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

MODEL_PROMPT = """You are an expert in identifying regional biases in social media comments about Indian states and regions. Your task is to classify whether a comment contains regional biases or not.

Task: Classify the given comment as either "REGIONAL BIAS" (1) or "NON-REGIONAL BIAS" (0).

Definition:
- REGIONAL BIAS (1): Comments that reinforce or propagate biases about Indian states, regions, or their people based on linguistic, cultural, economic, political, or infrastructural aspects. The comments can reflect either positive or negative biases towards specific states or regions.
- NON-REGIONAL BIAS (0): Comments that are neutral or factual without generalisations, or unrelated to regional characteristics.

Step-by-Step Analysis Process:
Step 1: Identify Regional References
Think: Does this comment mention or refer to:
- Specific Indian states (e.g., Bihar, Kerala, Punjab, etc.)
- Regional groups (e.g., South Indians, North Indians, Biharis, etc.)
- Cities or regions within India
- Language communities within India

Step 2: Check for Elements reinforcing biases
Look for these patterns:
- Generalisations about people from a state or a regional group ("All X are Y")
- Assumptions about state/regional characteristics
- Comparative statements that imply superiority/inferiority
- Overgeneralized cultural, linguistic, economic, political, or infrastructural claims

Step 3: Assess the Nature of the Statement
Consider:
- Is this a factual observation or a generalised assumption?
- Does it reinforce existing biases?
- Is it based on a broad generalisation?
- Does it perpetuate divisions?

Step 4: Final Classification
Based on the analysis above, classify as:
- REGIONAL BIAS (1) if the comment reinforces regional biases or stereotypes
- NON-REGIONAL BIAS (0) if the comment is neutral, factual, or doesn't contain regional bias.

Your response must include a brief line of reasoning followed by the final classification in the format "Classification: [0 or 1]"."""

def get_device(target_device: str):
    # Sets up GPU visibility and creates the output directory.
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if "cuda" in target_device:
        try:
            device_idx = int(target_device.split(":")[-1])
            if device_idx >= torch.cuda.device_count():
                print(f"Device {target_device} not found. Switching to cuda:0", flush=True)
                return torch.device("cuda:0")
        except ValueError:
            pass 
    return torch.device(target_device)

def load_model_and_tokenizer(model_id: str, device: str):
    # Handles authentication and loads the full model.
    print(f"Loading model: {model_id} in 16-bit precision on {device}.", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16, 
        device_map=device
    )
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_few_shot_examples(regional_bias_path: str, non_regional_bias_path: str, total_examples: int) -> tuple[str, set]:
    # Loads the few_shot_examples
    print(f"Loading {total_examples} balanced examples for the prompt.", flush=True)
    num_examples_per_class = total_examples // 2
    all_examples = []

    # Load Regional Bias
    try:
        df_regional = pd.read_csv(regional_bias_path)
        n_samples = min(num_examples_per_class, len(df_regional[df_regional['level-1'] == 1]))
        regional_samples = df_regional[df_regional['level-1'] == 1].sample(n=n_samples, random_state=42)
        for _, row in regional_samples.iterrows():
            all_examples.append({'comment': str(row['comment']), 'label': 1})
        print(f"   Success. Loaded {len(regional_samples)} regional examples.", flush=True)
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Could not find regional bias file: {regional_bias_path}", flush=True)
        sys.exit(1)

    # Load Non-Regional Bias
    try:
        df_non_regional = pd.read_csv(non_regional_bias_path)
        n_samples = min(num_examples_per_class, len(df_non_regional[df_non_regional['level-1'] == 0]))
        non_regional_samples = df_non_regional[df_non_regional['level-1'] == 0].sample(n=n_samples, random_state=42)
        for _, row in non_regional_samples.iterrows():
            all_examples.append({'comment': str(row['comment']), 'label': 0})
        print(f"   Success. Loaded {len(non_regional_samples)} non-regional examples.", flush=True)
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Could not find non-regional bias file: {non_regional_bias_path}", flush=True)
        sys.exit(1)

    random.seed(42)
    random.shuffle(all_examples)

    examples_str = ""
    used_comments_set = set()
    for example in all_examples:
        comment = example['comment'].strip()
        label = example['label']
        used_comments_set.add(comment)
        reasoning = "This is an example of a comment with regional bias." if label == 1 else "This is an example of a comment with no regional bias."
        examples_str += f"\n--- Example ---\nComment: \"{comment}\"\nReasoning: {reasoning}\nClassification: {label}\n--- End Example ---\n"

    return examples_str, used_comments_set

def parse_response(response: str) -> tuple[str, int]:
    # Robustly parses a single model response text to ensure a 0 or 1 output.
    reasoning_match = re.search(r"(.*?)Classification:", response, re.IGNORECASE | re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else response.strip()

    if re.search(r"Classification:?\s*1", response, re.IGNORECASE):
        return reasoning, 1
    if re.search(r"Classification:?\s*0", response, re.IGNORECASE):
        return reasoning, 0

    return reasoning, 0

def main():
    # Main function to orchestrate the entire classification process.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device_obj = get_device(TARGET_GPU)
    device_str = "cuda:0" if "cuda" in str(device_obj) and ":" not in str(device_obj) else str(device_obj)

    print(f"Using device: {device_str}", flush=True)
    
    # Load Data
    few_shot_prompt, used_comments = load_few_shot_examples(
        regional_bias_path=REGIONAL_BIAS_EXAMPLES_PATH,
        non_regional_bias_path=NON_REGIONAL_BIAS_EXAMPLES_PATH,
        total_examples=50 
    )
    
    print(f"\nLoading main dataset from: {INFERENCE_FILE_PATH}", flush=True)
    try:
        df_full = pd.read_csv(INFERENCE_FILE_PATH)
    except FileNotFoundError:
        print(f"FATAL ERROR: Could not find file {INFERENCE_FILE_PATH}", flush=True)
        return

    print(f"Removing {len(used_comments)} few-shot examples from inference set.", flush=True)
    df = df_full[~df_full['comment'].astype(str).str.strip().isin(used_comments)].copy()

    processed_count = 0
    if os.path.exists(RESULTS_CSV_PATH):
        try:
            existing_results = pd.read_csv(RESULTS_CSV_PATH)
            processed_count = len(existing_results)
            print(f"--> Found existing progress file with {processed_count} completed comments.", flush=True)
        except pd.errors.EmptyDataError:
            print("--> Progress file exists but is empty. Starting from scratch.", flush=True)
            processed_count = 0
    else:
        print("--> No existing progress file. Starting from scratch.", flush=True)

    if processed_count >= len(df):
        print("All comments have already been processed! Skipping inference.", flush=True)
        df_to_process = pd.DataFrame()
    else:
        df_to_process = df.iloc[processed_count:]
        print(f"--> Resuming from index {processed_count}. Remaining comments: {len(df_to_process)}", flush=True)

    # Load Model 
    if not df_to_process.empty:
        model, tokenizer = load_model_and_tokenizer(MODEL_ID, device=device_str)
        
        # Inference Loop
        with torch.no_grad():
            for i in tqdm(range(0, len(df_to_process), BATCH_SIZE), desc="Classifying Batches", mininterval=1.0):
                batch_df = df_to_process.iloc[i:i+BATCH_SIZE]
                batch_results = []
                batch_prompts = []
 
                for _, row in batch_df.iterrows():
                    comment_text = str(row['comment'])
                    full_prompt = (f"{few_shot_prompt}\n"
                                   f"--- Classify the following comment ---\n"
                                   f"Comment: \"{comment_text}\"")
                    messages = [{"role": "system", "content": MODEL_PROMPT}, {"role": "user", "content": full_prompt}]
                    templated_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    batch_prompts.append(templated_prompt)

                # Tokenize
                inputs = tokenizer(
                    batch_prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=4096
                ).to(model.device)

                # Generate
                generated_ids = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=512, 
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=0.0
                )
                
                decoded_responses = tokenizer.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                # Parse
                for idx, raw_response in enumerate(decoded_responses):
                    reasoning, prediction = parse_response(raw_response)
                    original_row = batch_df.iloc[idx]
                    batch_results.append({
                        'comment': original_row['comment'],
                        'true_label': original_row['level-1'],
                        'predicted_label': prediction,
                        'model_response': reasoning
                    })

                batch_results_df = pd.DataFrame(batch_results)

                is_first_write = not os.path.exists(RESULTS_CSV_PATH)
                batch_results_df.to_csv(RESULTS_CSV_PATH, mode='a', header=is_first_write, index=False)
    
    print(f"\nAll processing complete. Full results are in {RESULTS_CSV_PATH}", flush=True)

    if os.path.exists(RESULTS_CSV_PATH):
        try:
            full_results_df = pd.read_csv(RESULTS_CSV_PATH)
            
            y_true = full_results_df['true_label'].astype(int)
            y_pred = full_results_df['predicted_label'].astype(int)
            
            report = classification_report(y_true, y_pred, target_names=['NON-REGIONAL BIAS (0)', 'REGIONAL BIAS (1)'], zero_division=0)
            
            with open(REPORT_PATH, 'w') as f:
                f.write("Classification Report\n=====================\n\n" + report)
            print("\nClassification Report:\n", report, flush=True)
            
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['NON-REGIONAL BIAS (0)', 'REGIONAL BIAS (1)'],
                        yticklabels=['NON-REGIONAL BIAS (0)', 'REGIONAL BIAS (1)'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.savefig(CONFUSION_MATRIX_PATH)
            print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}", flush=True)
        except Exception as e:
            print(f"Error generating report: {e}")

if __name__ == "__main__":
    main()
