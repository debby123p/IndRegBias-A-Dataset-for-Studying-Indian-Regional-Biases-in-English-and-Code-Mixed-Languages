import os
import pandas as pd
import re
import torch
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

TARGET_GPU = "" # Target GPU node
BASE_DATA_PATH = "" # Dataset file path
OUTPUT_DIR = "" # Output Directory
COMMENT_COLUMN_NAME = "comment"
GROUND_TRUTH_COLUMN_NAME = "is RB?"
BATCH_SIZE = 16

# Input/Output files
INFERENCE_FILE_PATH = os.path.join(BASE_DATA_PATH, "")
RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, "")

MODEL_ID = "Qwen/Qwen3-8B" # Model ID

MODEL_PROMPT = """
You are an expert in identifying regional biases in social media comments about Indian states and regions. Your task is to classify whether a comment contains regional biases or not.

Task: Classify the given comment as either "REGIONAL BIAS" (1) or "NON-REGIONAL BIAS" (0).

Definition:
- REGIONAL BIAS (1): Comments that reinforce or propagate biases about Indian states, regions, or their people based on linguistic, cultural, economic, political, or infrastructural aspects. The comments can reflect either positive or negative biases towards specific states or regions.
- NON-REGIONAL BIAS (0): Comments that are neutral or factual without generalisations, or unrelated to regional characteristics.

Step-by-Step Analysis Process:
Step 1: Identify Regional References
Think: Does this comment mention or refer to:
- Specific Indian states (e.g., Bihar, Kerala, Punjab, etc.)
- Regional groups (e.g., South Indians, North Indians, etc.)
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

Your response must include a brief line of reasoning followed by the final classification in the format "Classification: [0 or 1]".
"""

def setup_environment():
    # Sets up GPU visibility and creates the output directory.
    print(f"Restricting execution to GPU: {TARGET_GPU}")
    
    if "cuda" in TARGET_GPU:
        try:
            if not os.path.exists(OUTPUT_DIR):
                print(f"Creating output directory: {OUTPUT_DIR}")
                os.makedirs(OUTPUT_DIR)
        except Exception as e:
            print(f"Error creating directory: {e}")

def load_model_and_tokenizer():
    # Loads model in the full model.
    print(f"Loading model: {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16, 
        device_map=TARGET_GPU
    )
    
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def parse_response(response):
    # Parses the model output for classification and reasoning
    reasoning_match = re.search(r"(.*?)Classification:", response, re.IGNORECASE | re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else response.strip()

    prediction = 0 # Default
    
    match = re.search(r'Classification:\s*([01])', response)
    if match:
        prediction = int(match.group(1))
    else:
        if re.search(r'REGIONAL BIAS|1', response, re.IGNORECASE):
            prediction = 1
        elif re.search(r'NON-REGIONAL BIAS|0', response, re.IGNORECASE):
            prediction = 0

    return reasoning, prediction

def generate_evaluation_outputs(csv_path):
    # Generates classification report and confusion matrix 
    print("Generating evaluation metrics...", flush=True)
    try:
        df = pd.read_csv(csv_path)
        y_true = df['true_label'].astype(int)
        y_pred = df['predicted_label'].astype(int)

        report = classification_report(y_true, y_pred, target_names=['NON-REGIONAL BIAS (0)', 'REGIONAL BIAS (1)'], zero_division=0)
        report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Classification Report for {MODEL_ID}\n\n{report}")
        print(f"Report saved to {report_path}")
        print(report)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['NON-REGIONAL BIAS (0)', 'REGIONAL BIAS (1)'],
                    yticklabels=['NON-REGIONAL BIAS (0)', 'REGIONAL BIAS (1)'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {MODEL_ID}')
        
        cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to {cm_path}")

    except Exception as e:
        print(f"Error generating evaluation outputs: {e}")

def main():
    setup_environment()
    
    print(f"Reading main dataset: {INFERENCE_FILE_PATH}")
    try:
        df = pd.read_csv(INFERENCE_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Input file not found at {INFERENCE_FILE_PATH}")
        return

    if COMMENT_COLUMN_NAME not in df.columns:
         raise ValueError(f"Column '{COMMENT_COLUMN_NAME}' not found in CSV.")

    print(f"Total comments to process: {len(df)}")

    processed_count = 0
    if os.path.exists(RESULTS_CSV_PATH):
        try:
            processed_count = len(pd.read_csv(RESULTS_CSV_PATH))
            print(f"Resuming from index {processed_count}...")
        except pd.errors.EmptyDataError:
            print("Results file empty. Starting from scratch.")
    
    if processed_count >= len(df):
        print("All comments processed.")
        generate_evaluation_outputs(RESULTS_CSV_PATH)
        return

    df_to_process = df.iloc[processed_count:]
    
    model, tokenizer = load_model_and_tokenizer()
    print(f"Starting inference on {len(df_to_process)} comments...")
    
    for i in tqdm(range(0, len(df_to_process), BATCH_SIZE), desc="Processing Batches"):
        batch_df = df_to_process.iloc[i:i+BATCH_SIZE]
        batch_prompts = []
    
        for _, row in batch_df.iterrows():
            comment_text = str(row[COMMENT_COLUMN_NAME])
            messages = [
                {"role": "system", "content": MODEL_PROMPT},
                {"role": "user", "content": f"Comment: \"{comment_text}\"\n\nBased on your analysis, provide your reasoning and final classification."}
            ]
            batch_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=4096 
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

        decoded_responses = tokenizer.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        batch_results = []
        for idx, response in enumerate(decoded_responses):
            reasoning, prediction = parse_response(response)
            original_row = batch_df.iloc[idx]
            batch_results.append({
                'comment': original_row[COMMENT_COLUMN_NAME],
                'true_label': original_row[GROUND_TRUTH_COLUMN_NAME] if GROUND_TRUTH_COLUMN_NAME in original_row else -1,
                'predicted_label': prediction,
                'model_response': reasoning
            })

        results_df = pd.DataFrame(batch_results)
        is_first_write = not os.path.exists(RESULTS_CSV_PATH)
        results_df.to_csv(RESULTS_CSV_PATH, mode='a', header=is_first_write, index=False)

    print("\nInference Complete.")
    
    if GROUND_TRUTH_COLUMN_NAME in df.columns:
        generate_evaluation_outputs(RESULTS_CSV_PATH)

if __name__ == "__main__":
    main()
