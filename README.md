# IndRegBias: A Dataset for Studying Indian Regional Biases in English and Code-Mixed Social Media Comments
## Overview
This research examines the effectiveness of Large Language Models (LLMs) in detecting regional bias within social media comments in the Indian context. In a culturally diverse nation like India, regional biases can perpetuate stereotypes and create social divisions. This project addresses the challenge that many current LLMs are trained primarily on Western datasets, leading to a lack of awareness of the nuanced biases prevalent in India.
Our goal is to make LLMs aware of regional bias in Indian content and improve their detection performance. This work contributes a novel dataset, a methodology for bias detection, and provides insights into the strengths and limitations of current LLMs for this task.

## Dataset

A key contribution of this research is a new, culturally specific dataset of user-generated comments for detecting regional bias. The dataset is available through [dataset.zip](https://github.com/debby123p/IndRegBias-A-Dataset-for-Studying-Indian-Regional-Biases-in-English-and-Code-Mixed-Social-Media-Com/blob/main/Dataset.zip).

- Data Source: Comments collected from Reddit and YouTube.

- Final Size: After cleaning, the final dataset consists of 25,000 user comments.
  
- Annotation: A rigorous human annotation process was implemented, involving two groups of three annotators. Inter-annotator agreement was high, with a Cohen's kappa value of 0.91 for binary classification and 0.83 for multi-class classification.

### Annotation Schema
Each of the 25,000 comments was manually classified using a multi-level severity schema:


Level 1: Bias Identification: Is the comment a regional bias (1) or not (0). 


Level 2: Severity of Bias: The severity of the comment was rated as Mild (1), Moderate (2), or Severe (3).


Level 3: State Identification: The name of the Indian state being targeted in the comment was recorded.

### Data Analysis Highlights

These comments are collected from videos or subreddit pages belonging to different regions, where the languages are mixed, such as English, Hinglish (a mix of Hindi and English), a mix of Bengali and English, a mix of Malayalam and English, or Marathi and English; thus, we have a multilingual and code-mixed language dataset.

- Out of 25,000 comments,13,020 (52.1%) contained regional biases, and 11,980 (47.9%) contained non-regional biases.
  
- Severity breakdown of biased comments is as Mild (3,747 comments), Moderate (6,663 comments), and Severe (2,610 comments).

- The data also showed the distribution of comments across different regions of India, with a large number of biased comments found for South India (5,802 comments) and North India (3,906 comments).

- In the state-wise breakdown, Kerala emerged as the most represented state with 1,841 comments, followed by Goa (1,103) and West Bengal (1,072).

## Methodology

We deployed a series of experiments in the following setting with the Chain-of-thought (CoT) prompting technique for the classification of level-1 and level-2 annotations:

- Zero-Shot Setting

- Few-Shot Setting

- Fine-Tuning (Using Parameter-Efficient Fine-Tuning (PEFT) and Low-Rank Adaptation (LoRA))
  
      - Instruction-Based Supervised Fine-Tuning (SFT) for Binary Classification
  
      - Classification-Based Supervised Fine-Tuning (SFT) for Multi-class Classification 

### Models Evaluated
Eight prominent LLMs were selected for their instruction-following and reasoning abilities:

- qwen/Qwen3-8B

- qwen/qwen3-32B

- krutrim-ai-labs/Krutrim-2-instruct
  
- mistralai/Mistral-7B-Instruct-v0.3

- ai4bharat/Airavata

- sarvamai/sarvam-m

- mistralai/Mistral-Nemo-Base-2047

- meta-llama/Llama-3.2-3B 

- deepseek-ai/DeepSeek-R1-Distill-Llama-8B 

- google/gemma-1.1-7b-it

- google/gemini-2.5-pro

- microsoft/Phi-4-mini-reasoning(4b) 

## Results

### Results: Binary Classification 

1) Zero-Shot Results

   The results for the models are presented in the zero-shot setting in the table. The codes for which are available in the folder named zero-shot-binary-classification, which is in [zero-shot-binary-classification](https://github.com/debby123p/CAN-LARGE-LANGUAGE-MODELS-DETECT-INDIAN-REGIONAL-BIAS-DATASET-ANALYSIS/tree/main/zero_shot/zero-shot-binary-classification).
   
| | **Accuracy** | **Precision** | | **F1-Score** | |
| :--- | :---: | :---: | :---: | :---: | :---: |
| | | **Regional Bias (1)** | **Non-Regional Bias (0)** | **Regional Bias (1)** | **Non-Regional Bias (0)** |
| **Qwen/Qwen3-8B** | **0.74** | 0.71 | 0.80 | **0.78** | 0.69 |
| **Qwen/Qwen3-32B** | **0.74** | **0.79** | 0.69 | 0.72 | **0.75** |
| krutrim-ai-labs/Krutrim-2-instruct | 0.73 | 0.68 | 0.84 | **0.78** | 0.65 |
| mistralai/Mistral-7B-Instruct-v0.3 | 0.70 | 0.65 | 0.83 | 0.76 | 0.60 |
| google_gemini2.5_pro | 0.69 | 0.64 | 0.84 | 0.76 | 0.58 |
| ai4bharat/Airavata | 0.58 | 0.60 | 0.55 | 0.58 | 0.57 |
| sarvamai/sarvam-m | 0.57 | 0.63 | 0.53 | 0.49 | 0.62 |
| meta-llama/Llama-3.2-3B | 0.55 | 0.57 | 0.53 | 0.57 | 0.54 |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 0.54 | 0.53 | **0.87** | 0.69 | 0.09 |
| google/gemma-1.1-7b-it | 0.52 | 0.52 | 0.82 | 0.69 | 0.01 |
| mistralai/Mistral-Nemo12b | 0.53 | 0.53 | 0.68 | 0.69 | 0.10 |
| microsoft/Phi-4-mini-reasoning(4b) | 0.48 | 0.60 | 0.48 | 0.00 | 0.65 |

3) Few-Shot Results

   We conducted the few-shot experiments with a smaller number of support (different combinations of regional biases and non-regional biases) on a smaller dataset of 1000 comments for the model Qwen3_8b. The codes for which are available in [qwen_3_8b](https://github.com/debby123p/CAN-LARGE-LANGUAGE-MODELS-DETECT-INDIAN-REGIONAL-BIAS-DATASET-ANALYSIS/tree/main/few-shot).

| | **Exp-1** <br> (15 Non-Reg) | | **Exp-2** <br> (15 Reg) | | **Exp-3** <br> (20 Reg/20 Non) | | **Exp-4** <br> (20 Reg/10 Non) | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Metric** | **Zero** | **Few** | **Zero** | **Few** | **Zero** | **Few** | **Zero** | **Few** |
| _**Non-regional biases**_ | | | | | | | | |
| Precision | **0.84** | 0.72 | 0.81 | **0.88** | **0.85** | 0.82 | 0.82 | **0.84** |
| F1-Score | 0.70 | **0.78** | 0.70 | **0.75** | 0.69 | **0.77** | 0.70 | **0.75** |
| _**Regional Biases**_ | | | | | | | | |
| Precision | 0.70 | **0.82** | 0.69 | **0.73** | 0.68 | **0.76** | 0.71 | **0.75** |
| F1-Score | **0.78** | 0.74 | 0.76 | **0.81** | 0.78 | **0.79** | 0.79 | **0.81** |

   After looking through the inferences of the above experiment, we proceeded with experiments on the entire dataset with the type of support we have provided in Exp-2, Exp-3, and Exp-4, as we got improved performances in comparison to the zero-shot results. The codes for which are, which is available in [Few-shot-binary-classification](https://github.com/debby123p/CAN-LARGE-LANGUAGE-MODELS-DETECT-INDIAN-REGIONAL-BIAS-DATASET-ANALYSIS/tree/main/few-shot).

| | **Exp-1** <br> (50 R) | | **Exp-2** <br> (25R/25N) | | **Exp-3** <br> (30R/20N) | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Metric** | **ZS** | **FS** | **ZS** | **FS** | **ZS** | **FS** |
| _**Non-regional biases**_ | | | | | | |
| Precision | 0.80 | **0.87** | **0.79** | 0.71 | **0.80** | **0.80** |
| F1-Score | 0.69 | **0.75** | 0.69 | **0.78** | 0.69 | **0.79** |
| _**Regional Biases**_ | | | | | | |
| Precision | 0.70 | **0.74** | 0.71 | **0.85** | 0.71 | **0.84** |
| F1-Score | 0.77 | **0.82** | **0.77** | 0.75 | 0.78 | **0.81** |

   The few-shot results show improvement over the zero-shot results, as presented in the above table. The experiment with providing only 50 regional biases as support has outperformed other few-shot experiments. Also, a balanced and an unbalanced set of support shows improvement in performance in comparison to the zero-shot.   

5) Fine-Tuning Results

   Fine-tuned Qwen3_32b has emerged as the absolute best performer, achieving high reliability with F1-scores and precision nearing 0.90 for both bias categories. Followed by Qwen3_8b model, which has also performed better than zero-shot, achieving near 0.90 scores for precision and F1-scores. The codes for which are available in [binary-classification](https://github.com/debby123p/CAN-LARGE-LANGUAGE-MODELS-DETECT-INDIAN-REGIONAL-BIAS-DATASET-ANALYSIS/tree/main/fine-tuning).

| | **Qwen3-8B** | | **Qwen3-32B** | |
| :--- | :---: | :---: | :---: | :---: |
| **Metric** | **Zero** | **Fine-Tune** | **Zero** | **Fine-Tune** |
| _**Non-regional biases**_ | | | | |
| Precision | 0.796 | **0.888** | 0.688 | **0.870** |
| F1-Score | 0.706 | **0.920** | 0.742 | **0.936** |
| _**Regional Biases**_ | | | | |
| Precision | 0.692 | **0.902** | 0.790 | **0.900** |
| F1-Score | 0.772 | **0.904** | 0.724 | **0.902** |

### Results: Multi-Class Classification

1) Zero-Shot Results
   
We evaluated the top-performing models on severity classification (Mild, Moderate, Severe). Mistral-7B-v0.3 achieved the best performance, securing the highest precision for 'Mild' and 'Moderate' categories (0.64) and the top F1-score for 'Severe'. Qwen3-8B showed competitive results, particularly in the 'Severe' category, outperforming Mistral-Nemo. Conversely, Qwen3-32B struggled with 'Mild' and 'Severe' classifications, showing strength only in the 'Moderate' category. The codes for which are, which is available in [zero-shot-multi-classification](https://github.com/debby123p/IndRegBias-A-Benchmark-for-Studying-Indian-Regional-Biases-in-English-and-Code-Mixed-Social-Media-I/tree/main/zero_shot/zero-shot-multi-class-classification).

| Model | Mild (P) | Mild (F1) | Mod. (P) | Mod. (F1) | Severe (P) | Severe (F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-8B** | 0.57 | 0.57 | 0.59 | **0.66** | **0.44** | 0.22 |
| **Qwen3-32B** | 0.51 | 0.58 | 0.62 | 0.57 | 0.39 | 0.36 |
| **Mistral-7B** | **0.64** | **0.62** | **0.64** | 0.54 | 0.36 | **0.45** |
| **Mistral-Nemo** | 0.52 | 0.58 | 0.61 | 0.59 | 0.41 | 0.35 |
| **Krutrim-2** | 0.48 | 0.59 | 0.61 | 0.55 | 0.41 | 0.32 |

3) Few-Shot Results

We evaluated Mistral-7B-v0.3 using three few-shot strategies. Experiment B (30 balanced examples) yielded the best stability, significantly boosting the F1-score for 'Moderate' comments (0.54 $\to$ 0.64) and Precision for 'Mild' comments. Experiment C (imbalanced support) achieved the highest precision for 'Severe' bias (0.48) but suffered a significant drop in F1-score (0.31), indicating the model became overly conservative. Small support sets (Exp A) showed minimal improvement. The codes for which are, which is available in [Few-shot-multi-classification](https://github.com/debby123p/IndRegBias-A-Benchmark-for-Studying-Indian-Regional-Biases-in-English-and-Code-Mixed-Social-Media-I/tree/main/few-shot/Few-shot-multi-class-clasification).

| Experiment | Support Set | Class | Zero-Shot (P / F1) | Few-Shot (P / F1) |
| :--- | :--- | :--- | :---: | :---: |
| **Exp A** | 9 Examples (3/class) | Mild (1) | **0.64** / **0.62** | 0.58 / **0.62** |
| | | Mod (2) | **0.64** / 0.54 | **0.64** / **0.58** |
| | | Sev (3) | 0.36 / **0.45** | **0.40** / 0.44 |
| **Exp B** | 30 Examples (10/class) | Mild (1) | 0.64 / **0.62** | **0.66** / 0.61 |
| | | Mod (2) | **0.64** / 0.54 | **0.65** / **0.64** |
| | | Sev (3) | 0.36 / **0.45** | **0.41** / 0.41 |
| **Exp C** | 30 Examples (Imbalanced*) | Mild (1) | **0.64** / 0.62 | 0.62 / **0.63** |
| | *12 Mild, 6 Mod, 12 Sev | Mod (2) | **0.64** / 0.54 | 0.62 / **0.67** |
| | | Sev (3) | 0.35 / **0.45** | **0.48** / 0.31 |

4) Fine-Tuning Results

We fine-tuned Mistral-7B-v0.3 to address limitations in few-shot learning. The fine-tuned model demonstrated significant improvements across all severity classes compared to the zero-shot baseline. The most substantial gains were observed in the 'Severe' category (Precision increased from 0.360 to 0.586), confirming that training on curated data effectively enhances the detection of high-severity regional bias. The codes for which are, which is available in [multi-class classification](https://github.com/debby123p/IndRegBias-A-Benchmark-for-Studying-Indian-Regional-Biases-in-English-and-Code-Mixed-Social-Media-I/tree/main/fine-tuning/multi-class%20classification).

| Severity Level | Zero-Shot (Precision) | Zero-Shot (F1) | Fine-Tuning (Precision) | Fine-Tuning (F1) |
| :--- | :---: | :---: | :---: | :---: |
| **Mild (1)** | **0.788** | 0.620 | 0.756 | **0.676** |
| **Moderate (2)** | 0.628 | 0.608 | **0.662** | **0.716** |
| **Severe (3)** | 0.360 | 0.452 | **0.586** | **0.530** |
   
## Conclusion

The transition to IndRegBias marks a significant advancement in the computational analysis and mitigation of regional biases within the Indian context. By expanding the dataset to 25,000 comments, we provide code-mixed and multilingual comments. By ensuring class balance, we provide a robust benchmark that addresses the critical gap in the availability of such datasets. Our comprehensive evaluation of multiple models in zero-shot, few-shot, and fine-tuning settings shows the initial difficulty in inferring regional biases from the comments, which improves with model fine-tuning. The analysis of the data has shed more light on regional biases in India. Furthermore, the inclusion of Indic-centric models and a fine-grained analysis of the results for state/region-level performance highlights which geographic regions performed better or worse, providing a roadmap for developing more culturally aware and inclusive technologies. This work establishes a solid foundation for future research aimed at building fairer AI systems for diverse regional landscapes.

## 📝 Citation

If you find this dataset or code useful in your research, please cite our work:

```bibtex
@article{panda2026indregbias,
  title={IndRegBias: A Dataset for Studying Indian Regional Biases in English and Code-Mixed Social Media Comments},
  author={Panda, Debasmita and Anil, Akash and Shukla, Neelesh Kumar},
  journal={arXiv preprint arXiv:2601.06477},
  year={2026},
  url={[https://arxiv.org/abs/2601.06477](https://arxiv.org/abs/2601.06477)}
}
