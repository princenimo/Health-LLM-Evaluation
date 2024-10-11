# Health LLM Evaluation 
This script runs the core components for evaluating a health-based language model on a question-answering task. The script processes the inference, extracts the answers, and calculates accuracy metrics for evaluation.

## Prerequisites
Ensure you have the following dependencies installed:

* Python 3.10
* Necessary Python packages:
  * torch
  * transformers
  * datasets
  * numpy
  * pandas
    
You can install the required packages using pip:

`pip install torch transformers datasets numpy pandas`

## Usage
To run the evaluation, use the following format:

`./mcq_inference_pipeline.sh -m <model_name_or_model_path> -d <data_path> -i <inferenced_result_dir> -p <prompt_type> -n <num_of_shots>`
### Parameters:

* -m: Path to the pre-trained model or the model's name if using a model from the HuggingFace Hub.
* -d: Path to the dataset that contains the health-related questions.
* -i: Directory to store the inference results and extracted answers.
* -p: The type of prompt to use (e.g., MCQ, open-ended).
* -n: Number of shots (examples) provided to the model during evaluation (zero-shot, few-shot, etc.).

### Example

`./mcq_inference_pipeline.sh -m "BioMistral/BioMistral-7B" -d "/path/to/health_dataset.json" -i "/path/to/results" -p "MCQ" -n 3`

## Process Overview
1. Running Inference
The script will run the model on the specified dataset and store the outputs in the specified directory:

`python3 /path/to/qa_inference.py $INFER_ARGS`

2. Extracting Answers
The script extracts the model's predicted answers and stores them in JSON format for further evaluation:

`python3 /path/to/eval_medqa.py $EVAL_ARGS`

3. Accuracy Calculation
Finally, the script calculates the accuracy of the model based on the extracted answers and the benchmark data:


`python3 /path/to/get_accuracy.py $REPORT_ARGS`

### Output
The inference results and extracted answers will be saved in the directory specified by the -i parameter.
A report containing the accuracy metrics will be generated and saved in the same directory.

### Directory Structure:
```
/path/to/results/
├── output/
│   ├── extracted_answers.json
│   └── accuracy_report.txt
└── answers/
    └── <answer_files>.json
```
## Contact
If you encounter any issues, please reach out to me.
