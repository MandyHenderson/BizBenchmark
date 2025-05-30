# 📊 BizBenchmark

<p align="center">
    <a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/Python-3.8+-1f425f.svg?color=purple">
    </a>
</p>

---

Code and data for evaluating large language models on business decision-making, analysis, and instruction.

## 🔗 Dataset Access

- **GitHub**: [https://github.com/MandyHenderson/BizBenchmark](https://github.com/MandyHenderson/BizBenchmark)
- **Hugging Face**: [https://huggingface.co/datasets/CatherineHao/BizBench](https://huggingface.co/datasets/CatherineHao/BizBench)

## 👋 Overview

BizBenchmark is a comprehensive benchmark for evaluating large language models in applied business contexts across four domains: Economics (ECON), Finance (FIN), Operations Management (OM), and Statistics (STAT).

The benchmark consists of over 31,228 high-quality examples covering a wide range of topics and task formats, designed to assess both theoretical understanding and practical reasoning. BizBenchmark provides a robust framework for evaluating LLMs' capabilities in business decision-making, analysis, and instruction.

Given a *business question* and *context*, a language model is tasked with generating an *answer* that demonstrates business reasoning capabilities.

To access BizBenchmark, copy and run the following code:
```python
from datasets import load_dataset
bizbench = load_dataset('MandyHenderson/BizBenchmark')
```

## 🧪 Question Types

**Four business domains**: Economics (ECON), Finance (FIN), Operations Management (OM), Statistics (STAT)

BizBenchmark supports 8 question types:

| Type | Parameter | Directory | Domain Files | Description |
|------|-----------|-----------|--------------|-------------|
| Single Choice | `single` | `Single_Choice/` | ECON, FIN, OM, STAT | Choose one correct answer from A/B/C/D |
| Multiple Choice | `multiple` | `Multiple_Choice/` | ECON, FIN, OM, STAT | Choose one or more correct answers |
| True/False | `tf` | `TF/` | ECON, FIN, OM, STAT | True/False with justification |
| Fill-in-the-Blank | `fill` | `Fill-in-the Blank/` | STAT | Complete missing information |
| Numerical | `numerical` | `Numerical_QA/` | STAT | Quantitative calculations |
| Proof | `proof` | `Proof/` | STAT | Mathematical derivations |
| Table | `table` | `Table_QA/` | ECON, FIN, OM | Data interpretation from tables |
| General | `general` | `General_QA/` | ECON, FIN, OM, STAT | Open-ended analysis |

## 📋 Data Format

### Single Choice Example
```json
{
  "qid": "financial-single-choice-123",
  "question": "What is the primary goal of portfolio diversification?",
  "gold_answer": "A",
  "options": ["A) Risk reduction", "B) Return maximization", "C) Cost minimization", "D) Tax optimization"],
  "question_context": "Portfolio theory suggests..."
}
```

### General Q&A Example
```json
{
  "qid": "Economic-Report-662-0-0-3",
  "question": "Calculate the implied valuation of the IPO given the conditions.",
  "gold_answer": "The implied valuation depends on shares offered and price per share...",
  "question_context": "Conference call transcript discussing IPO details..."
}
```

### Table Analysis Example
```json
{
  "qid": "econ-table-3619-2",
  "question": "Test whether the reform effect is statistically significant at 1% level.",
  "gold_answer": "t-statistic = 0.040/0.005 = 8.0. Since 8.0 > 2.576, reject null hypothesis...",
  "formula_context": "OLS regression: Y = α + β Reform + controls",
  "table_html": "<table>...</table>"
}
```

## 🚀 Set Up

Clone and install:
```bash
git clone https://github.com/MandyHenderson/BizBenchmark
cd BizBenchmark
pip install -r requirements.txt
cd LLMBench
```

Configure your API keys in `model/model.py` and `evaluation.py`:
```python
API_KEY = "your-api-key-here"
BASE_URL = "https://api.deepseek.com/v1"
```

## 💽 Usage

### Run Inference
```bash
python main.py \
    --dataset_path ./dataset \
    --output_path ./result \
    --model deepseek-chat \
    --question_type single \
    --temperature 0.2 \
    --top_p 0.95
```

### Run Evaluation
```bash
python evaluation.py \
    --eval_path ./result \
    --out_path ./eval \
    --model deepseek-chat \
    --question_type single \
    --temperature 0.2 \
    --top_p 0.95
```

### Parameters
- `--model`: LLM model name
- `--question_type`: One of `single`, `multiple`, `tf`, `fill`, `numerical`, `proof`, `table`, `general`
- `--temperature`: Sampling temperature (0.0-2.0)
- `--top_p`: Nucleus sampling (0.0-1.0)

## 📋 Output Structure

### Inference Results
```
result/
└── {question_type}/                    # e.g., single, multiple, general
    └── {model_name}/                   # e.g., deepseek-chat, gpt-4
        └── tem{temperature}/           # e.g., tem0.2, tem0.7
            └── top_k{top_p}/           # e.g., top_k0.95, top_k0.9
                └── evaluation/
                    ├── Single_Choice_ECON_eval.json
                    ├── Single_Choice_FIN_eval.json
                    ├── Single_Choice_OM_eval.json
                    ├── Single_Choice_STAT_eval.json
                    ├── Single_Choice_ECON_eval.jsonl
                    ├── Single_Choice_FIN_eval.jsonl
                    ├── Single_Choice_OM_eval.jsonl
                    └── Single_Choice_STAT_eval.jsonl
```

### Evaluation Results
```
eval/
└── {question_type}/                    # e.g., single, multiple, general
    └── {model_name}/                   # e.g., deepseek-chat, gpt-4
        └── tem{temperature}/           # e.g., tem0.2, tem0.7
            └── top_k{top_p}/           # e.g., top_k0.95, top_k0.9
                ├── Single_Choice_ECON/
                │   ├── Single_Choice_ECON_evaluated_by_llm.json
                │   ├── Single_Choice_ECON_evaluation_log.jsonl
                │   └── summary/
                │       └── Single_Choice_ECON_summary.json
                ├── Single_Choice_FIN/
                │   ├── Single_Choice_FIN_evaluated_by_llm.json
                │   ├── Single_Choice_FIN_evaluation_log.jsonl
                │   └── summary/
                │       └── Single_Choice_FIN_summary.json
                ├── Single_Choice_OM/
                │   ├── Single_Choice_OM_evaluated_by_llm.json
                │   ├── Single_Choice_OM_evaluation_log.jsonl
                │   └── summary/
                │       └── Single_Choice_OM_summary.json
                └── Single_Choice_STAT/
                    ├── Single_Choice_STAT_evaluated_by_llm.json
                    ├── Single_Choice_STAT_evaluation_log.jsonl
                    └── summary/
                        └── Single_Choice_STAT_summary.json
```

### Example Complete Path
```bash
# Inference output
result/single/deepseek-chat/tem0.2/top_k0.95/evaluation/Single_Choice_ECON_eval.json

# Evaluation output  
eval/single/deepseek-chat/tem0.2/top_k0.95/Single_Choice_ECON/Single_Choice_ECON_evaluated_by_llm.json
```

## 🔧 Configuration

### Concurrency
Adjust in `model/model.py`:
```python
MAX_WORKERS = 1  # Increase based on API limits
```

Adjust in `evaluation.py`:
```python
MAX_CONCURRENCY = 700  # Adjust based on API limits
```

### Resume
The benchmark automatically resumes from checkpoints. Simply re-run the same command.

### Custom Prompts
Edit templates in `utils/prompt.py` for different question types.

## 📊 Results

Sample evaluation output:
```json
{
  "total_items_successfully_graded_by_llm_grader": 150,
  "items_graded_as_correct_by_llm_grader": 127,
  "accuracy_according_to_llm_grader": 0.8467
}
```

## 🛠️ Troubleshooting

**API rate limits**: Reduce `MAX_WORKERS` and `MAX_CONCURRENCY`

**JSON errors**: Automatic repair via `json-repair` library

**Resume issues**: Check that parameters match exactly between inference and evaluation

## 📈 Example Workflows

### Quick evaluation:
```bash
# Single domain
python main.py --question_type single --model gpt-4
python evaluation.py --question_type single --model gpt-4

# Multiple domains
for qtype in single multiple general; do
    python main.py --question_type $qtype --model gpt-4
    python evaluation.py --question_type $qtype --model gpt-4
done
```

### Parameter sweep:
```bash
for temp in 0.1 0.2 0.5; do
    for model in gpt-4 deepseek-chat; do
        python main.py --model $model --temperature $temp --question_type single
        python evaluation.py --model $model --temperature $temp --question_type single
    done
done
```

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch  
3. Submit a pull request

For dataset questions, see the main BizBenchmark repository.

---

**Architecture:**
- `main.py`: Command-line interface
- `model/model.py`: LLM inference with concurrent processing
- `evaluation.py`: LLM-based grading
- `dataloader/dataloader.py`: Data loading with resume
- `utils/prompt.py`: Question-specific prompts
- `utils/utils.py`: Utility functions 
