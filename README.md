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
- **Hugging Face**: [https://huggingface.co/datasets/MandyHenderson/BizBenchmark](https://huggingface.co/datasets/MandyHenderson/BizBenchmark)

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

BizBenchmark question types organized by domain:

| Domain | Category | Subcategory | Parameter | Directory |
|--------|----------|-------------|-----------|-----------|
| **OM** | QA | Table QA | `table` | `Table_QA/` |
| | | General QA | `general` | `General_QA/` |
| | Choice | Single Choice | `single` | `Single_Choice/` |
| | | Multiple Choice | `multiple` | `Multiple_Choice/` |
| | T/F | --- | `tf` | `TF/` |
| **Economics** | QA | Table QA | `table` | `Table_QA/` |
| | | General QA | `general` | `General_QA/` |
| | | Financial News QA | `general` | `General_QA/` |
| | Choice | Single Choice | `single` | `Single_Choice/` |
| | | Multiple Choice | `multiple` | `Multiple_Choice/` |
| | T/F | --- | `tf` | `TF/` |
| **Finance** | QA | Table QA | `table` | `Table_QA/` |
| | | General QA | `general` | `General_QA/` |
| | | Financial News QA | `general` | `General_QA/` |
| | Choice | Single Choice | `single` | `Single_Choice/` |
| | | Multiple Choice | `multiple` | `Multiple_Choice/` |
| | T/F | --- | `tf` | `TF/` |
| **Statistics** | QA | General QA | `general` | `General_QA/` |
| | | Numerical QA | `numerical` | `Numerical_QA/` |
| | | Proof | `proof` | `Proof/` |
| | | Fill-in-the-blank | `fill` | `Fill-in-the-Blank/` |
| | Choice | Single Choice | `single` | `Single_Choice/` |
| | | Multiple Choice | `multiple` | `Multiple_Choice/` |
| | T/F | --- | `tf` | `TF/` |

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
```

## ⚙️ Model Configuration

BizBenchmark supports two evaluation modes:

### 🌐 API Mode (Default)
Configure your API keys in `model/model.py` and `evaluation.py`:
```python
API_KEY = "your-api-key-here"
BASE_URL = "https://api.deepseek.com/v1"
```

### 🖥️ Local Model Mode

For local model deployment, configure the model path in `model/local_model.py`:

```python
# Model configuration
MODEL_DIR = "/path/to/your/model"  # Update this path
DTYPE = torch.float16
TP_SIZE = 4  # Tensor parallelism size for multi-GPU
MAX_NEW_TOKENS = 8192
BATCH_SIZE = 14  # Adjust based on GPU memory
```

#### Local Model Requirements

**Dependencies:**
```bash
pip install torch transformers deepspeed
```

**Multi-GPU Setup (recommended):**
```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 main.py --model_type local [other args]

# Multi-node setup
torchrun --nnodes=2 --nproc_per_node=4 --master_addr="master_ip" --master_port=12345 \
    main.py --model_type local [other args]
```

**Single GPU:**
```bash
python main.py --model_type local [other args]
```

#### Performance Configuration

**Batch Size Tuning:**
- `BATCH_SIZE = 14`: Default for 24GB GPU
- `BATCH_SIZE = 8`: For 16GB GPU
- `BATCH_SIZE = 4`: For 12GB GPU

**Memory Optimization:**
- Use `torch.float16` for inference
- Adjust `MAX_NEW_TOKENS` based on requirements
- Monitor GPU memory usage with `nvidia-smi`

## 💽 Usage

### API Mode (Default)

#### Run Inference
```bash
python main.py \
    --dataset_path ../Dataset \
    --output_path ./result \
    --domain ECON \
    --model deepseek-chat \
    --question_type single \
    --temperature 0.2 \
    --top_p 0.95 \
    --model_type api
```

#### Run Evaluation
```bash
python evaluation.py \
    --eval_path ./result \
    --out_path ./eval \
    --domain ECON \
    --model deepseek-chat \
    --question_type single \
    --temperature 0.2 \
    --top_p 0.95 \
    --model_type api
```

### Local Model Mode

#### Run Inference (Single GPU)
```bash
python main.py \
    --dataset_path ../Dataset \
    --output_path ./result \
    --domain ECON \
    --model local-model \
    --question_type single \
    --temperature 0.2 \
    --top_p 0.95 \
    --model_type local
```

#### Run Inference (Multi-GPU)
```bash
torchrun --nproc_per_node=4 main.py \
    --dataset_path ../Dataset \
    --output_path ./result \
    --domain ECON \
    --model local-model \
    --question_type single \
    --temperature 0.2 \
    --top_p 0.95 \
    --model_type local
```

#### Run Evaluation (Local Model)
```bash
python evaluation.py \
    --eval_path ./result \
    --out_path ./eval \
    --domain ECON \
    --model local-model \
    --question_type single \
    --temperature 0.2 \
    --top_p 0.95 \
    --model_type local
```

### Batch Processing

#### Run All Domains and Question Types
```bash
# API mode
python main.py --batch_all --model_type api

# Local mode (multi-GPU recommended)
torchrun --nproc_per_node=4 main.py --batch_all --model_type local
```

#### Selective Batch Processing
```bash
# Specific domains
python main.py --batch_all --domains ECON FIN --model_type local

# Specific question types
python main.py --batch_all --question_types single multiple --model_type local

# Dry run (preview tasks)
python main.py --batch_all --dry_run --model_type local
```

### Parameters
- `--dataset_path`: Path to dataset directory (default: `../Dataset`)
- `--domain`: Business domain (ECON, FIN, OM, STAT) - **Required**
- `--model`: LLM model name
- `--question_type`: One of `single`, `multiple`, `tf`, `fill`, `numerical`, `proof`, `table`, `general`
- `--temperature`: Sampling temperature (0.0-2.0)
- `--top_p`: Nucleus sampling (0.0-1.0)
- `--model_type`: `api` (default) or `local`

## 📋 Output Structure

### Inference Results
```
result/
└── {domain}/                          # e.g., ECON, FIN, OM, STAT
    └── {question_type}/                # e.g., single, multiple, general
        └── {model_name}/               # e.g., deepseek-chat, local-model
            └── tem{temperature}/       # e.g., tem0.2, tem0.7
                └── top_k{top_p}/       # e.g., top_k0.95, top_k0.9
                    └── evaluation/
                        ├── Single_Choice_ECON_eval.json
                        ├── Single_Choice_ECON_eval.jsonl
                        └── ...
```

### Evaluation Results
```
eval/
└── {domain}/                          # e.g., ECON, FIN, OM, STAT
    └── {question_type}/                # e.g., single, multiple, general
        └── {model_name}/               # e.g., deepseek-chat, local-model
            └── tem{temperature}/       # e.g., tem0.2, tem0.7
                └── top_k{top_p}/       # e.g., top_k0.95, top_k0.9
                    ├── Single_Choice_ECON/
                    │   ├── Single_Choice_ECON_evaluated_by_llm.json
                    │   ├── Single_Choice_ECON_evaluation_log.jsonl
                    │   └── summary/
                    │       └── Single_Choice_ECON_summary.json
                    └── ...
```

### Example Complete Path
```bash
# Inference output
result/ECON/single/local-model/tem0.2/top_k0.95/evaluation/Single_Choice_ECON_eval.json

# Evaluation output  
eval/ECON/single/local-model/tem0.2/top_k0.95/Single_Choice_ECON/Single_Choice_ECON_evaluated_by_llm.json
```

## 🔧 Configuration

### API Mode Configuration
Adjust in `model/model.py`:
```python
MAX_WORKERS = 1  # Increase based on API limits
```

Adjust in `evaluation.py`:
```python
MAX_CONCURRENCY = 700  # Adjust based on API limits
```

### Local Model Configuration
Adjust in `model/local_model.py`:
```python
BATCH_SIZE = 14  # Batch size for inference
MAX_NEW_TOKENS = 8192  # Maximum tokens to generate
TP_SIZE = 4  # Tensor parallelism size
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

### API Mode
**API rate limits**: Reduce `MAX_WORKERS` and `MAX_CONCURRENCY`

**JSON errors**: Automatic repair via `json-repair` library

### Local Model Mode
**GPU memory issues**: 
- Reduce `BATCH_SIZE` in `local_model.py`
- Use smaller model or lower precision
- Monitor with `nvidia-smi`

**Model loading errors**:
- Verify `MODEL_DIR` path exists
- Check model format compatibility
- Ensure sufficient disk space

**Multi-GPU issues**:
- Verify CUDA and NCCL installation
- Check GPU visibility with `nvidia-smi`
- Use `CUDA_VISIBLE_DEVICES` if needed

**Resume issues**: Check that parameters match exactly between inference and evaluation

## 📈 Example Workflows

### Quick Evaluation
```bash
# API mode
python main.py --domain ECON --question_type single --model gpt-4
python evaluation.py --domain ECON --question_type single --model gpt-4

# Local mode
python main.py --domain ECON --question_type single --model local-model --model_type local
python evaluation.py --domain ECON --question_type single --model local-model --model_type local
```

### Multi-domain Evaluation
```bash
# API mode
for domain in ECON FIN OM STAT; do
    python main.py --domain $domain --question_type single --model gpt-4
    python evaluation.py --domain $domain --question_type single --model gpt-4
done

# Local mode (batch processing recommended)
torchrun --nproc_per_node=4 main.py --batch_all --domains ECON FIN OM STAT --model_type local
```

### Parameter Sweep
```bash
# API mode
for temp in 0.1 0.2 0.5; do
    for model in gpt-4 deepseek-chat; do
        for domain in ECON FIN OM STAT; do
            python main.py --domain $domain --model $model --temperature $temp --question_type single
            python evaluation.py --domain $domain --model $model --temperature $temp --question_type single
        done
    done
done

# Local mode
for temp in 0.1 0.2 0.5; do
    torchrun --nproc_per_node=4 main.py --batch_all --temperature $temp --model_type local
done
```

### Performance Monitoring
```bash
# Monitor GPU usage during local inference
watch -n 1 nvidia-smi

# Check process status
ps aux | grep python

# Monitor disk usage
df -h ./result ./eval
```

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch  
3. Submit a pull request

For dataset questions, see the main BizBenchmark repository.

---

**Architecture:**
- `main.py`: Command-line interface and model initialization
- `model/model.py`: API-based LLM inference with concurrent processing
- `model/local_model.py`: Local model inference with DeepSpeed support
- `evaluation.py`: LLM-based grading with API/local model support
- `dataloader/dataloader.py`: Data loading with resume functionality
- `utils/prompt.py`: Question-specific prompts
- `utils/utils.py`: Utility functions 