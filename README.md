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

## Actual Scenarios
#### 1. For Analyst task - Market Trend Analysis. The used dataset is the Financial Phrase Bank (FPB)
   
**Data Structure:**
- ID: a unique identifier for each instance.
- Sentence: a tokenized line from the dataset
- Label: a label corresponding to the class as a string: 'positive', 'negative' or 'neutral'

**Data Instances:**
```
{
  “ID": 1,
  "sentence": "Pharmaceuticals group Orion Corp reported a fall in its third-quarter earnings that were hit by larger expenditures on R&D and marketing .",
  "label": "negative"
}
```

**Process:**
The origin dataset covers a collection of 4840 sentences which are annotated by 16 people with adequate background knowledge on financial markets. Given the large number of  overlapping annotations, the origin dataset contains 4 alternative reference datasets based on the strength of majority agreement, including 'Sentences_50Agree', 'Sentences_66Agree', 'Sentences_75Agree' and 'Sentences_AllAgree'. We only utilize and process the subdataset 'Sentences_AllAgree' for our benchmark.
The final dataset includes 2242 instances with 100% annotator agreement.

**Data Path:**
```
Dataset/Actual_Task/Analyst_Market_Trend_Analysis/FPB.json
```

**Citation information:**
```
@article{Malo2014GoodDO,
  title={Good debt or bad debt: Detecting semantic orientations in economic texts},
  author={P. Malo and A. Sinha and P. Korhonen and J. Wallenius and P. Takala},
  journal={Journal of the Association for Information Science and Technology},
  year={2014},
  volume={65}
}
```

#### 2. For Trader task - Market Trend Analysis. The used dataset is the M&A dataset

**Data Structure:**
- ID: a unique identifier for each instance.
- Instruction: A constant instruction string shared by all records.
- Text: The news article or tweet from the dataset.
- Answer: a label corresponding to the class as a string: 'complete', or 'rumour'.


**Statistics:**
It contains 500 records.

**Data Path:**
```
Dataset/Actual_Task/Trader_Market_Trend_Analysis/MA.json
```

**Citation information:**
```
@inproceedings{yang_2020_generating,
  title={Generating Plausible Counterfactual Explanations for Deep Transformers in Financial Text Classification},
  author={Yang, Linyi and Kenny, Eoin and Ng, Tin Lok James and Yang, Yi and Smyth, Barry and Dong, Ruihai},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
  year={2020},
  pages={6150--6160}
}
```

#### 3. For Consultant task - Market Trend Analysis. The used dataset is the FOMC dataset

**Data Structure:**
- ID: a unique identifier for each instance.
- Specific instruction: a constant instruction string released by the central bank, describing the classification task.
- Text: a text containing an excerpt from the central bank's statement.
- Answer: a label corresponding to the class as a string: 'complete', or 'rumour'.

**Statistics:**
It contains 496 records.

**Data Path:**
```
Dataset/Actual_Task/Trader_Market_Trend_Analysis/FOMC.json
```

**Citation information:**
```
@inproceedings{shah2023trillion,
  title={Trillion Dollar Words: A New Financial Dataset, Task \& Market Analysis},
  author={Agam Shah and Suvan Paturi and Sudheer Chava},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year={2023},
  pages={6664--6679},
  publisher={Association for Computational Linguistics}
}
```

#### 4. For Analyst task - Fraud Detection. The used dataset is the CCFraud dataset

**Data Structure:**
- ID: a unique identifier for each instance.
- Text: a string that includes the PCA-transformed principal components (V1 to V28) as numerical variables and the transaction amount, describing a client.
- Answer: a label corresponding to the class as a string: 'yes', or 'no'.

**Statistics:**
It contains 7974 records.

**Data Path:**
```
Dataset/Actual_Task/Analyst_Market_Trend_Analysis/CCFraud.json
```

**Citation information:**
```
@misc{feng2023empowering,
      title={Empowering Many, Biasing a Few: Generalist Credit Scoring through Large Language Models}, 
      author={Duanyu Feng and Yongfu Dai and Jimin Huang and Yifang Zhang and Qianqian Xie and Weiguang Han and Alejandro Lopez-Lira and Hao Wang},
      year={2023},
      eprint={2310.00566},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

#### 5. For Consultant task - Fraud Detection. The used dataset is the Taiwan dataset

**Data Structure:**
- ID: a unique identifier for each instance.
- Text: a natural-language serialization of firm-level financial indicators from the Taiwan Economic Journal bankruptcy dataset, listing feature names and values (e.g., profitability, leverage, liquidity, turnover, growth, cash flow, and per-share metrics) for a single company.
- Answer: a label corresponding to the class as a string: 'yes', or 'no'.

**Statistics:**
It contains 4773 records.

**Data Path:**
```
Dataset/Actual_Task/Consultant_Market_Trend_Analysis/Taiwan_Economic_Journal.json
```

**Citation information:**
```
@misc{feng2023empowering,
      title={Empowering Many, Biasing a Few: Generalist Credit Scoring through Large Language Models}, 
      author={Duanyu Feng and Yongfu Dai and Jimin Huang and Yifang Zhang and Qianqian Xie and Weiguang Han and Alejandro Lopez-Lira and Hao Wang},
      year={2023},
      eprint={2310.00566},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

#### 6. For Analyst task - Financial Document Analysis. The used datasets include FNXL, FinQA, TATQA, FinRED

For this task, we have processed four dataset as each dataset is designed to extract different aspects of financial information. FinRED enables the identification of relationships between entities, essential for understanding connections within financial documents; FinQA and TATQA facilitate the extraction of precise numerical insights from contexts; FNXL specializes labeling numeric data, which helps understanding the roles of various fianncial figures.


https://huggingface.co/datasets/TheFinAI/flare-ner/viewer/default/train?row=0&views%5B%5D=train

-----

**Data Structure for FNXL:**
- ID: a unique identifier for each instance.
- Specific Instruction: a detailed instruction for financial labeling task, specifying the requirement to identify and assign semantic role labels to each token in financial sentences, with comprehensive label categories covering various financial concepts and accounting terms.
- Text: financial sentence extracted from corporate reports and financial documents, typically including numerical values, financial metrics, accounting terms, and temporal references that require semantic role identification and labeling.
- Answer: token-level semantic role annotations in the format of 'token:label' pairs.
- Label: array-formatted semantic role labels corresponding to each token in the input sentence.

**Statistics for FNXL:**
It contains 318 records.

**Data Path for FNXL:**
```
Dataset/Actual_Task/Analyst_Financial_Document_Analysis/FNXL.json
```

**Citation information for FNXL:**
```
@inproceedings{sharma2023financial,
  title={Financial numeric extreme labelling: A dataset and benchmarking},
  author={Sharma, Soumya and Khatuya, Subhendu and Hegde, Manjunath and Shaikh, Afreen and Dasgupta, Koustuv and Goyal, Pawan and Ganguly, Niloy},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
  pages={3550--3561},
  year={2023}
}
```
------

**Data Structure for FinQA:**
- ID: a unique identifier for each instance.
- Specific Instruction: a detailed instruction for financial question answering task, directing the model to answer based on the provided context.
- Question: a specific financial question requiring numerical calculations or factual answers.
- Answer: precise numerical values or textual responses to financial questions.

**Statistics for FinQA:**
It contains 6251 records.

**Data Path for FinQA:**
```
Dataset/Actual_Task/Analyst_Financial_Document_Analysis/FinQA.json
```

**Citation information for FinQA:**
```
@article{chen2021finqa,
  title={Finqa: A dataset of numerical reasoning over financial data},
  author={Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and Borova, Iana and Langdon, Dylan and Moussa, Reema and Beane, Matt and Huang, Ting-Hao and Routledge, Bryan and others},
  journal={arXiv preprint arXiv:2109.00122},
  year={2021}
}
```

-----

**Data Structure for TATQA:**
- ID: a unique identifier for each instance.
- Specific Instruction: a detailed instruction for table-based financial question answering task, directing the model to answer based on the provided tabular context.
- Question: a specific financial question requiring table interpretation and numerical reasoning, requiring understanding of tabular data structure and relationships between different financial categories and time periods.
- Answer: precise answers derived from table analysis.

**Statistics for TATQA:**
It contains 1668 records.

**Data Path for TATQA:**
```
Dataset/Actual_Task/Analyst_Financial_Document_Analysis/TATQA.json
```

**Citation information for TATQA:**
```
@article{zhu2021tat,
  title={TAT-QA: A question answering benchmark on a hybrid of tabular and textual content in finance},
  author={Zhu, Fengbin and Lei, Wenqiang and Huang, Youcheng and Wang, Chao and Zhang, Shuo and Lv, Jiancheng and Feng, Fuli and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:2105.07624},
  year={2021}
}
```
-----

**Data Structure for FinRED:**
- ID: a unique identifier for each instance.
- Specific Instruction: a detailed instruction for financial named entity recognition tasks, providing guidelines to identify and classify named entities from financial documents into three categories: persons (PER), organizations (ORG), and locations (LOC).
- Text: sentences extracted from financial agreements and SEC fillings.
- Answer: structured names entity annotations in the format of 'entity name, entity type' pairs.

**Statistics for FinRED:**
It contains 408 records.

**Data Path for FinRED:**
```
Dataset/Actual_Task/Analyst_Financial_Document_Analysis/FinRED.json
```

**Citation information for FinRED:**
```
@inproceedings{sharma2022finred,
  title={FinRED: A dataset for relation extraction in financial domain},
  author={Sharma, Soumya and Nayak, Tapas and Bose, Arusarka and Meena, Ajay Kumar and Dasgupta, Koustuv and Ganguly, Niloy and Goyal, Pawan},
  booktitle={Companion Proceedings of the Web Conference 2022},
  pages={595--597},
  year={2022}
}
```

#### 7. For Consultant task - Financial Document Analysis. The used dataset is the EDTSUM dataset

**Data Structure:**
- ID: a unique identifier for each instance.
- Specific instruction: a constant instruction string, requiring the model to perform abstractive summarization on the given text. 
- Text: a text containing complete financial press releases or business documents.
- Answer: The condensed summary of the original text, highlighting core events, key entities, and main motivations.

**Statistics:**
It contains 2000 records.

**Data Path:**
```
Dataset/Actual_Task/Consultant_Financial_Document_Analysis/EDTSUM.json
```

**Citation information:**
```
@inproceedings{zhou2021trede,
     author = {Zhihan Zhou and Liqian Ma and Han Liu},
     title = {Trade the Event: Corporate Events Detection for News-Based Event-Driven Trading},
     booktitle = {Findings of the Association for Computational Linguistics: ACL-IJCNLP2021},
     pages = {2114--2124},
     year = {2021},
     publisher = {Association for Computational Linguistics}
}
```

#### 8. For Trader task - Asset Pricing. The used dataset is CIKM18 dataset

**Data Structure:**
- ID: a unique identifier for each instance.
- Specific instruction: a constant instruction string for the asset pricing prediction task, requireing the model to predict whether the price will rise of fall on a specific date.
- Text: a text containing the stock price time-series data, and the twitter tweets related to stocks.
- Answer: a label corresponding to the class as a string: 'rise', or 'fall'.

**Statistics:**
It contains 12905 records.

**Data Path:**
```
Dataset/Actual_Task/Trader_Asset_Pricing/CIKM18.json
```

**Citation information:**
```
@inproceedings{wu2018hybrid,
     author = {Huizhe Wu and Wei Zhang and Weiwei Shen and Jun Wang},
     title = {Hybrid Deep Sequential Modeling for Social Text-Driven Stock Prediction},
     booktitle = {Proceedings of the27th ACM International Conference on Information and Knowledge Management},
     pages = {1627--1630},
     year = {2018},
     publisher = {Association for Computational Linguistics}
}
```

#### 9. For Analyst task - Risk Management. The used dataset is the SECQUE

**Data Structure:**
- ID: a unique identifier for each instance.
- Text：the revelant context passage provided to the model, extracted without headers from the source document.
- Question: the user query about the context that the model should answer.
- Answer: the ground-truth reference answer derived from the context.

**Statistics:**
It contains 85 records.

**Data Path:**
```
Dataset/Actual_Task/Analyst_Risk_Management/SECQUE.json
```

**Citation information:**
```
@article{noga2025secque,
  title={SECQUE: A Benchmark for Evaluating Real-World Financial Analysis Capabilities},
  author={Yoash, Noga Ben and Brief, Meni and Ovadia, Oded and Shenderovitz, Gil and Mishaeli, Moshik and Lemberg, Rachel and Sheetrit, Eitam},
  journal={arXiv preprint arXiv:2504.04596},
  year={2025}
}
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
