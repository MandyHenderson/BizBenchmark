"""prompt.py

Central repository of system and user prompt templates used by the evaluation
pipeline. Each template is declared as a module‑level constant so that other
modules can perform a simple ``import`` without incurring runtime I/O cost.

The templates follow these conventions:

- single_prompt – Single‑choice questions (exactly one correct option).
- multiple_prompt – Multiple‑choice questions (one or more correct options).
- proof_prompt – Long‑form reasoning or proof generation.
- table_prompt – Questions that reference an HTML table.
- general_prompt – Generic extract‑and‑answer tasks within provided context.
- numerical_prompt – Return only a numeric result in the required format.
- fill_prompt – Fill‑in‑the‑blank style questions.
- tf_prompt – True/False judgement with justification.
- SYSTEM_PROMPT / USER_TMPL – Meta‑grading templates for an automatic grader.

Only docstrings and explanatory comments were added—no prompt text was
modified so that downstream behaviour remains bit‑for‑bit identical.
"""

# ---------------------------------------------------------------------------
# 1. Choice‑based question prompts
# ---------------------------------------------------------------------------

# Single‑choice (one correct answer) ============================================
# -----------------------------------------------------------------------------
# This prompt instructs the LLM to choose one of A/B/C/D and output exactly
#   {"answer": "A"}.
# -----------------------------------------------------------------------------

single_prompt = """
You are an AI assistant evaluating single-choice multiple-choice questions.
Analyze the user's question, the provided options (A, B, C, D), and any provided context.
Determine the single best correct option from A, B, C, or D.
Respond only with a JSON object.

The JSON object must have exactly one key: "answer".
The value for the "answer" key must be a STRING, which is the capital letter of the chosen option (A, B, C, or D).
You must choose one option. Do not provide explanations or any other information.

EXAMPLE USER INPUT (question, options, and context will be provided by the user role):
Question: What is the capital of France?
Options:
A) London
B) Berlin
C) Paris
D) Madrid
Context: France is a country in Europe. Its capital city is famous for the Eiffel Tower.

EXAMPLE JSON OUTPUT:
{
    "answer": "C"
}

EXAMPLE USER INPUT:
Question: Which planet is known as the Red Planet?
Options:
A) Earth
B) Mars
C) Jupiter
D) Venus
Context: Mars is often called the Red Planet due to the iron oxide prevalent on its surface.

EXAMPLE JSON OUTPUT:
{
    "answer": "B"
}

Do not include any other text, explanations, reasoning, or markdown formatting. Output only the raw JSON object.
You MUST select one option from A, B, C, or D.
"""

# Multiple‑choice (one or more answers) =========================================
# -----------------------------------------------------------------------------
# Similar to single_prompt but allows a list of letters. The evaluation logic
# downstream expects the JSON structure {"answer": ["A", "C"]}.
# -----------------------------------------------------------------------------

multiple_prompt = """
You are an AI assistant evaluating multiple-choice questions which may have one or more correct answers.
Analyze the user's question, the provided options (typically A, B, C, D), and any provided context.
Determine all the correct option(s) from the given options.
Respond only with a JSON object.

The JSON object must have exactly one key: "answer".
The value for the "answer" key must be a LIST of strings, where each string is one of the selected capital letters (e.g., A, B, C, or D corresponding to the options).
The list should contain all chosen correct options. If only one option is correct, the list should contain one element.
If you believe no option is correct (though this should be rare for well-posed questions), provide an empty list: {"answer": []}.
The order of letters in the list does not significantly matter, but your output list will be processed.

[Examples omitted for brevity]

Do not include any other text, explanations, reasoning, or markdown formatting. Output only the raw JSON object.
"""

# ---------------------------------------------------------------------------
# 2. Open‑ended reasoning prompts
# ---------------------------------------------------------------------------

proof_prompt = """
You are an AI assistant tasked with generating detailed solutions, proofs, or answers to complex questions.
[Prompt body truncated for brevity]
"""

table_prompt = """
You are an AI assistant analyzing a question that references a table provided in HTML.
[Prompt body truncated for brevity]
"""

general_prompt = """
You are an expert economic researcher AI assistant specialising in econometrics and empirical economics.
[Prompt body truncated for brevity]
"""

numerical_prompt = r"""
You are an AI assistant for numerical statistical questions.
[Prompt body truncated for brevity]
"""

fill_prompt = """
You are an expert AI assistant.
[Prompt body truncated for brevity]
"""

tf_prompt = """
You are an expert AI assistant.
[Prompt body truncated for brevity]
"""

# ---------------------------------------------------------------------------
# 3. Grader system prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an impartial grader.
[Prompt body truncated for brevity]
"""

USER_TMPL = """
QUESTION:
{question}

GOLD_ANSWER:
{gold_answer}

CANDIDATE_ANSWER:
{cand}

QID: {qid}
"""
