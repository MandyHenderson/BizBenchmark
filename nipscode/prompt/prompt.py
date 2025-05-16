import sys
import os
import json # For formatting the TF prompt example and potentially others

# Add the parent directory to sys.path to allow import from dataloader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from dataloader import loader
except ImportError:
    print("Error: Could not import 'loader' from 'dataloader'. Ensure dataloader/loader.py exists and sys.path is correct.")
    def load_all_questions(data_dir):
        print(f"[Warning] Using dummy load_all_questions. Real data loading failed for {data_dir}")
        return [], {}

def format_options(options_list):
    if not options_list or not isinstance(options_list, list):
        return "No options provided or options are in an invalid format."
    return "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options_list)])

def format_table_html(table_html_data):
    if not table_html_data:
        return "No table HTML provided."
    if isinstance(table_html_data, dict):
        return "\n".join([f"Table {name}:\n{html}" for name, html in table_html_data.items()])
    elif isinstance(table_html_data, str):
        return table_html_data
    return "Table HTML is in an invalid format."

def format_prompt(question_data: dict) -> str:
    source_type = question_data.get("question_source_type", "Unknown")
    question_text = question_data.get("question", "No question text provided.")
    options = question_data.get("options")
    table_html = question_data.get("table_html")
    formula_context = question_data.get("formula_context")
    question_context = question_data.get("question_context")
    background_text = question_data.get("background_text")
    # title = question_data.get("heading", "") # For TF, if title is in 'heading'
    # If the title for TF questions comes from a different field, adjust here.
    # Assuming 'heading' might contain a title for TF, or a general title.
    title_for_tf = question_data.get("heading", "[No Title Provided]") 

    prompt = ""
    # Common instruction for JSON output
    json_instruction_main = (
        "Respond with **exactly** one JSON object, with no markdown fences or other text outside the JSON structure. "
        "The JSON object must conform to the following structure:\n"
    )

    if source_type == "TF":
        context_paragraphs = question_context or background_text or "[No Context Paragraphs Provided]"
        
        # This prompt for TF is quite specific and good, largely retaining its structure.
        # The key is that the MODEL should output JSON.
        prompt_structure = {
            "task_description": (
                "You are an expert financial analyst AI assistant.\n"
                "The user will provide context consisting of a title and paragraphs extracted from a financial research paper.\n"
                "The user will then provide a statement (question) based on the text.\n\n"
                "Your task is to:\n"
                "1. Carefully analyze the provided title and context paragraphs.\n"
                "2. Determine if the user's statement is TRUE or FALSE based *only* on the information given in the context. "
                "**Do not use any external knowledge, data, or assumptions.**\n"
                "3. If the context is insufficient to definitively determine TRUE or FALSE, explain why.\n"
                "4. Respond with **exactly** one JSON object, with no markdown fences or other text outside the JSON structure. "
                "The JSON object must have a single key: \"answer\".\n"
                "5. The value for \"answer\" must start with \"TRUE.\" or \"FALSE.\" (or explain insufficiency), "
                "followed by a brief explanation citing evidence *from the provided context* to support your conclusion.\n"
                "6. **Do NOT wrap the JSON in triple back-ticks, and do NOT add any text before or after it.**\n\n"
                "Example of expected JSON output format:\n"
                "{\n"
                "  \"answer\": \"TRUE. <Your brief explanation based *only* on the provided context>\"\n"
                "}\n"
            ),
            "title": title_for_tf,
            "context_paragraphs": context_paragraphs,
            "user_statement_question": question_text
        }
        prompt = (
            f"{prompt_structure['task_description']}\n"
            f"Title: {prompt_structure['title']}\n\n"
            f"Context Paragraphs:\n{prompt_structure['context_paragraphs']}\n\n"
            f"User's Statement (Question):\n{prompt_structure['user_statement_question']}"
        )

    elif source_type == "Fill-in-the Blank":
        json_output_example = '{\n  \"answers\": [\"content for blank 1\", \"content for blank 2\", ... ]\n}'
        if question_text.count("[BLANK]") == 1 or question_text.count("{}") == 1:
             json_output_example = '{\n  \"answers\": [\"content for the blank\"]\n}'

        prompt = (
            "You are an expert AI assistant. The user will provide a text with one or more [BLANK] parts (or placeholders like {}).\n\n"
            "Your task is to:\n"
            "1. Carefully read the text.\n"
            "2. Determine the most appropriate content to fill in each [BLANK] or placeholder.\n"
            f"3. {json_instruction_main}{json_output_example}\n\n"
            f"Text with [BLANK](s):\n{question_text}\n\n"
            "Your response (JSON object):"
        )

    elif source_type == "General QA":
        json_output_example = '{\n  \"answer\": \"Your comprehensive and accurate answer here.\"\n}'
        prompt = "You are an expert AI assistant. The user will provide a question, possibly with some background context.\n\n"
        prompt += "Your task is to:\n"
        prompt += "1. Carefully analyze the question and any provided context.\n"
        prompt += "2. Provide a comprehensive and accurate answer to the question.\n"
        prompt += f"3. {json_instruction_main}{json_output_example}\n\n"
        if question_context:
            prompt += f"Background Context:\n{question_context}\n\n"
        prompt += f"Question:\n{question_text}\n\n"
        prompt += "Your response (JSON object):"

    elif source_type == "Multiple Choice":
        json_output_example = '{\n  \"answer\": [\"A\", \"C\"]\n}' 
        prompt = "You are an expert AI assistant. The user will provide a multiple-choice question and a list of options.\n\n"
        prompt += "Your task is to:\n"
        prompt += "1. Carefully analyze the question and the options.\n"
        prompt += "2. Identify ALL options that correctly answer the question. If no option is correct, the value for \"answer\" should be an empty list: [].\n"
        prompt += f"3. {json_instruction_main}{json_output_example}\n\n"
        if background_text:
            prompt += f"Background Information:\n{background_text}\n\n"
        prompt += f"Question:\n{question_text}\n\n"
        prompt += f"Options:\n{format_options(options)}\n\n"
        prompt += "Your response (JSON object with a list of correct option letters):"

    elif source_type == "Single Choice":
        json_output_example = '{\n  \"answer\": \"B\"\n}'
        prompt = "You are an expert AI assistant. The user will provide a multiple-choice question and a list of options.\n\n"
        prompt += "Your task is to:\n"
        prompt += "1. Carefully analyze the question and the options.\n"
        prompt += "2. Identify the SINGLE BEST option that correctly answers the question.\n"
        prompt += f"3. {json_instruction_main}{json_output_example}\n\n"
        if question_context: 
            prompt += f"Background Information:\n{question_context}\n\n"
        prompt += f"Question:\n{question_text}\n\n"
        prompt += f"Options:\n{format_options(options)}\n\n"
        prompt += "Your response (JSON object with the single correct option letter):"

    elif source_type == "Numerical QA":
        json_output_example = '{\n  \"answer\": \"The final numerical answer as a string (e.g., \'123.45\', or \'x = 5\')\"\n}'
        prompt = "You are an expert AI assistant in quantitative problem solving. The user will provide a numerical problem, possibly with context or formulas.\n\n"
        prompt += "Your task is to:\n"
        prompt += "1. Carefully analyze the problem statement, any provided context, and formulas.\n"
        prompt += "2. Solve the problem. The primary output must be the JSON with the final answer.\n"
        prompt += f"3. {json_instruction_main}"
        prompt += f"{json_output_example}\n"
        prompt += "   The value of \"answer\" should be the final numerical result as a string.\n\n"
        if question_context:
            prompt += f"Problem Context:\n{question_context}\n\n"
        if formula_context:
            prompt += f"Relevant Formulas:\n{formula_context}\n\n"
        prompt += f"Problem:\n{question_text}\n\n"
        prompt += "Your response (JSON object with the numerical answer):"

    elif source_type == "Proof":
        json_output_example = '{\n  \"answer\": \"Your rigorous, step-by-step proof or explanation here.\"\n}'
        prompt = "You are an expert AI assistant in mathematical and logical reasoning. The user will provide a statement or problem requiring a proof.\n\n"
        prompt += "Your task is to:\n"
        prompt += "1. Carefully analyze the statement/problem.\n"
        prompt += "2. Provide a rigorous, step-by-step proof or the completed proof.\n"
        prompt += f"3. {json_instruction_main}{json_output_example}\n\n"
        if question_context:
            prompt += f"Context or Given Information:\n{question_context}\n\n"
        prompt += f"Statement/Problem to Prove:\n{question_text}\n\n"
        prompt += "Your response (JSON object with the proof):"

    elif source_type == "Table QA":
        json_output_example = '{\n  \"answer\": \"Your answer derived from the table. You can include a brief explanation within this string if necessary for clarity.\"\n}'
        prompt = "You are an expert AI assistant skilled in data interpretation from tables. The user will provide a question, one or more tables, and possibly related formulas or context.\n\n"
        prompt += "Your task is to:\n"
        prompt += "1. Carefully analyze the question, the provided table(s), and any additional context or formulas.\n"
        prompt += "2. Extract and synthesize information from the table(s) to answer the question.\n"
        prompt += f"3. {json_instruction_main}{json_output_example}\n"
        prompt += "   The \"answer\" field should contain your answer derived from the table.\n\n"
        if question_context:
            prompt += f"Additional Context:\n{question_context}\n\n"
        if formula_context:
            prompt += f"Relevant Formulas:\n{formula_context}\n\n"
        prompt += f"Table(s):\n{format_table_html(table_html)}\n\n"
        prompt += f"Question:\n{question_text}\n\n"
        prompt += "Your response (JSON object with the answer and explanation):"
        
    else: # Fallback for unknown types
        prompt = (
            f"Received an unprocessed or unknown question type: '{source_type}'.\n"
            f"Please define a specific prompt for this type or check the data.\n\n"
            f"Raw Question Data:\n{json.dumps(question_data, indent=2, ensure_ascii=False)}"
        )
    
    return prompt

if __name__ == "__main__":
    data_dir_from_user = r"E:\shuju\NIPS_FINAL\FIANL_Benchmark_31559"
    
    print(f"--- Prompt Generation Test ---")
    print(f"Attempting to load questions from data directory: {data_dir_from_user}")
    all_questions, _ = loader.load_all_questions(data_dir_from_user)

    if not all_questions:
        print("\nNo questions loaded. Cannot test prompt generation.")
    else:
        print(f"\nSuccessfully loaded {len(all_questions)} questions for testing prompt generation.")
        
        types_to_test = [
            "TF",
            "Fill-in-the Blank",
            "General QA",
            "Multiple Choice",
            "Single Choice",
            "Numerical QA",
            "Proof",
            "Table QA"
        ]
        
        questions_by_type_found = {}
        for q in all_questions:
            q_type = q.get("question_source_type")
            if q_type in types_to_test and q_type not in questions_by_type_found:
                 questions_by_type_found[q_type] = q
        
        if not questions_by_type_found:
            print("\nCould not find representative questions for all specified types to test prompt generation.")
            print("Showing prompts for the first 5 loaded questions instead (if any):")
            for i, question_data in enumerate(all_questions[:5]):
                print(f"\n--- Prompt for Question {i+1} (QID: {question_data.get('qid', 'N/A')}, Type: {question_data.get('question_source_type', 'N/A')}) ---")
                generated_prompt = format_prompt(question_data)
                print(generated_prompt)
        else:
            print("\n--- Generated Prompts for Sample Questions by Type ---")
            for q_type in types_to_test:
                if q_type in questions_by_type_found:
                    question_data = questions_by_type_found[q_type]
                    print(f"\n--- Prompt for a '{q_type}' question (QID: {question_data.get('qid', 'N/A')}) ---")
                    generated_prompt = format_prompt(question_data)
                    print(generated_prompt)
                else:
                    print(f"\n--- No sample question found for type: '{q_type}' in the loaded data. ---")

    print(f"\n--- End of Prompt Generation Test ---")
