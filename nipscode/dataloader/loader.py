import json
import os

def load_all_questions(data_dir):
    questions = []
    file_question_counts = {} # To store counts per file

    if not os.path.isdir(data_dir):
        print(f"Error: Data directory '{data_dir}' not found or is not a directory. Cannot load questions.")
        return questions, file_question_counts

    for subject_folder_name in os.listdir(data_dir):
        subject_dir_path = os.path.join(data_dir, subject_folder_name)
        
        if not os.path.isdir(subject_dir_path):
            # print(f"Debug: Skipping non-directory entry in data_dir: {subject_folder_name}")
            continue

        # print(f"Debug: Processing subject folder: {subject_folder_name}")
        for json_filename in os.listdir(subject_dir_path):
            if not json_filename.endswith('.json'):
                # print(f"Debug: Skipping non-JSON file: {json_filename} in {subject_folder_name}")
                continue

            file_path = os.path.join(subject_dir_path, json_filename)
            relative_file_path = os.path.join(subject_folder_name, json_filename) # Path relative to data_dir
            questions_in_current_file = 0
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        print(f"Warning: JSON file is empty, skipping: {file_path}")
                        file_question_counts[relative_file_path] = 0
                        continue
                    data_from_file = json.loads(content)

                if not isinstance(data_from_file, list):
                    print(f"Warning: Top-level structure in {file_path} is not a list. Skipping this file.")
                    file_question_counts[relative_file_path] = 0
                    continue

                current_file_temp_questions = []
                for item_from_list in data_from_file:
                    if not isinstance(item_from_list, dict):
                        print(f"Warning: Found a non-dictionary item in top-level list of {file_path}. Skipping this item: {type(item_from_list)}")
                        continue

                    # Assign the question source type based on the subfolder name
                    question_source_type_tag = subject_folder_name

                    # Check for the nested structure (e.g., "statistic_proof.json" format)
                    if 'items' in item_from_list and isinstance(item_from_list['items'], list):
                        # This is a 'folder' entry with an 'items' list
                        # print(f"Debug: Found 'items' key in an entry from {file_path}. Processing nested questions.")
                        for actual_question_item in item_from_list['items']:
                            if isinstance(actual_question_item, dict):
                                actual_question_item['question_source_type'] = question_source_type_tag
                                actual_question_item['original_file_relative_path'] = relative_file_path
                                current_file_temp_questions.append(actual_question_item)
                            else:
                                print(f"Warning: Found a non-dictionary element within 'items' list in {file_path} (under item {item_from_list.get('folder', 'N/A')}). Skipping this sub-item.")
                    else:
                        # This is assumed to be a direct question item
                        item_from_list['question_source_type'] = question_source_type_tag
                        item_from_list['original_file_relative_path'] = relative_file_path
                        current_file_temp_questions.append(item_from_list)
                
                questions.extend(current_file_temp_questions)
                questions_in_current_file = len(current_file_temp_questions)
                file_question_counts[relative_file_path] = questions_in_current_file
            
            except json.JSONDecodeError as jde:
                print(f"Warning: JSON decoding error in {file_path}. Details: {jde}. Skipping this file.")
                file_question_counts[relative_file_path] = 0 # Record as 0 if file fails to parse
            except Exception as e:
                print(f"Error processing file {file_path}: {type(e).__name__} - {e}. Skipping this file.")
                file_question_counts[relative_file_path] = 0 # Record as 0 for other errors
    
    return questions, file_question_counts

if __name__ == "__main__":
    # 使用原始字符串处理 Windows 路径
    data_dir = r""

    print(f"--- Dataloader Test ---")
    print(f"Attempting to load questions from data directory: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"CRITICAL ERROR: The specified data directory does NOT exist: ")
        print(f"  '{data_dir}'")
        print(f"Please ensure the path is correct and the directory is accessible.")
        print(f"Test cannot proceed without a valid data directory.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    elif not os.path.isdir(data_dir):
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"CRITICAL ERROR: The specified path is NOT a directory: ")
        print(f"  '{data_dir}'")
        print(f"Test cannot proceed.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        all_questions, file_counts = load_all_questions(data_dir)
        
        print(f"\n--- Load_all_questions Function Execution Summary ---")
        print(f"Total questions loaded: {len(all_questions)}")

        if all_questions or file_counts: # Proceed if there are questions or at least file stats
            print(f"\n--- Breakdown by Subfolder (Question Source Type) ---")
            folder_counts = {}
            for q in all_questions:
                q_type = q.get('question_source_type', 'Unknown Type')
                folder_counts[q_type] = folder_counts.get(q_type, 0) + 1
            
            if folder_counts:
                for q_type, count in sorted(folder_counts.items()): # Sort for consistent order
                    print(f"  - Subfolder '{q_type}': {count} questions")
            else:
                print("  No questions were successfully processed to categorize by subfolder.")

            print(f"\n--- Breakdown by Individual File ---")
            if file_counts:
                for file_path_rel, count in sorted(file_counts.items()): # Sort for consistent order
                    # Normalize path separators for display consistency if needed, though os.path.join should be fine
                    print(f"  - File '{file_path_rel.replace(os.sep, '/')}': {count} questions")
            else:
                print("  No files were processed or no counts were recorded per file.")

        else:
            print(f"\nNo questions were loaded and no file statistics were generated. Potential reasons:")
            print(f"  1. The directory '{data_dir}' was not found or is not a directory.")
            print(f"  2. The directory exists but is empty or contains no scannable subdirectories/JSON files.")
            print(f"  3. All JSON files encountered issues (e.g., empty, malformed, wrong structure). Check warnings above.")
            
    print(f"\n--- End of Dataloader Test ---")

# Ensure to check the console for any 'Warning:' messages from the load_all_questions function during execution.
# These warnings provide details about files or items that might have been skipped.
