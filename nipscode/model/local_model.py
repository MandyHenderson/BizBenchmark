from .base_model import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore
import torch
from typing import Optional, Any # Added Optional, Any

class LocalModel(BaseModel):
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        # Defer actual loading to a separate method or first inference call 
        # to avoid long load times on instantiation if not immediately used.
        # For now, keep original loading for simplicity as it's a placeholder.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            print(f"LocalModel '{self.model_name_or_path}' loaded successfully on {self.device}.")
        except Exception as e:
            print(f"Error loading LocalModel '{self.model_name_or_path}': {e}")
            print("LocalModel will not be functional. Please ensure the model name/path is correct and dependencies are installed.")
            self.tokenizer = None
            self.model = None

    def infer(self, user_prompt: str, **kwargs: Any) -> Optional[str]: # Updated signature
        if not self.tokenizer or not self.model:
            print(f"LocalModel '{self.model_name_or_path}' is not properly initialized. Cannot infer.")
            return None
        
        # kwargs are ignored for now but are part of the signature for interface compatibility
        # For example, expect_json_object, system_prompt are not used by this basic local model

        try:
            input_ids = self.tokenizer.encode(user_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                # Common generation parameters, can be made configurable
                output = self.model.generate(
                    input_ids,
                    max_length=kwargs.get("max_tokens", 512), # Use max_tokens from kwargs if provided
                    pad_token_id=self.tokenizer.eos_token_id # Suppress warning
                )
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # The prompt is often included in the output, so remove it for a cleaner response.
            if decoded_output.startswith(user_prompt):
                return decoded_output[len(user_prompt):].strip()
            return decoded_output.strip()
        except Exception as e:
            print(f"Error during LocalModel inference: {e}")
            return None

if __name__ == '__main__':
    print("--- model/local_model.py direct execution test ---")
    
    # Test with a known small model that is likely to be available or quick to download
    # Using a very small model for testing to avoid large downloads unless already cached.
    # For a real test, ensure the specified model (e.g., "gpt2") is appropriate.
    test_model_name = "sshleifer/tiny-gpt2" # A very small GPT-2 model for quick testing
    print(f"Attempting to instantiate LocalModel with: '{test_model_name}'")
    
    local_model_instance = None
    try:
        local_model_instance = LocalModel(model_name_or_path=test_model_name)
    except Exception as e:
        print(f"Failed to instantiate LocalModel during test: {e}")

    if local_model_instance and local_model_instance.model:
        print("LocalModel instantiated successfully.")
        test_prompt = "Hello, world! This is a test prompt for the local model."
        print(f"Test prompt: {test_prompt}")
        
        # Test infer method
        response = local_model_instance.infer(test_prompt, max_tokens=50) # Pass a kwarg
        
        if response is not None:
            print(f"Raw Model response: {response}")
        else:
            print("Failed to get a response from LocalModel.infer().")
    else:
        print(f"LocalModel '{test_model_name}' could not be loaded or initialized. Skipping inference test.")
        print("This might be due to network issues, an incorrect model name, or missing dependencies (transformers, torch, sentencepiece).")

    print("--- model/local_model.py test complete ---")
