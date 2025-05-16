from .base_model import BaseModel
# import openai # Legacy import, replaced by the line below
from openai import AsyncOpenAI # Use Async client
import json # Though not directly used in this file after refactor, good for context
import os # For API key from environment in potential test block
from typing import Optional # Add this import
import asyncio # For the test block

class APIModel(BaseModel):
    def __init__(self, api_key: str, model_name: str = "deepseek-chat", base_url: str = "https://api.deepseek.com"):
        super().__init__() # Assuming BaseModel has an __init__
        
        if not api_key:
            print("Warning: APIModel initialized with no API key. API calls will fail.")
            # Consider raising an error: raise ValueError("API key is required for APIModel")
        
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        
        try:
            # Instantiate AsyncOpenAI client
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except Exception as e:
            print(f"Error initializing AsyncOpenAI client in APIModel: {e}")
            self.client = None # Ensure client is None if initialization fails

    async def infer(self,  # Make infer an async method
              user_prompt: str, 
              system_prompt: str = "You are a helpful AI assistant. Please follow the user instructions precisely.", 
              expect_json_object: bool = False, 
              temperature: float = 0.2, 
              max_tokens: int = 4096) -> Optional[str]:
        """
        Sends the prompt(s) to the configured chat API and returns the model's content string.

        Args:
            user_prompt: The main user prompt/query.
            system_prompt: The content for the system message.
            expect_json_object: If True, sets response_format to {'type': 'json_object'}.
            temperature: The sampling temperature.
            max_tokens: The maximum number of tokens to generate.

        Returns:
            The content string from the model's response, or None if an error occurs.
        """
        if not self.client:
            print("Error: APIModel client is not initialized.")
            return None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if expect_json_object:
                api_params["response_format"] = {"type": "json_object"}
            
            # print(f"DEBUG: APIModel.infer sending params: {json.dumps(api_params, indent=2)[:500]}...") # Debug snippet
            response = await self.client.chat.completions.create(**api_params) # await the call
            
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                content = response.choices[0].message.content.strip()
                # print(f"DEBUG: APIModel.infer received content: {content[:200]}...") # Debug snippet
                return content
            else:
                print("Error: APIModel - API response did not contain expected choices or message content.")
                return None
        except Exception as e:
            print(f"Error during API call in APIModel.infer: {e}")
            # import traceback
            # print(traceback.format_exc()) # For more detailed error logging
            return None

async def main_test(): # Wrapper for async tests
    print("--- model/api_model.py direct execution test (async) ---")
    api_key_env = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key_env:
        print("DEEPSEEK_API_KEY environment variable not set. Skipping APIModel live test.")
    else:
        print("Attempting to instantiate APIModel with key from environment.")
        try:
            test_model = APIModel(api_key=api_key_env)
            if test_model.client:
                print("APIModel instantiated successfully.")
                # Test 1: Non-JSON
                print("\n--- Test 1: Non-JSON query ---")
                non_json_response = await test_model.infer("Tell me a short joke.", expect_json_object=False)
                if non_json_response:
                    print(f"Response: {non_json_response}")
                else:
                    print("Failed to get non-JSON response.")
                
                # Test 2: JSON (using a simplified TF-style prompt)
                print("\n--- Test 2: JSON query ---")
                tf_style_user_prompt = (
                    "Title: Test Doc\n"
                    "Context Paragraphs: The sky is blue.\n"
                    "User's Statement (Question): The sky is green."
                )
                tf_style_system_prompt = (
                    "You are an AI assistant. Determine if the user's statement is TRUE or FALSE based on context. "
                    "Respond with JSON: {\"answer\": \"TRUE/FALSE. <explanation>\"}."
                )
                json_response = await test_model.infer(user_prompt=tf_style_user_prompt, system_prompt=tf_style_system_prompt, expect_json_object=True)
                if json_response:
                    print(f"JSON Response String: {json_response}")
                    try:
                        parsed = json.loads(json_response)
                        print(f"Parsed JSON: {json.dumps(parsed, indent=2)}")
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON response: {e}")
                else:
                    print("Failed to get JSON response.")
            else:
                print("APIModel client initialization failed during test.")
        except Exception as e:
            print(f"Error during APIModel test: {e}")
    print("--- model/api_model.py test complete ---")

if __name__ == '__main__':
    asyncio.run(main_test()) # Run the async test function
