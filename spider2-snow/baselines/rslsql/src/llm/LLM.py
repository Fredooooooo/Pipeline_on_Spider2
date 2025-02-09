import openai
import json
import torch
import os
import sys
import re
sys.path.append(".")
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from configs.config import api, base_url, model, model_path, cuda_visible
import json

def append_into_jsonl_file(file_path, data_to_append):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data_to_append, ensure_ascii=False) + '\n')

os.environ["HF_DATASETS_CACHE"] = model_path
os.environ["HF_HOME"] = model_path
os.environ["HF_HUB_CACHE"] = model_path
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible

class GPT:
    def __init__(self):
        self.client = openai.OpenAI(api_key=api, base_url=base_url)

    def __call__(self, instruction, prompt):
        num = 0
        flag = True
        while num < 3 and flag:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    stream=False,

                )
            except Exception as e:
                print(e)
                continue
            try:
                json.loads(response.choices[0].message.content)
                flag = False
            except:
                flag = True
                num += 1

        return response.choices[0].message.content


class QWQ:
    def __init__(self, device="cuda"):
        """
        Initializes the QWQ model for local inference.
        :param device: Device to run inference on ("cuda" for GPU, "cpu" for CPU).
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True, torch_dtype="auto"
        ).eval()

    def count_tokens(self, input_data):
        """
        Count the number of tokens in a string or JSON file using a tokenizer.

        Args:
            input_data (str or dict): A string or a JSON file path.

        Returns:
            int: Number of tokens.
        """
        # Handle input data
        if isinstance(input_data, str):
            # If input is a file path
            if input_data.endswith(".txt"):
                with open(input_data, "r", encoding="utf-8") as file:
                    text = file.read()
            elif input_data.endswith(".json"):
                with open(input_data, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    text = json.dumps(data)  # Convert JSON to string
            else:
                # Assume input is a plain string
                text = input_data
        else:
            raise ValueError("Input must be a string (text or file path).")

        # Tokenize the text and count tokens
        tokens = self.tokenizer.tokenize(text)
        return len(tokens)
    
    def extract_json(self, text):
        """
        Extracts and cleans JSON from the generated text using a stack-based approach
        to handle nested curly braces.
        :param text: Model's generated text.
        :return: Clean JSON string if found, else an empty JSON.
        """
        stack = []
        start_index = -1

        for i, char in enumerate(text):
            if char == '{':
                if not stack:
                    start_index = i  # Mark the start of the outermost JSON object
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack:
                        # Found the complete JSON object
                        json_str = text[start_index:i + 1]
                        json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
                        return json_str
        return "{}"

    def __call__(self, instruction, prompt):
        """
        Runs inference using the locally loaded QWQ model with strict JSON output.
        :param instruction: System instruction for the model.
        :param prompt: User input prompt.
        :return: Model response as a strictly formatted JSON string.
        """
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        num = 0
        flag = True
        response_text = ""
        
        while num < 3 and flag:
            try:
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **model_inputs, max_new_tokens=1024
                    )
                generated_ids = [
                    output_ids[len(input_ids):] 
                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                raw_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                # response_text = self.extract_json(raw_response)  # Extract JSON part
                # jsonl_data = {
                #     "raw_response": raw_response,
                #     "response_text": response_text
                # }
                # path = '/fred/Pipeline_on_Spider2/spider2-snow/baselines/rslsql/src/llm/result.jsonl'
                # append_into_jsonl_file(path, jsonl_data)
                json.loads(raw_response)  # Validate JSON
                flag = False
            except Exception as e:
                print(f"Error: {e}")
                num += 1
                flag = True
                
        return raw_response
