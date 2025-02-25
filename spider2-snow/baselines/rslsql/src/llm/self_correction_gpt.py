import openai
import json
import torch
import os
import sys
sys.path.append(".")
from transformers import AutoModelForCausalLM, AutoTokenizer
from configs.config import api, base_url, model, model_path, cuda_visible

os.environ["HF_DATASETS_CACHE"] = model_path
os.environ["HF_HOME"] = model_path
os.environ["HF_HUB_CACHE"] = model_path
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible

class GPT:
    def __init__(self):
        self.client = openai.OpenAI(api_key=api, base_url=base_url)

    def __call__(self, message):
        num = 0
        flag = True
        while num < 3 and flag:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=message,
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
        message.append({'role': 'assistant', 'content': response.choices[0].message.content})

        return message,response.choices[0].message.content



class QWQ:
    def __init__(self, device="cuda"):
        """
        Initializes the QWQ model for local inference.
        :param model_path: Path to the locally stored model weights.
        :param device: Device to run inference on ("cuda" for GPU, "cpu" for CPU).
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True, torch_dtype="auto"
        ).eval()
    
    def __call__(self, message):
        """
        Runs inference using the locally loaded QWQ model.
        :param message: List of messages for chat-based interaction.
        :return: Updated message list and model response.
        """
        text = self.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        num = 0
        flag = True
        response_text = ""
        
        while num < 3 and flag:
            try:
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **model_inputs, max_new_tokens=512
                    )
                generated_ids = [
                    output_ids[len(input_ids):] 
                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                json.loads(response_text)  # Ensure response is valid JSON
                flag = False
            except Exception as e:
                print(f"Error: {e}")
                num += 1
                flag = True
                
        message.append({'role': 'assistant', 'content': response_text})
        return message, response_text
