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

sys_message = '''{table_info}

### Select the best SQL query to answer the  question:
{candidate_sql}

Your answer should be returned by json format.
{
    "sql": "...",# your SQL query
}
'''


class GPT:
    def __init__(self):
        self.client = openai.OpenAI(api_key=api, base_url=base_url)

    def __call__(self, table_info, candidate_sql):
        prompt = sys_message.replace("{table_info}", table_info).replace("{candidate_sql}", candidate_sql)
        num = 0
        flag = True
        while num < 3 and flag:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
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
        :param model_path: Path to the locally stored model weights.
        :param device: Device to run inference on ("cuda" for GPU, "cpu" for CPU).
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        ).to(self.device)
    
    def __call__(self, table_info, candidate_sql):
        """
        Runs inference using the locally loaded QWQ model to select the best SQL query.
        :param table_info: Information about the database table.
        :param candidate_sql: Candidate SQL queries.
        :return: The best SQL query in JSON format.
        """
        prompt = sys_message.replace("{table_info}", table_info).replace("{candidate_sql}", candidate_sql)
        
        messages = [{"role": "user", "content": prompt}]
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
                
        return response_text

