import os
import argparse
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from tqdm import tqdm

# 文件处理函数--针对txt文件
def process_file(file_path):
    with open(file_path, encoding='utf-8') as f:
        text = f.read()
        # 简单的切分成句子
        sentences = text.split('\n')
        return text, sentences

# 构建prompt
def generate_rag_prompt(data_point):
    return f'''###Instruction:
    {data_point['instruction']}
    ### Input:
    {data_point['input']}
    ### Response:
'''

# 文档Embedder类
class DocumentEmbedder:
    def __init__(
        self,
        model_name='BAAI/bge-large-zh-v1.5',
        max_length=128,
        max_number_of_sentences=20
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # 每个句子最大长度
        self.max_length = max_length
        # 考虑的句子的最大数量
        self.max_number_of_sentences = max_number_of_sentences

    def get_document_embedding(self, sentences):
        sentences = sentences[:self.max_number_of_sentences]
        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        return torch.mean(model_output.pooler_output, dim=0, keepdim=True)

# 生成大模型类定义
class GenerativeModel:
    def __init__(
        self,
        model_path='Qwen1.5-0.5B',
        max_input_length=200,
        max_generated_length=200
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            device_map='auto',
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_sid='left',
            add_eos_token=True,
            add_bos_token=True,
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_length = max_input_length
        self.max_generated_length = max_generated_length
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def answer_prompt(self, prompt):
        encoded_input = self.tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors='pt'
        )
        outputs = self.model.generate(
            input_ids=encoded_input['input_ids'].to(self.device),
            attention_mask=encoded_input['attention_mask'].to(self.device),
            max_new_tokens=self.max_generated_length,
            do_sample=False
        )
        decoder_text = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        return decoder_text


if __name__ == '__main__':
    