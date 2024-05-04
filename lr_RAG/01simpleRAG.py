"""
一.前言
二.环境搭建
三.simple_RAG代码介绍
四.simple_RAG运行

一.前言
    1.1 为什么开发该项目?
        使用Huggingface的transformers和Meta的faiss实现一个非常简单和教学的检索增强生成实现
        不需要OpenAI密钥
    1.2 requirement
        python 3.10 torch 2.2.0 transformers 4.38.2 faiss 1.7.2 argparse 1.1 peft 0.9.0 trl 0.7.11
        CUDA 12.2 deepspeed 0.13.1 bitsandbytes 0.41.3 flash-attn 2.5

二.环境搭建
    2.1 下载代码-不需要
    2.2 构建环境 conda create -n py310 python=3.10 source activate py310
    2.3 安装依赖
        新建requirements.txt文件,将依赖项写入
        faiss==1.7.2
        argparse==1.1
        transformers==4.37.2
        torch==2.0.0
        tqdm==4.66.2

        pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --ignore-installed
    2.4 embeddings下载
        git lfs install
        git clone https://huggingface.co/BAAI/bge-large-zh-v1.5

三.代码介绍
    3.1 导包
    3.2 文件处理函数定义
"""

# 3.1 导包
import os
import argparse
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForCausualLM
import torch
from tqdm import tqdm


# 3.2 文件处理函数定义
def process_file(file_path):
    """
    加载一个本地文件,这里以txt文件为例
    使用不同方法将文件进行分割
    分割方法:spacy\langchain
    """
    with open(file_path, encoding='utf-8') as f:
        text = f.read()
        # 分割方法细化
        sentences = text.split('\n')
        return text, sentences

# 3.3 构建prompt函数定义
def generate_rag_prompt(data_point):
    return f"""### Instruction:
        {data_point['instruction']}
        ### Input:
        {data_point['input']}
        ### Response:
    """

# 3.4 文档Embedder类定义
class DocumentEmbedder:
    def __init__(
        self,
        model_name='BAAI/bge-large-zh-v1.5',
        max_length=128,
        max_number_of_sentences=20
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.max_number_of_sentences = max_number_of_sentences
    
    def get_document_embeddings(self, sentences):
        sentences = sentences[:self.max_number_of_sentences]