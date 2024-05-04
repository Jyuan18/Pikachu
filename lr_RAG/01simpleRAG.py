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
        加载文本,分割文本
    3.3 构建prompt函数定义
        自定义问答模版
    3.4 文档Embedder类定义
    3.5 生成大模型类定义
    3.6 主函数定义

四.simple_RAG运行
    4.1 构建数据集
        rag_documents/
    4.2 运行py脚本
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
        # 分割方法待细化
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
        self.max_length = max_length  # maximum number of tokens per sentence
        # maximum number of sentences to be considered
        self.max_number_of_sentences = max_number_of_sentences

    def get_document_embeddings(self, sentences):
        # keep only the first k sentences for GPU purposes
        sentences = sentences[:self.max_number_of_sentences]
        # tokenize the sentences
        encoded_input = self.tokenizer(sentences,
                                       padding=True,
                                       truncation=True,
                                       max_length=128,
                                       return_tensors='pt')
        # compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        # document's embedding is the average of all sentences
        return torch.mean(model_output.pooler_output, dim=0, keepdim=True)


# 3.5 生成大模型类定义
class GenerativeModel:
    def __init__(
        self,
        model_path='Qwen/Qwen1.5-0.5B',
        max_input_length=200,
        max_generated_length=200
    ):
        self.model = AutoModelForCausualLM.from_pretrained(
            model_path,
            device_map='auto',
            torch_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side='left',
            add_eos_token=True,
            add_bos_token=True,
            use_fast=False,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_length = max_input_length
        self.max_generated_length = max_generated_length
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

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
            outputs, skip_special_tokens=True)
        return decoder_text


# 3.6 主函数定义
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--document_directory',
                        help='The directory that has the documents',
                        default='rag_documents')
    parser.add_argument('--embedding_model',
                        help='The Huggingface path to the embedding model to use',
                        default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--generative_model',
                        help='The Huggingface path to the generative model to use',
                        default='Writer/camel-5b-hf')
    parser.add_argument('--nubmer_of_docs',
                        help='The number of relevant documents to use for context',
                        default=2)
    args = parser.parse_args()
    
    # process all files by chunking them into sentences
    # keep track of the original documents filepath
    print('Splitting documents into sentences...')
    documents = {}
    for idx, file in enumerate(tqdm(os.listdir(args.documents_directory)[:10])):
        current_filepath = os.path.join(args.documents_directory, file)
        text, sentences = process_file(current_filepath)
        documents[idx] = {'file_path': file,
                          'sentences': sentences,
                          'document_text': text}
    
    # now for all sentences  get embeddings
    print('Getting document embeddings...')
    document_embedder = DocumentEmbedder(model_name=args.embedding_model,
                                         max_length=128,
                                         max_number_of_sentences=20)
    embeddings = []
    for idx in tqdm(documents):
        embeddings.append(document_embedder.get_document_embeddings(
            documents[idx]['sentences']))
    # concatenate all embeddings
    embeddings = torch.concat(embeddings, dim=0).data.cpu().numpy()
    embedding_dimensions = embeddings.shape[1]

    # use faiss to build an index we can use to search through the embeddings
    faiss_index = faiss.IndexFlatIP(int(embedding_dimensions))
    faiss_index.add(embeddings)

    question = 'Who is you'
    query_embedding = document_embedder.get_document_embeddings([question])
    distances, indices = faiss_index.search(query_embedding.data.cpu().numpy(),
                                            k=int(args.numeber_of_docs))
    
    # use the k-closest documents to provide context to the generative model's answer
    context = ''
    for idx in indices[0]:
        context += documents[idx]['document_text']
    rag_prompt = generate_rag_prompt({'instruction': question,
                                      'input': context})
    
    # use the generative model to give an answer to the question
    # use the retrieved documents for context
    print('Generating answer...')
    generative_model = GenerativeModel(model_path=args.generative_model,
                                       max_input_length=200,
                                       max_generated_length=200)
    answer = generative_model.answer_prompt(
        rag_prompt)[0].split('### Response:')[1]
    print(answer)
