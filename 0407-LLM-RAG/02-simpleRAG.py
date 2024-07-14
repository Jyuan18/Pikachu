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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--documents_directory',
        help="The directory that has the documents",
        default='rag_documents'
    )
    parser.add_argument(
        '--embedding_model',
        help='The HuggingFace path to the embedding model to use',
        default='Writer/camel-5b-hf'
    )
    parser.add_argument(
        '--number_of_docs',
        help='The number of relevant documents to use for context',
        default=2
    )
    args = parser.parse_args()
    
    # process all files in the directory by chunking them into sentences
    # keep track of the original documents filepath
    print('splitting documents into sentences...')
    documents = {}
    for idx, file in enumerate(tqdm(os.listdir(args.documents_directory)[:10])):
        current_filepath = os.path.join(args.documents_directory, file)
        text, sentences = process_file(current_filepath)
        documents[idx] = {
            'file_path': file,
            'sentences': sentences,
            'document_text': text
        }
    
    # Now for all sentences get embeddings
    print('getting document embeddings...')
    document_embedder = DocumentEmbedder(
        model_name=args.embedding_model,
        max_length=128,
        max_number_of_sentences=20
    )
    embeddings = []
    for idx in tqdm(documents):
        # Embed the document
        embeddings.append(document_embedder.get_document_embeddings(documents[idx]['sentences']))
    # concatenate all embeddings
    embeddings = torch.concat(embeddings, dim=0).data.cpu().numpy()
    embedding_dimensions = embeddings.shape[1]
    
    # use faiss to build an index we can use to search through the embeddings
    # Ideally you'll want to cache this index so you don't have to build it every time
    faiss_index = faiss.IndexFlatIP(int(embedding_dimensions))
    faiss_index.add(embeddings)
    
    question = 'the content of query'
    query_embedding = document_embedder.get_document_embedding([question])
    distances, indices = faiss_index.search(query_embedding.data.cpu().numpy(),
                                            k=int(args.number_of_docs))
    # use the k-closest documents to provide context to the generative model's answer
    context = ''
    for idx in indices[0]:
        context += documents[idx]['document_text']
    
    rag_prompt = generate_rag_prompt(
        {'instruction': question,
         'input': context}
    )
    # use the generative model to give an answer to the question
    # use the retrieved documents for context
    print('generating answer...')
    generative_model = GenerativeModel(
        model_path=args.generative_model,
        max_input_length=200,
        max_generated_length=200
    )
    answer = generative_model.answer_prompt(rag_prompt)[0].split('### Response:')[1]
    print(answer)