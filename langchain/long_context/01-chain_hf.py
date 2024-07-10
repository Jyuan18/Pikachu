from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
# from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
import concurrent.futures

from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional, Tuple, Dict
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.callbacks import Callbacks
from concurrent.futures import ThreadPoolExecutor

# split-text
def split_text(data):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            "。",
            " ",
            "",
        ],
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=True
    )

    texts = text_splitter.create_documents([data])
    return texts


# 加载本地qwen模型
def load_model_tokenizer(device):
    model_path = '/data/llm_models/Qwen1.5-0.5B-Chat'
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype='auto',
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path
    )
    return model, tokenizer


class Qwen(LLM, ABC):
    max_token: int = 8192
    temperature: float = 0.01
    top_p: float = 0.9
    history_len: int = 3
    
    # 基于本地 llama3 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
    
    @property
    def _llm_type(self) -> str:
        return "Qwen"
    
    @property
    def _history_len(self) -> int:
        return self.history_len
    
    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to('cuda')
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=self.max_token,
            top_p=self.top_p,
            temperature=self.temperature
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def get_num_tokens(self, text: str) -> int:
        tokenized_text = self.tokenizer(text, return_tensors="pt")
        return len(tokenized_text.input_ids[0])
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"max_token": self.max_token,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "history_len": self.history_len}


if __name__ == '__main__':
    with open('./law_text.txt') as f:
        data = f.read()
    
    texts = split_text(data)
    
    model, tokenizer = load_model_tokenizer('auto')
    llm = Qwen(model, tokenizer)

    prompt_template = """总结下面法律文本内容,要求保留原文中的关键信息.
    法律文本:{text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['text'])
    
    map_chain = load_summarize_chain(llm, chain_type='map_reduce', return_intermediate_steps=True, map_prompt=prompt, combine_prompt=prompt)
    
    reduce_chain = map_chain



# 处理文档数据集
    result = map_chain.invoke(texts)

    print(result)