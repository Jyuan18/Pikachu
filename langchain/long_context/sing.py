'''
s1:text->split->docs([document])

s2:single document 怎么处理? 摘要

s3:多个 摘要如何分批处理,顺序额如何拼接

s4:给vllm模型batch处理

s5:结果拼接
'''

# s1:split texts to docs[document]
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional, Tuple, Dict
from langchain.callbacks.manager import CallbackManagerForLLMRun

from vllm import LLM, SamplingParams


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


# s2:create a single handle chain
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


def load_model_tokenizer_vllm(device):
    model_path = '/data/llm_models/Qwen1.5-0.5B-Chat'
    sampling_params = SamplingParams(
        temperature=0.01,
        top_p=0.9
    )
    llm = LLM()
    


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






if __name__ == "__main__":
    # s1
    with open('./law_text.txt') as f:
        data = f.read()
        
    texts = split_text(data)
    
    # s2
    prompt_template =  """总结下面法律文本内容,要求保留原文中的关键信息.
    法律文本:{text}
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=['text'],
        template_format='f-string',
    )
    
    llm = ()
    
    # s3
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt
    )
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="text"
    )
    
    # 
    result = stuff_chain.invoke(texts)['output_text']
    print(result)
    