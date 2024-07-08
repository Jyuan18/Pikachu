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


# # 自定义 MapReduceDocumentsChain 来使用多个模型在 Map 阶段
# class CustomMapReduceDocumentsChain(MapReduceDocumentsChain):
#     def run(self, texts):
#         map_results = []
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             future_to_chain = {
#                 executor.submit(chain.run, [text]): chain
#                 for text, chain in zip(texts, map_chains * (len(texts) // len(map_chains) + 1))
#             }
#             for future in concurrent.futures.as_completed(future_to_chain):
#                 map_results.append(future.result())
#         # 使用 reduce_chain 进行合并
#         return self.reduce_chain.run(map_results)

class CustomMapReduceDocumentsChain(MapReduceDocumentsChain):
    def __init__(self, *args, model_a: LLMChain, model_b: LLMChain, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_a = model_a
        self.model_b = model_b

    def combine_docs(
        self,
        docs: List[Document],
        token_max: Optional[int] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Tuple[str, dict]:
        """Combine documents in a map reduce manner using multiple models."""
        half = len(docs) // 2
        docs_a = docs[:half]
        docs_b = docs[half:]

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(self._apply_model, self.model_a, docs_a, kwargs, callbacks)
            future_b = executor.submit(self._apply_model, self.model_b, docs_b, kwargs, callbacks)
            
            map_results_a = future_a.result()
            map_results_b = future_b.result()

        map_results = map_results_a + map_results_b

        question_result_key = self.llm_chain.output_key
        result_docs = [
            Document(page_content=r[question_result_key], metadata=docs[i].metadata)
            for i, r in enumerate(map_results)
        ]

        result, extra_return_dict = self.reduce_documents_chain.combine_docs(
            result_docs, token_max=token_max, callbacks=callbacks, **kwargs
        )

        if self.return_intermediate_steps:
            intermediate_steps = [r[question_result_key] for r in map_results]
            extra_return_dict["intermediate_steps"] = intermediate_steps

        return result, extra_return_dict

    def _apply_model(self, model: LLMChain, docs: List[Document], kwargs: Dict[str, Any], callbacks: Callbacks):
        return model.apply(
            [{self.document_variable_name: d.page_content, **kwargs} for d in docs],
            callbacks=callbacks,
        )


if __name__ == '__main__':
    with open('./law_text.txt') as f:
        data = f.read()
    
    texts = split_text(data)
    
    model1, tokenizer1 = load_model_tokenizer('cuda:0')
    model2, tokenizer2 = load_model_tokenizer('cuda:1')
    model3, tokenizer3 = load_model_tokenizer('cuda:2')
    model4, tokenizer4 = load_model_tokenizer('cuda:3')
    
    llm1 = Qwen(model1, tokenizer1)
    llm2 = Qwen(model2, tokenizer2)
    llm3 = Qwen(model3, tokenizer3)
    llm4 = Qwen(model4, tokenizer4)

    prompt_template = """总结下面法律文本内容,要求保留原文中的关键信息.
    法律文本:{text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['text'])
    
    chain1 = load_summarize_chain(llm1, chain_type='map_reduce', return_intermediate_steps=True, map_prompt=prompt, combine_prompt=prompt)
    chain2 = load_summarize_chain(llm2, chain_type='map_reduce', return_intermediate_steps=True, map_prompt=prompt, combine_prompt=prompt)
    chain3 = load_summarize_chain(llm3, chain_type='map_reduce', return_intermediate_steps=True, map_prompt=prompt, combine_prompt=prompt)
    chain4 = load_summarize_chain(llm4, chain_type='map_reduce', return_intermediate_steps=True, map_prompt=prompt, combine_prompt=prompt)
    
    map_chains = [chain1, chain2, chain3, chain4]
    reduce_chain = chain1
    
    
    
    
    
    # combine_document_chain = CustomMapReduceDocumentsChain(
    #     llm_chain=map_chains[0],
    #     reduce_documents_chain=reduce_chain
    # )
    
    # result = combine_document_chain.invoke({'input_documents': texts}, return_only_outputs=True)
    # print(result)
    
    
    
    


    # combine_documents_chain = StuffDocumentsChain(
    #     llm_chain=reduce_chain,
    #     document_prompt=prompt,
    #     document_variable_name=document_variable_name
    # )
    # reduce_documents_chain = ReduceDocumentsChain(
    #     combine_documents_chain=combine_documents_chain,
    # )

    chain = CustomMapReduceDocumentsChain(
        llm_chain=chain3,
        reduce_documents_chain=reduce_chain,
        model_a=chain1,
        model_b=chain2
    )


# 处理文档数据集
    result = chain.run(texts)

    print(result)