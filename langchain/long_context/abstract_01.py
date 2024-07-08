from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain


from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun


# split-text
with open('string.txt') as f:
    data = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        "。",
        " ",
        "",
    ],
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False
)

texts = text_splitter.create_documents([data])
print(len(texts), type(texts))
# print(texts[0])
# print(texts[1])


# load-llm
device = 'cuda'
model_path = '/models/qwen1.5-7B-chat'

model = AutoModelForCausalLM.from_pretrained(
   pretrained_model_name_or_path=model_path,
   torch_dtype='auto',
   device_map='auto' 
)
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path
)

class Qwen(LLM, ABC):
    max_token: int = 8192
    temperature: float = 0.01
    top_p: float = 0.9
    history_len: int = 3
    
    def __init__(self):
        super().__init__()
    
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
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=self.max_token,
            top_p=self.top_p,
            temperature=self.temperature
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
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


llm = Qwen()

prompt_template = """请将下面法律文本进行总结归纳,要求尽可能保留关键信息.
法律文本:{text}.
"""
prompt = PromptTemplate.from_template(prompt_template)
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt
)

chain = load_summarize_chain(
    llm=llm,
    chain_type='map_reduce'
)

map_template = """
下面是一系列文本的合集,请总结出主要观点.
{docs}
"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

#
llm = Qwen()

prompt_template = """总结下面法律文本内容,要求保留原文中的关键信息.
法律文本:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['text'])
chain = load_summarize_chain(llm, chain_type='map_reduce', return_intermediate_steps=True, map_prompt=prompt, combine_prompt=prompt)

# chain = load_summarize_chain(llm, chain_type='map_reduce', return_intermediate_steps=True)

# chain.run(texts)
chain({'input_documents':texts}, return_only_outputs=True)

