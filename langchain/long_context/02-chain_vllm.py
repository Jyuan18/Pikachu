from langchain.text_splitter import RecursiveCharacterTextSplitter
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from openai import OpenAI


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
    docs = []
    for i in texts:
        docs.append(i.page_content)
    return docs


# vllm加载本地模型,方法一:脚本; 方法二:命令行
# 方法一:
def load_model(model_path):
    
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
    )

    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, repetition_penalty=1.05, max_tokens=4096)

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
        dtype='auto'
    )
    return tokenizer, llm, sampling_params

"""
# 方法二:
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-7B-Instruct \
    --tensor-parallel-size 4

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2-7B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me something about large language models."},
    ]
)
print("Chat response:", chat_response)
"""

def deal_prompt(docs):
    sys_prompt = """总结下面法律文本内容,要求保留原文中的关键信息.
    法律文本:{text}
    """
    prompt_list = []
    for text in docs:
        prompt_list.append(sys_prompt.replace('{text}', text))
    
    return prompt_list


def generate_result(tokenizer, llm, sampling_params, prompt_list):
    texts = []
    for prompt in prompt_list:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(text)

    # generate outputs
    # outputs = llm.generate([text], sampling_params)
    outputs = llm.generate(texts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")