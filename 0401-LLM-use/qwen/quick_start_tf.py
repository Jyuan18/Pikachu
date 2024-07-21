from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

if torch.cuda.is_available():
    device = 'cuda'
    num_gpus = torch.cuda.device_count()
    print(f'Number of GPUs available: {num_gpus}')
else:
    print('No GPUs available')
    num_gpus = 0
    device = 'cpu'

def print_memory_usage():
    for gpu_id in range(num_gpus):
        allocated_memory = torch.cuda.memory_allocated(gpu_id)
        reserved_memory = torch.cuda.memory_reserved(gpu_id)
        print(f"GPU {gpu_id} - Allocated memory: {allocated_memory / (1024 ** 2)} MB")
        print(f"GPU {gpu_id} - Reserved memory: {reserved_memory / (1024 ** 2)} MB")

print('before loading model')
print_memory_usage()

device = 'cuda'

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path='',
    torch_dtype='auto',
    device_map='auto',
    attn_implementation="flash_attention2",
)

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='',
)

print('after loading model')
print_memory_usage()

prompt = '1+1='
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer(
    [text],
    return_tensors='pt'
).to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)