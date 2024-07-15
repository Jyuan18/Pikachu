from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='google-bert/bert-base-uncased',
    
)

sequence = "In a hole in the ground there lived a hobbit."
print(tokenizer(sequence))
# {'input_ids': [101, 1999, 1037, 4920, 1999, 1996, 2598, 2045, 2973, 1037, 7570, 10322, 4183, 1012, 102], 
#  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

'''
tokenizer返回一个包含三个重要对象的字典：

input_ids 是与句子中每个token对应的索引。
attention_mask 指示是否应该关注一个token。
token_type_ids 在存在多个序列时标识一个token属于哪个序列。
'''

batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_inputs = tokenizer(batch_sentences)
print(encoded_inputs)
# {'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102],
#                [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
#                [101, 1327, 1164, 5450, 23434, 136, 102]],
#  'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0]],
#  'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1],
#                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                     [1, 1, 1, 1, 1, 1, 1]]}

# 填充
# 句子的长度并不总是相同，这可能会成为一个问题，因为模型输入的张量需要具有统一的形状。填充是一种策略，通过在较短的句子中添加一个特殊的padding token，以确保张量是矩形的。

# 将 padding 参数设置为 True，以使批次中较短的序列填充到与最长序列相匹配的长度

# 截断
# 另一方面，有时候一个序列可能对模型来说太长了。在这种情况下，您需要将序列截断为更短的长度。

# 将 truncation 参数设置为 True，以将序列截断为模型接受的最大长度：

batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
print(encoded_input)

# {'input_ids': tensor([[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
#                       [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
#                       [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]]),
#  'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
#  'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}