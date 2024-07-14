# 知识文本切分
## 方法一:一般的文本分块方法,直接按长度
text = 'xxx'
chunks = []
chunk_size = 128
for i in range(0, len(text), chunk_size):
    chunk = text[i: i+chunk_size]
    chunks.append(chunk)
print(chunks)

## 方法二:正则拆分
### 方法一中按长度切分往往会在句子中间切分开,在中文场景下正则表达式可以用来识别中文标点符号,从而将文本拆分成单独的句子
### 这种方法依赖于中文句号\问号\感叹号等标点符号作为句子结束的标志
import re
def split_sentences(text):
    sentence_delimiters = re.compile(u'[。？！；]\n')
    sentences = sentence_delimiters.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

## 方法三:Spacy text splitter方法
### Spacy是一个执行自然语言处理的各种任务的库
import spacy
input_text = ''
nlp = spacy.load('zh_core_web_sm')
doc = nlp(input_text)
for s in doc.sents:
    print(s)

## 方法四:基于langchain的character TextSplitter方法
### chunk_size,chunk_overlop,separator, strip_whitespace

## 方法五:基于langchain的递归字符切分方法
### 拆分器首先查找两个换行符,一旦段落被分割,它就会查看块的大小,如果块太大,就被下一个分隔符分割

## 方法六:基于langchain的html切分
### HTMLHeaderTextSplitter

## 方法七:基于langchain的markdown切分
###

## 方法八:基于langchain的python代码切分

## 方法九:latex方法

## 方法十:基于bert nsp的文本语义拆分方法

## 方法十一:语义段的切分及段落关键信息抽取
# 成分句法分析 命名实体识别
# 语义角色标注
# 直接法,关键词提取,HanLP
# 垂直领域建议的方法,仿照ChatLaw的做法,即:训练一个生成关键词的模型.ChatLaw是训练了一个KeyLLM
