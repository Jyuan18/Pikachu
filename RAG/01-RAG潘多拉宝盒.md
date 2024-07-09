# RAG潘多拉宝盒

## 一 为什么需要RAG
- 幻觉问题
- 时效性问题
- 数据安全问题

## 二 介绍一下RAG
> RAG-Retrieval Augmented Generation

当LLM面对解答问题或创作文本任务时,首先会在大规模文档库中搜索并筛选出与任务紧密相关的素材,继而依据这些素材精准指导后续的回答生成或文本构造过程,旨在通过这种方式提升模型输出的准确性和可靠性.

> 架构

本地知识文件->读取内容->文本切分->构成documents->得到每个documents的embedding->构建faiss索引
输入query->query的embedding->faiss搜索->得到top k个相关的document->将document的文本进行拼接,得到context
用context和query填充prompt模板,得到给llm的prompt->给到llm回答

## 三 RAG包含哪些模块
- 模块一:版面分析
  - 本地知识文件读取(pdf\txt\html\doc\excel\png\jpg\语音)
  - 知识文件复原
- 模块二:知识库构建
  - 知识文本切割,并构成doc文本
  - doc文本embedding
  - doc文本构成索引
- 模块三:大模型微调
- 模块四:基于RAG的知识问答
  - 用户query embedding
  - query召回
  - query排序
  - 将top k个相关的doc进行拼接,构建context
  - 基于query和context构建prompt
  - 将prompt喂给llm生成answer

## 四 RAG优点
无需训练即可为模型注入额外的信息资源,从而显著提升其回答的精确度
- 可扩展性
- 准确性
- 可控性
- 可解释性
- 多功能性
- 时效性
- 领域定制型
- 安全性

## 五 对比RAG和SFT
TODO怎么插入表格
TODO怎么插入图片链接

## 六 实战篇
### 6.1 simpleRAG

### 6.2 版面分析
> why

> how
### 6.2.1 本地知识文件获取
> 富文本txt


> PDF文档--pdfplumber

```
import pdfplumber
file_name = "**.pdf"
output_file = '**.txt'
with pdfplumber.open(file_name) as p:
    page_count = len(p.pages)
    for i in range(0, page_count):
        page = p.pages[i]
        textdata = page.extract_text()
        data = open(output_file, 'a')
        data.write(text_data)
```

## 七 项目篇
### 7.1 RAGFlow

### 7.2 QAnything

### 7.3 ElasticSearch-Langchain

### 7.4 Langchain-Chatchat
