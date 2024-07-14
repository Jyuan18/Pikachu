# [[RAG](https://articles.zsxq.com/id_389scfi6p7ms.html)]潘多拉宝盒

## 一 为什么需要RAG
- 幻觉问题:基于统计的概率方法逐词生成文本,这一机制内在地导致其可能出现看似逻辑严谨实则缺乏事实依据的输出;
- 时效性问题:包含最新信息的数据难以及时融入模型训练
- 数据安全问题:企业数据的业务计算需要放在本地完成

## 二 介绍一下RAG
> RAG-Retrieval Augmented Generation

当LLM面对解答问题或创作文本任务时,首先会在大规模文档库中搜索并筛选出与任务紧密相关的素材,继而依据这些素材精准指导后续的回答生成或文本构造过程,旨在通过这种方式提升模型输出的准确性和可靠性.

> 流程-架构

- 本地知识文件->读取内容->文本切分->构成documents->得到每个documents的embedding->构建faiss索引
- 输入query->query的embedding->faiss搜索->得到top k个相关的document->将document的文本进行拼接,得到context
- 用context和query填充prompt模板,得到给llm的prompt->给到llm回答


## 三 重点:RAG包含哪些模块-架构解析
- 模块一:版面分析及文本切分
  - 本地知识文件读取(pdf\txt\html\doc\excel\png\jpg\语音)
  - 知识文件复原
  - 文本切分的方式
    - 如何最大化保留有用的语义信息
- 模块二:知识库构建和索引构建
  - 知识文本切割,并构成doc文本
  - doc文本embedding
  - doc文本构成索引
- 模块三:大模型微调
  - 全量微调FFT
  - 参数高效微调PEFT
    - SFT
    - RLHF
    - RLAIF
- 模块四:基于RAG的知识问答
  - 用户query embedding
  - query召回
    - 如何高效召回
  - query排序
    - rerank
  - 将top k个相关的doc进行拼接,构建context
  - 基于query和context构建prompt并喂给llm生成answer

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

## 六 模块解析
### 6.0 simpleRAG-流程
见02

### 6.1 模块一:版面分析及文本切分
> why

RAG在具体应用时如文档解析\智能化协作及对话系统构建中,特别是结构化或半结构化信息的处理需求时,需要进行版面分析,针对特定的布局结构进行提取信息

> how:针对不同类型的文件,要采取特定的访问与解析策略来有效获取其中蕴含的知识

- 富文本txt
  - python-read(), readline(), readlines()
  - read()一次性读取所有文本
  - readline()读取第一行内容
  - readlines()读取全部内容,以列表的格式返回,一行为一个元素
- pdf
  - pdfplumber
  - PyMuPDF
- HTML
  - BeautifulSoup
- docx
  - python-docx不支持doc文档
  - doc和docx存在本质差异,一个是二进制,另一个是xml格式文件
- OCR
  - tesseract
  - paddleocr
  - hn ocr
- ASR-Automatic Speech Recognition
  - 将一段语音信号转换成相应的文本信息,让机器通过识别和理解,把语音信号转变为相应的文本或命令
  - STT
  - WeTextProcessing
  - Wenet

> 知识文件复原
- 法一:基于规则的知识文件复原
  - 根据识别段落的左右边距和末尾的标点符号进行合并
  - 如何判断该段落是否为原始段落末尾
    - 如果最后一个字符离右边界举例较远,那么为段落末尾
    - 如果最后一个字符为终止符,那么为段落末尾
- 法二:基于Bert NSP进行上下句拼接
  - 利用bert等模型来判断两个句子是否具有语义衔接关系,设置相似度阈值t,从前往后依次判断两个相邻段落的相似度分数是否大于t,如果大于则合并,否则断开
```python
def is_nextsent(sent, next_sent):
  encoding = tokenizer(sent, next_sent, return_tensors='pt', truncation=True, padding=False)
  with torch.no_grad():
    outputs = model(**encoding, labels=torch.LongTensor([1]))
    logits = outputs.logits
    probs = torch.softmax(logtis/TEMPERATURE, dim=1)
    next_sentence_prob = probs[:, 0].item()
  if next_sentence_prob <= MERGE_RATIO:
    return False
  else:
    return True
```
> 6.4 优化策略篇

> pdf文件解析问题

pdf文档是非结构化文档的代表,将pdf描述为输出指令的集合更准确,而不是数据格式
pdf文件由一系列指令组成,这些指令指示pdf阅读器显示符号的位置和方式,这与HTML等文件格式形成鲜明对比

- 解析pdf的三种方法
  - 方法一:基于规则的方法,根据文档的组织特征确定每个部分的风格和内容
  - 方法二:基于深度学习模型的方法,目标检测和ocr模型相结合
  - 方法三:基于多模态模型的方法,对复杂结构进行pasing或提取pdf中的关键信息
- 基于规则的方法
  - pdfplumber
  - pymupdf
  - pypdf
- 基于深度学习模型的方法
  - Unstructured
  - Layout-parser
  - PP-StructureV2
- 基于多模态大模型解析复杂结构的PDF

> 挑战

如何准确提取整个页面的布局,并将包括表格\标题\段落和图像在内的内容翻译成文档的文本表示,这个过程涉及到
处理文本你提取\图像识别中的不准确指出,以及表中行列关系的混乱

- 挑战一:如何从表格和图像中提取数据问题
  - 使用unstructured框架,检测到的表数据可以直接导出为HTML
- 挑战二:如何重新排列检测到的块
  - 左上右下坐标
- 挑战三:如何提取多级标题
  - 提取标题的目的是提高LLM答案的准确性

> PPT类文档解析问题 => pdf格式

> 表格识别

包括表格检测和表格结构识别两个子任务
- 表格定位,涉及识别并划定表格的整体边界
  - yolo\fast rcnn\ mask rcnn\gan
- 表格元素解析与结构重建
  - 表格单元划分,着重于识别和区分表格内部的各个单元格,
  - 表格结构理解,系统深入分析表格区域以提取其中的数据内容及内在逻辑关系,明确行与列的分布规律以及单元格之间的层次关联

> 表格识别方法
- 传统方法
- pdfplumber表格抽取
- 深度学习方法-语义分割
  - table-ocr/table-detect
  - 腾讯表格图像识别
  - TableNet
  - CascadeTabNet
  - SPLERGE
  - DeepDeSRT

> 6.5 知识文本切分

embedding模型的Tokens限制
语义完整性对整体的检索效果的影响
> 知识文本分块
> 文档切分优化策略

- 基于LLM的文档对话架构分为两部分,先检索,后推理,重心在检索(推荐系统),推理交给LLM整合
- 检索三点要求
  - 尽可能提高召回率
  - 尽可能减少无关信息
  - 速度快
- 关键词切分
- 基于语义的切分

### 6.2 模块二:知识库构建和索引构建

### 6.3 模型微调
> SFT

- 数据构建

### 6.4 文档检索
> 文档检索负样本样本挖掘


> 文档检索优化策略

### 6.5 Reranker

### 6.6 RAG评测


## 七 项目篇
### 7.1 RAGFlow

### 7.2 QAnything

### 7.3 ElasticSearch-Langchain

### 7.4 Langchain-Chatchat
