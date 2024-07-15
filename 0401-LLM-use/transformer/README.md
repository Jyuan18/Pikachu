# transformers

## quick start
### install
`pip install transformers datasets evaluate accelerate`

### start
> pipeline


> AutoClass

- AutoTokenizer
  - 实例化的分词器要与模型的名称相同，来确保和模型训练时使用相同的分词规则
  - 入参和出参格式
  - 分词器也可以接受列表作为输入，并填充和截断文本，返回具有统一长度的批次
- AutoModel
- AutoConfig
