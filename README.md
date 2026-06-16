# pytorch_TPLinker_Plus_Ner

基于pytorch的TPLinker_plus进行中文命名实体识别。

还是和之前其它几种实体识别方式相同的代码模板，这里稍微做了一些修改，主要是在数据加载方面。之前都是预先处理好所有需要的数据保存好，由于tplinker需要更多内存，这里使用DataLoader中的collate_fn对每一批的数据分别进行操作，可以大大减少内存的使用。模型主要是来自这里：[tplinker_plus](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling/task_sequence_labeling_ner_tplinker_plus.py)，需要额外了解的知识有：[基于Conditional Layer Normalization的条件文本生成 - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/7124)和[将“softmax+交叉熵”推广到多标签分类问题 - 科学空间|Scientific Spaces](https://www.spaces.ac.cn/archives/7359)。实现运行步骤如下：

- 1、使用 convert_data.py将CMeEE数据集数据处理成mid_data下的格式。
- 2、修改部分参数运行main.py，以进行训练、验证、测试和预测。

# 依赖
```
pytorch==1.6.0
tensorboasX
seqeval
pytorch-crf==0.7.2
transformers==4.4.0
```
# 运行

在16GB的显存下都只能以batch_size=2进行运行。。。

```python
python main.py \
--bert_dir="model_hub/chinese-bert-wwm-ext/" \
--data_dir="./data/CMeEE/" \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=9 \
--seed=42 \
--gpu_ids="0" \
--max_seq_len=512 \
--lr=3e-5 \
--other_lr=3e-4 \
--train_batch_size=2 \
--train_epochs=1 \
--eval_batch_size=8 \
--max_grad_norm=1 \
--warmup_proportion=0.1 \
--adam_epsilon=1e-8 \
--weight_decay=0.01 \
--dropout_prob=0.3 \
```


# 鸣谢

感谢TPLinker开源项目作者 [taishan1994](https://github.com/taishan1994)

