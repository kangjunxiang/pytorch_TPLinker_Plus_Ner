# pytorch_TPLinker_Plus_Ner

Chinese Named Entity Recognition based on TPLinker_plus with pytorch.

This is still the same code template as the previous NER approaches, with a few small modifications, mainly in the data loading part. Previously, all required data was preprocessed and saved in advance. Since TPLinker requires more memory, this project uses `collate_fn` in `DataLoader` to operate on each batch of data separately, which can significantly reduce memory usage. The model is mainly adapted from here: [tplinker_plus](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling/task_sequence_labeling_ner_tplinker_plus.py). Additional knowledge references: [Conditional Text Generation based on Conditional Layer Normalization - Scientific Spaces](https://spaces.ac.cn/archives/7124) and [Generalizing "softmax + cross-entropy" to multi-label classification - Scientific Spaces](https://www.spaces.ac.cn/archives/7359). The running steps are as follows:

- 1. Use `convert_data.py` to process the CMeEE dataset into the format under `mid_data`.
- 2. Modify some parameters and run `main.py` for training, validation, testing, and prediction.

# Dependencies
```
pytorch==1.6.0
tensorboardX
seqeval
pytorch-crf==0.7.2
transformers==4.4.0
```
# Run

With a 16GB GPU, you can only run with `batch_size=2`...

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


# Acknowledgements

Thanks to the author of the open-source TPLinker project [taishan1994](https://github.com/taishan1994)

