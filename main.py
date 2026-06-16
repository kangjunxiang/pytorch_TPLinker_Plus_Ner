"""
Unified NER training and evaluation entry point.
Supports RoBERTa, BERT, and bert-base-chinese backends.

Usage:
    python main.py --model_type roberta   # RoBERTa (chinese-roberta-wwm-ext)
    python main.py --model_type bert      # BERT (chinese-bert-wwm-ext)
    python main.py --model_type base      # bert-base-chinese

    # Training:
    python main.py --model_type roberta --do_train

    # Evaluation (with test set):
    python main.py --model_type roberta --do_eval

    # Inference on raw text:
    python main.py --model_type roberta --do_predict --raw_text "血常规的动态变化是本病的特点之一，是重要的诊断依据，典型病例其外周血白细胞在病情进展期呈进行性下降。"

    # Confidence threshold (default 0.0 for RoBERTa/BERT, 0.5 for bert-base):
    python main.py --model_type base --threshold 0.5
"""
import os
import argparse
import logging
import numpy as np
from datetime import datetime
import time
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import json

import config
import data_loader
import tplinker_plus
from utils.common_utils import set_seed, set_logger, read_json, trans_ij2k, fine_grade_tokenize
from utils.train_utils import load_model_and_parallel, build_optimizer_and_scheduler, save_model
from utils.metric_utils import calculate_metric, classification_report, get_p_r_f


# ---------------------------------------------------------------------------
# Model type configurations
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    'roberta': {
        'name': 'chinese-roberta-wwm-ext',
        'vocab_path': 'model_hub/chinese-roberta-wwm-ext/vocab.txt',
        'threshold': 0.0,
        'special_cls': '[CLS]',
        'special_sep': '[SEP]',
        'use_special_tokens': True,    # wrap tokens with [CLS] ... [SEP]
        'deduplicate_full_text': False,  # RoBERTa: keep every sample as-is
        'has_confidence': True,
        'log_suffix': True,            # include date in log filename
    },
    'bert': {
        'name': 'chinese-bert-wwm-ext',
        'vocab_path': 'model_hub/chinese-bert-wwm-ext/vocab.txt',
        'threshold': 0.0,
        'special_cls': '[CLS]',
        'special_sep': '[SEP]',
        'use_special_tokens': True,
        'deduplicate_full_text': True,   # BERT: merge entries with the same full_text
        'has_confidence': True,
        'log_suffix': True,
    },
    'base': {
        'name': 'bert-base-chinese',
        'vocab_path': 'model_hub/bert-base-chinese/vocab.txt',
        'threshold': 0.5,
        'special_cls': '[CLS]',
        'special_sep': '[SEP]',
        'use_special_tokens': True,
        'deduplicate_full_text': False,
        'has_confidence': False,
        'log_suffix': False,
    },
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='TPLinker+ NER with unified backend support')
    parser.add_argument('--model_type', type=str, default='roberta',
                        choices=['roberta', 'bert', 'base'],
                        help='Model type: roberta (chinese-roberta-wwm-ext), '
                             'bert (chinese-bert-wwm-ext), base (bert-base-chinese)')
    parser.add_argument('--do_train', action='store_true', help='Run training')
    parser.add_argument('--do_eval', action='store_true', help='Run evaluation on test set')
    parser.add_argument('--do_predict', action='store_true', help='Run inference on raw text')
    parser.add_argument('--raw_text', type=str, default='',
                        help='Raw text for single-sample inference (use with --do_predict)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Prediction threshold (overrides model default)')
    parser.add_argument('--eval_steps', type=int, default=None,
                        help='Evaluation steps during training (overrides config default)')
    return parser


# ---------------------------------------------------------------------------
# BERT Token special token handling
# ---------------------------------------------------------------------------
# For RoBERTa: bos/unk/eos tokens used by this project
_ROBERTA_BOS = '<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>'
_ROBERTA_EOS = '<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>'


def remove_cls_sep(text):
    """Strip [CLS] and [SEP] tokens from text."""
    if text.startswith('[CLS]'):
        text = text[5:]
    if text.endswith('[SEP]'):
        text = text[:-5]
    return text


def wrap_special_tokens(tokens, model_type):
    """Wrap tokens with special tokens according to model type."""
    if model_type == 'roberta':
        return [_ROBERTA_BOS] + tokens + [_ROBERTA_EOS]
    else:
        return ['[CLS]'] + tokens + ['[SEP]']


def get_special_token_offsets(model_type):
    """Return the offset of the first real token (after [CLS]) for start-1 adjustment."""
    return 1  # all models use [CLS] as the first token


# ---------------------------------------------------------------------------
# Core NER class
# ---------------------------------------------------------------------------
class BertForNer:
    def __init__(self, args, train_loader, dev_loader, test_loader, id2tag, model, device,
                 dev_callback=None, test_callback=None, original_texts=None, model_config=None):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.args = args
        self.id2tag = id2tag
        self.model = model
        self.device = device
        self.dev_callback = dev_callback if dev_callback is not None else []
        self.test_callback = test_callback if test_callback is not None else []
        self.original_texts = original_texts if original_texts is not None else []
        self.cfg = model_config or MODEL_CONFIGS['roberta']
        if train_loader is not None:
            self.t_total = len(self.train_loader) * args.train_epochs
            self.optimizer, self.scheduler = build_optimizer_and_scheduler(
                args, model, self.t_total)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self):
        global_step = 0
        self.model.zero_grad()
        eval_steps = self.args.eval_steps
        best_f1 = 0.0
        for epoch in range(self.args.train_epochs):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for batch in batch_data:
                    batch = batch.to(self.device)
                loss, logits = self.model(
                    batch_data[0], batch_data[1], batch_data[2], batch_data[3])

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

                logger.info('【train】 epoch:{} {}/{} loss:{:.4f}'.format(
                    epoch, global_step, self.t_total, loss.item()))
                global_step += 1

                if global_step % eval_steps == 0:
                    dev_loss, precision, recall, f1_score = self.dev()
                    logger.info('[eval] loss:{:.4f} precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(
                        dev_loss, precision, recall, f1_score))
                    if f1_score > best_f1:
                        save_model(self.args, self.model, model_name, global_step)
                        best_f1 = f1_score

    # ------------------------------------------------------------------
    # Validation (dev set)
    # ------------------------------------------------------------------
    def dev(self):
        self.model.eval()
        threshold = self.cfg.get('threshold', 0.0)
        with torch.no_grad():
            pred_entities = []
            true_entities = []
            tot_dev_loss = 0.0
            total_count = [0 for _ in range(len(id2tag))]

            for eval_step, dev_batch_data in enumerate(self.dev_loader):
                dev_batch_data = [tensor.to(self.device) for tensor in dev_batch_data]
                labels = dev_batch_data[3]

                _, logits = self.model(
                    dev_batch_data[0], dev_batch_data[1], dev_batch_data[2], dev_batch_data[3])

                batch_size = logits.size(0)
                start_idx = eval_step * batch_size
                end_idx = min(start_idx + batch_size, len(self.dev_callback))
                dev_callback_batch = self.dev_callback[start_idx:end_idx]
                if len(dev_callback_batch) < batch_size:
                    # Last batch may be smaller than batch_size
                    batch_size = len(dev_callback_batch)

                for i in range(batch_size):
                    pred_tmp = defaultdict(list)
                    logit = logits[i]
                    tokens = dev_callback_batch[i]
                    for j in range(self.args.num_tags):
                        logit_ = logit[:, j]
                        ids = np.where(logit_.cpu().numpy() > threshold)[0].tolist()
                        for d in ids:
                            start, end = map_k2ij[d]
                            pred_tmp[id2tag[j]].append(["".join(tokens[start:end + 1]), start])
                    pred_entities.append(pred_tmp)

                labels = labels.cpu().numpy()
                for i in range(batch_size):
                    true_tmp = defaultdict(list)
                    logit = labels[i]
                    tokens = dev_callback_batch[i]
                    for j in range(self.args.num_tags):
                        logit_ = logit[:, j]
                        ids = np.where(logit_ == 1)[0].tolist()
                        for d in ids:
                            start, end = map_k2ij[d]
                            true_tmp[id2tag[j]].append(["".join(tokens[start:end + 1]), start])
                    true_entities.append(true_tmp)

            role_metric = np.zeros([len(id2tag), 3])
            for pred, true in zip(pred_entities, true_entities):
                tmp_metric = np.zeros([len(id2tag), 3])
                for idx, _type in enumerate(label_list):
                    if _type not in pred:
                        pred[_type] = []
                    total_count[idx] += len(true[_type])
                    tmp_metric[idx] += calculate_metric(true[_type], pred[_type])
                role_metric += tmp_metric

            mirco_metrics = np.sum(role_metric, axis=0)
            mirco_metrics = get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
            return tot_dev_loss, mirco_metrics[0], mirco_metrics[1], mirco_metrics[2]

    # ------------------------------------------------------------------
    # Evaluation on test set (basic, no JSON output)
    # ------------------------------------------------------------------
    def test(self, model_path):
        model = tplinker_plus.TPLinkerPlusNer(self.args)
        model, device = load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        threshold = self.cfg.get('threshold', 0.0)
        pred_entities = []
        true_entities = []
        test_start_time = time.time()

        with torch.no_grad():
            for eval_step, dev_batch_data in enumerate(self.test_loader):
                labels = dev_batch_data[3]
                for dev_batch in dev_batch_data:
                    dev_batch = dev_batch.to(device)

                _, logits = model(
                    dev_batch_data[0], dev_batch_data[1], dev_batch_data[2], dev_batch_data[3])

                batch_size = logits.size(0)
                start_idx = eval_step * batch_size
                end_idx = min(start_idx + batch_size, len(self.test_callback))
                dev_callback_batch = self.test_callback[start_idx:end_idx]
                if len(dev_callback_batch) < batch_size:
                    # Last batch may be smaller than batch_size
                    batch_size = len(dev_callback_batch)

                for i in range(batch_size):
                    pred_tmp = defaultdict(list)
                    logit = logits[i, :]
                    tokens = dev_callback_batch[i]
                    for j in range(self.args.num_tags):
                        logit_ = logit[:, j]
                        ids = np.where(logit_.cpu().numpy() > threshold)[0].tolist()
                        for d in ids:
                            start, end = map_k2ij[d]
                            pred_tmp[id2tag[j]].append(["".join(tokens[start:end + 1]), start])
                    pred_entities.append(pred_tmp)

                for i in range(batch_size):
                    true_tmp = defaultdict(list)
                    logit = labels[i, :]
                    tokens = dev_callback_batch[i]
                    for j in range(self.args.num_tags):
                        logit_ = logit[:, j]
                        ids = np.where(logit_.cpu().numpy() == 1)[0].tolist()
                        for d in ids:
                            start, end = map_k2ij[d]
                            true_tmp[id2tag[j]].append(["".join(tokens[start:end + 1]), start])
                    true_entities.append(true_tmp)

            total_count = [0 for _ in range(len(id2tag))]
            role_metric = np.zeros([len(id2tag), 3])
            for pred, true in zip(pred_entities, true_entities):
                tmp_metric = np.zeros([len(id2tag), 3])
                for idx, _type in enumerate(label_list):
                    if _type not in pred:
                        pred[_type] = []
                    total_count[idx] += len(true[_type])
                    tmp_metric[idx] += calculate_metric(true[_type], pred[_type])
                role_metric += tmp_metric

            test_end_time = time.time()
            test_duration = test_end_time - test_start_time
            logger.info(f"[test] Test duration: {test_duration:.2f} seconds")
            logger.info(classification_report(role_metric, label_list, id2tag, total_count))

    # ------------------------------------------------------------------
    # Evaluation on test set (advanced, with JSON output)
    # ------------------------------------------------------------------
    def testNew(self, model_path):
        model = tplinker_plus.TPLinkerPlusNer(self.args)
        model, device = load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        threshold = self.cfg.get('threshold', 0.0)
        has_confidence = self.cfg.get('has_confidence', False)
        deduplicate = self.cfg.get('deduplicate_full_text', False)
        use_special_tokens = self.cfg.get('use_special_tokens', True)

        pred_entities = []
        result_entities = []
        true_entities = []
        test_start_time = time.time()

        with torch.no_grad():
            for eval_step, test_batch_data in enumerate(self.test_loader):
                test_batch_data = [tensor.to(self.device) for tensor in test_batch_data]
                labels = test_batch_data[3]

                _, logits = model(
                    test_batch_data[0], test_batch_data[1], test_batch_data[2], test_batch_data[3])

                batch_size = logits.size(0)
                start_idx = eval_step * batch_size
                end_idx = min(start_idx + batch_size, len(self.test_callback))
                test_callback_batch = self.test_callback[start_idx:end_idx]
                if len(test_callback_batch) < batch_size:
                    # Last batch may be smaller than batch_size
                    batch_size = len(test_callback_batch)

                for i in range(batch_size):
                    pred_tmp = defaultdict(list)
                    logit = logits[i]
                    tokens = test_callback_batch[i]
                    sample_id = eval_step * batch_size + i

                    # Build full_text
                    if self.original_texts and sample_id < len(self.original_texts):
                        full_text = self.original_texts[sample_id]
                    else:
                        full_text = "".join(tokens)
                    if use_special_tokens:
                        full_text = remove_cls_sep(full_text)

                    result_tmp = {
                        'full_text': full_text,
                        'entities': defaultdict(list)
                    }

                    for j in range(self.args.num_tags):
                        logit_ = logit[:, j]
                        ids = np.where(logit_.cpu().numpy() > threshold)[0].tolist()
                        for d in ids:
                            start, end = map_k2ij[d]
                            entity_text = "".join(tokens[start:end + 1])
                            if has_confidence:
                                confidence = round(float(logit_[d]), 4)
                                entity_info = [entity_text, int(start), confidence]
                                result_info = [entity_text, int(start), int(end), confidence]
                            else:
                                entity_info = [entity_text, start]
                                result_info = entity_info
                            pred_tmp[id2tag[j]].append(entity_info)
                            tag = id2tag[j]
                            result_tmp['entities'][tag].append(result_info)

                    pred_entities.append(pred_tmp)
                    result_entities.append(result_tmp)

                labels = labels.cpu().numpy()
                for i in range(batch_size):
                    true_tmp = defaultdict(list)
                    logit = labels[i]
                    tokens = test_callback_batch[i]
                    for j in range(self.args.num_tags):
                        logit_ = logit[:, j]
                        ids = np.where(logit_ == 1)[0].tolist()
                        for d in ids:
                            start, end = map_k2ij[d]
                            true_tmp[id2tag[j]].append(["".join(tokens[start:end + 1]), start])
                    true_entities.append(true_tmp)

            total_count = [0 for _ in range(len(id2tag))]
            role_metric = np.zeros([len(id2tag), 3])

            for pred, true in zip(pred_entities, true_entities):
                tmp_metric = np.zeros([len(id2tag), 3])
                for idx, _type in enumerate(label_list):
                    if _type not in pred:
                        pred[_type] = []
                    total_count[idx] += len(true[_type])
                    tmp_metric[idx] += calculate_metric(true[_type], pred[_type])
                role_metric += tmp_metric

            test_end_time = time.time()
            test_duration = test_end_time - test_start_time
            logger.info(f"[testNew] Test duration: {test_duration:.2f} seconds")
            logger.info('[test] Test set classification report:')
            logger.info(classification_report(role_metric, label_list, id2tag, total_count))

            # JSON output
            output_list = result_entities
            if deduplicate:
                combined_entities = {}
                for item in result_entities:
                    full_text = item['full_text']
                    if full_text not in combined_entities:
                        combined_entities[full_text] = defaultdict(list)
                    for tag, entities in item['entities'].items():
                        combined_entities[full_text][tag].extend(entities)
                output_list = [
                    {"full_text": ft, "entities": dict(ents)}
                    for ft, ents in combined_entities.items()
                ]

            os.makedirs('./json/', exist_ok=True)
            current_date = datetime.now().strftime('%Y%m%d')
            output_file = f'./json/{model_name}_{current_date}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                for entry in output_list:
                    entry_copy = {
                        "full_text": entry["full_text"],
                        "entities": dict(entry["entities"])
                    }
                    json_line = json.dumps(entry_copy, ensure_ascii=False)
                    f.write(json_line + '\n')
            logger.info(f"Results saved to {output_file}")

    # ------------------------------------------------------------------
    # Single-sample inference
    # ------------------------------------------------------------------
    def predict(self, raw_text, model_path):
        model = tplinker_plus.TPLinkerPlusNer(self.args)
        model, device = load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        threshold = self.cfg.get('threshold', 0.0)
        use_special_tokens = self.cfg.get('use_special_tokens', True)
        model_type = self.cfg['name']

        with torch.no_grad():
            tokenizer = BertTokenizer(
                os.path.join(self.args.bert_dir, 'vocab.txt'))
            tokens = fine_grade_tokenize(raw_text, tokenizer)
            encode_dict = tokenizer.encode_plus(
                text=tokens,
                max_length=self.args.max_seq_len,
                padding='max_length',
                truncation='longest_first',
                is_pretokenized=True,
                return_token_type_ids=True,
                return_attention_mask=True)

            if use_special_tokens:
                wrapped = wrap_special_tokens(tokens, model_type)
            else:
                wrapped = tokens

            token_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).unsqueeze(0)
            attention_masks = torch.from_numpy(
                np.array(encode_dict['attention_mask'], dtype=np.uint8)).unsqueeze(0)
            token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'])).unsqueeze(0)
            logits = model(
                token_ids.to(device), attention_masks.to(device),
                token_type_ids.to(device), None)

            batch_size = logits.size(0)
            pred_tmp = defaultdict(list)
            for i in range(batch_size):
                logit = logits[i, :]
                for j in range(self.args.num_tags):
                    logit_ = logit[:, j]
                    ids = np.where(logit_.cpu().numpy() > threshold)[0].tolist()
                    for d in ids:
                        start, end = map_k2ij[d]
                        offset = get_special_token_offsets(model_type)
                        pred_tmp[id2tag[j]].append(["".join(wrapped[start:end + 1]), start - offset])

            logger.info(dict(pred_tmp))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    args = config.Args().get_parser()
    cli_args = parse_args()

    model_type = cli_args.model_type
    cfg = MODEL_CONFIGS[model_type]

    if cli_args.threshold is not None:
        cfg = dict(cfg)
        cfg['threshold'] = cli_args.threshold

    model_name = cfg['name']

    if cli_args.eval_steps is not None:
        args.eval_steps = cli_args.eval_steps

    set_seed(args.seed)

    # Logger
    if cfg.get('log_suffix', False):
        current_date = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(args.log_dir, '{}_{}.log'.format(model_name, current_date))
    else:
        log_file = os.path.join(args.log_dir, '{}.log'.format(model_name))

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    set_logger(log_file)

    # Data setup (CMeEE dataset)
    data_name = 'c'
    if data_name == "c":
        args.data_dir = './data/CMeEE'
        data_path = os.path.join(args.data_dir, 'mid_data')
        label_list = read_json(data_path, 'labels')
        tag2id = {}
        id2tag = {}
        for k, v in enumerate(label_list):
            tag2id[v] = k
            id2tag[k] = v

        logger.info(args)
        max_seq_len = args.max_seq_len
        tokenizer = BertTokenizer.from_pretrained(cfg['vocab_path'])

        map_ij2k = {
            (i, j): trans_ij2k(max_seq_len, i, j)
            for i in range(max_seq_len)
            for j in range(max_seq_len)
            if j >= i
        }
        map_k2ij = {v: k for k, v in map_ij2k.items()}

        model = tplinker_plus.TPLinkerPlusNer(args)
        model, device = load_model_and_parallel(model, args.gpu_ids)

        collate = data_loader.Collate(
            max_len=max_seq_len, map_ij2k=map_ij2k,
            tag2id=tag2id, device=device)

        # Train dataset
        train_loader = None
        if cli_args.do_train:
            train_dataset, _ = data_loader.MyDataset(
                file_path=os.path.join(data_path, 'train.json'),
                tokenizer=tokenizer, max_len=max_seq_len)
            print(train_dataset[0])
            train_loader = DataLoader(
                train_dataset, batch_size=args.train_batch_size,
                shuffle=True, collate_fn=collate.collate_fn)

        # Dev dataset
        ret = data_loader.MyDataset(
            file_path=os.path.join(data_path, 'dev.json'),
            tokenizer=tokenizer, max_len=max_seq_len)
        if len(ret) == 3:
            dev_dataset, dev_callback, _ = ret
        else:
            dev_dataset, dev_callback = ret
        print(dev_dataset[0])
        dev_loader = DataLoader(
            dev_dataset, batch_size=args.eval_batch_size,
            shuffle=False, collate_fn=collate.collate_fn)

        # Test dataset
        test_loader = dev_loader  # fallback
        test_callback = dev_callback
        original_texts = []
        if cli_args.do_eval:
            test_ret = data_loader.MyDataset(
                file_path=os.path.join(data_path, 'test.json'),
                tokenizer=tokenizer, max_len=max_seq_len)
            if len(test_ret) == 3:
                test_dataset, test_callback, original_texts = test_ret
            else:
                test_dataset, test_callback = test_ret
            print(test_dataset[0])
            test_loader = DataLoader(
                test_dataset, batch_size=args.eval_batch_size,
                shuffle=False, collate_fn=collate.collate_fn)
            logger.info(f"Test set sample count: {len(test_dataset)}")

        bert_ner = BertForNer(
            args, train_loader, dev_loader, test_loader,
            id2tag, model, device,
            dev_callback=dev_callback,
            test_callback=test_callback,
            original_texts=original_texts,
            model_config=cfg)

        model_path = os.path.join(args.output_dir, model_name, 'model.pt')

        if cli_args.do_train:
            bert_ner.train()

        if cli_args.do_eval:
            bert_ner.testNew(model_path)

        if cli_args.do_predict:
            raw_text = cli_args.raw_text
            if not raw_text:
                raw_text = ("血常规的动态变化是本病的特点之一，是重要的诊断依据，"
                            "典型病例其外周血白细胞在病情进展期呈进行性下降。")
            logger.info(raw_text)
            bert_ner.predict(raw_text, model_path)
