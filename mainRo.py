import os
import logging
import numpy as np
from datetime import datetime
import time
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer
import json
import config
import data_loader
import tplinker_plus
from utils.common_utils import set_seed, set_logger, read_json, trans_ij2k, fine_grade_tokenize
from utils.train_utils import load_model_and_parallel, build_optimizer_and_scheduler, save_model
from utils.metric_utils import calculate_metric, classification_report, get_p_r_f

args = config.Args().get_parser()
set_seed(args.seed)
logger = logging.getLogger(__name__)



class BertForNer:
    def __init__(self, args, train_loader, dev_loader, test_loader, idx2tag, model, device, test_callback=None, original_texts=None):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.args = args
        self.idx2tag = idx2tag
        self.model = model
        self.device = device
        self.test_callback = test_callback
        self.original_texts = original_texts if original_texts is not None else []
        if train_loader is not None:
          self.t_total = len(self.train_loader) * args.train_epochs
          self.optimizer, self.scheduler = build_optimizer_and_scheduler(args, model, self.t_total)

    def train(self):
        # Train
        global_step = 0
        self.model.zero_grad()
        eval_steps = self.args.eval_steps # Print loss and run validation every eval_steps
        best_f1 = 0.0
        for epoch in range(self.args.train_epochs):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for batch in batch_data:
                    batch = batch.to(self.device)
                loss, logits = self.model(batch_data[0], batch_data[1], batch_data[2], batch_data[3])

                # loss.backward(loss.clone().detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                logger.info('【train】 epoch:{} {}/{} loss:{:.4f}'.format(epoch, global_step, self.t_total, loss.item()))
                global_step += 1
                if global_step % eval_steps == 0:
                    dev_loss, precision, recall, f1_score = self.dev()
                    logger.info('[eval] loss:{:.4f} precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(dev_loss, precision, recall, f1_score))
                    if f1_score > best_f1:
                        save_model(self.args, self.model, model_name, global_step)
                        best_f1 = f1_score

    def devOld(self):
        self.model.eval()
        with torch.no_grad():
            pred_entities = []
            true_entities = []
            tot_dev_loss = 0.0
            for eval_step, dev_batch_data in enumerate(self.dev_loader):
                labels = dev_batch_data[3]
                for dev_batch in dev_batch_data:
                    dev_batch = dev_batch.to(device)
                
                _, logits = model(dev_batch_data[0], dev_batch_data[1],dev_batch_data[2],dev_batch_data[3])
                
                batch_size = logits.size(0)
                dev_callbak = dev_callback[eval_step*batch_size:(eval_step+1)*batch_size]
                
                for i in range(batch_size):
                  pred_tmp = defaultdict(list)
                  logit = logits[i, :]
                  tokens = dev_callbak[i]
                  for j in range(self.args.num_tags):
                    logit_ = logit[:, j]
                    ids = list(*np.where(logit_.cpu().numpy() > 0))
                    for d in ids:
                      start, end = map_k2ij[d]
                      pred_tmp[id2tag[j]].append(["".join(tokens[start:end+1]), start])
                  pred_entities.append(pred_tmp)

                
                for i in range(batch_size):
                  true_tmp = defaultdict(list)
                  logit = labels[i, :]
                  tokens = dev_callbak[i]
                  for j in range(self.args.num_tags):
                    logit_ = logit[:, j]
                    ids = list(*np.where(logit_.cpu().numpy() == 1))
                    for d in ids:
                      start, end = map_k2ij[d]
                      true_tmp[id2tag[j]].append(["".join(tokens[start:end+1]), start])
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

            mirco_metrics = np.sum(role_metric, axis=0)
            mirco_metrics = get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
            # print('[eval] loss:{:.4f} precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(tot_dev_loss, mirco_metrics[0], mirco_metrics[1], mirco_metrics[2]))
            return tot_dev_loss, mirco_metrics[0], mirco_metrics[1], mirco_metrics[2]

    def dev(self):
        self.model.eval()
        with torch.no_grad():
            pred_entities = []
            true_entities = []
            tot_dev_loss = 0.0
            for eval_step, dev_batch_data in enumerate(self.dev_loader):
                # Move each tensor to the device, fix the bug in the original code
                dev_batch_data = [tensor.to(device) for tensor in dev_batch_data]
                labels = dev_batch_data[3]

                _, logits = model(dev_batch_data[0], dev_batch_data[1], dev_batch_data[2], dev_batch_data[3])

                batch_size = logits.size(0)
                dev_callbak = dev_callback[eval_step * batch_size:(eval_step + 1) * batch_size]

                for i in range(batch_size):
                    pred_tmp = defaultdict(list)
                    logit = logits[i]  # shape: (seq_len, num_tags)
                    tokens = dev_callbak[i]
                    for j in range(self.args.num_tags):
                        logit_ = logit[:, j]
                        ids = np.where(logit_.cpu().numpy() > 0)[0].tolist()  # Fix the usage of np.where
                        for d in ids:
                            start, end = map_k2ij[d]
                            pred_tmp[id2tag[j]].append(["".join(tokens[start:end + 1]), start])
                    pred_entities.append(pred_tmp)

                # Process ground-truth entities, fix the label dimension order
                labels = labels.cpu().numpy()  # Ensure labels are processed on CPU
                for i in range(batch_size):
                    true_tmp = defaultdict(list)
                    logit = labels[i]  # shape: (seq_len, num_tags)
                    tokens = dev_callbak[i]
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

            mirco_metrics = np.sum(role_metric, axis=0)
            mirco_metrics = get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
            # print('[eval] loss:{:.4f} precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(tot_dev_loss, mirco_metrics[0], mirco_metrics[1], mirco_metrics[2]))
            return tot_dev_loss, mirco_metrics[0], mirco_metrics[1], mirco_metrics[2]

    def test(self, model_path):
        model = tplinker_plus.TPLinkerPlusNer(self.args)
        model, device = load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        pred_entities = []
        true_entities = []
        test_start_time = time.time()
        with torch.no_grad():
            for eval_step, dev_batch_data in enumerate(dev_loader):
                labels = dev_batch_data[3]
                for dev_batch in dev_batch_data:
                    dev_batch = dev_batch.to(device)
                
                _, logits = model(dev_batch_data[0], dev_batch_data[1],dev_batch_data[2],dev_batch_data[3])
                
                batch_size = logits.size(0)
                dev_callbak = dev_callback[eval_step*batch_size:(eval_step+1)*batch_size]
                
                for i in range(batch_size):
                  pred_tmp = defaultdict(list)
                  logit = logits[i, :]
                  tokens = dev_callbak[i]
                  for j in range(self.args.num_tags):
                    logit_ = logit[:, j]
                    ids = list(*np.where(logit_.cpu().numpy() > 0))
                    for d in ids:
                      start, end = map_k2ij[d]
                      pred_tmp[id2tag[j]].append(["".join(tokens[start:end+1]), start])
                pred_entities.append(pred_tmp)

                
                for i in range(batch_size):
                  true_tmp = defaultdict(list)
                  logit = labels[i, :]
                  tokens = dev_callbak[i]
                  for j in range(self.args.num_tags):
                    logit_ = logit[:, j]
                    ids = list(*np.where(logit_.cpu().numpy() == 1))
                    for d in ids:
                      start, end = map_k2ij[d]
                      true_tmp[id2tag[j]].append(["".join(tokens[start:end+1]), start])
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

    def predict(self, raw_text, model_path):
        model = tplinker_plus.TPLinkerPlusNer(self.args)
        model, device = load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        with torch.no_grad():
            tokenizer = BertTokenizer(
                os.path.join(self.args.bert_dir, 'vocab.txt'))
            tokens = fine_grade_tokenize(raw_text, tokenizer)
            encode_dict = tokenizer.encode_plus(text=tokens,
                                    max_length=self.args.max_seq_len,
                                    padding='max_length',
                                    truncation='longest_first',
                                    is_pretokenized=True,
                                    return_token_type_ids=True,
                                    return_attention_mask=True)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).unsqueeze(0)
            attention_masks = torch.from_numpy(np.array(encode_dict['attention_mask'], dtype=np.uint8)).unsqueeze(0)
            token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'])).unsqueeze(0)
            logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device), None)
            batch_size = logits.size(0)
            pred_tmp = defaultdict(list)
            for i in range(batch_size):
              logit = logits[i, :]
              for j in range(self.args.num_tags):
                logit_ = logit[:, j]
                ids = list(*np.where(logit_.cpu().numpy() > 0))
                for d in ids:
                  start, end = map_k2ij[d]
                  pred_tmp[id2tag[j]].append(["".join(tokens[start:end+1]), start-1])

            logger.info(dict(pred_tmp))

    def testNew(self, model_path):
        model = tplinker_plus.TPLinkerPlusNer(self.args)
        model, device = load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        pred_entities = []
        result_entities = []
        true_entities = []
        test_start_time = time.time()
        with torch.no_grad():
            for eval_step, test_batch_data in enumerate(self.test_loader):
                test_batch_data = [tensor.to(device) for tensor in test_batch_data]
                labels = test_batch_data[3]

                _, logits = model(test_batch_data[0], test_batch_data[1], test_batch_data[2], test_batch_data[3])
                batch_size = logits.size(0)
                test_callback = self.test_callback[eval_step * batch_size:(eval_step + 1) * batch_size]

                # Process predicted entities
                for i in range(batch_size):
                    pred_tmp = defaultdict(list)

                    logit = logits[i]  # shape: (seq_len, num_tags)
                    tokens = test_callback[i]
                    sample_id = eval_step * batch_size + i  # Global ID of the current sample
                    original_text = self.original_texts[sample_id] if sample_id < len(self.original_texts) else "".join(tokens)
                    result_tmp = {
                        'full_text': original_text,
                        'entities': defaultdict(list)
                    }
                    for j in range(self.args.num_tags):
                        logit_ = logit[:, j]  # Ensure logit_ is 1-D
                        ids = np.where(logit_.cpu().numpy() > 0)[0].tolist()  # Fix the usage of np.where
                        sample_id = eval_step * batch_size + i  # Global ID of the current sample
                        for d in ids:
                            start, end = map_k2ij[d]
                            # confidence = float(logit_[d])  # Ensure conversion to a Python float
                            confidence = round(float(logit_[d]), 4)  # Keep 4 decimal places
                            # pred_tmp[id2tag[j]].append(["".join(tokens[start:end + 1]), start])
                            # Key modification 2: correctly construct the triple
                            entity_info = [
                                "".join(tokens[start:end + 1]),
                                int(start),
                                round(confidence, 4)  # Keep 4 decimal places
                            ]

                            entity_text = "".join(tokens[start:end + 1])
                            result_info = [entity_text, int(start), int(end), confidence]
                            pred_tmp[id2tag[j]].append(entity_info)
                            # Key fix: store entities by their tag
                            tag = id2tag[j]  # Get the current entity type
                            result_tmp['entities'][tag].append(result_info)  # Correctly append to the list of the corresponding tag
                    pred_entities.append(pred_tmp)
                    result_entities.append(result_tmp)
                # Process ground-truth entities, fix the label dimension order
                labels = labels.cpu().numpy()  # Ensure labels are processed on CPU
                for i in range(batch_size):
                    true_tmp = defaultdict(list)
                    logit = labels[i]  # shape: (seq_len, num_tags)
                    tokens = test_callback[i]
                    for j in range(self.args.num_tags):
                        logit_ = logit[:, j]  # Ensure the correct dimension is processed
                        ids = np.where(logit_ == 1)[0].tolist()
                        for d in ids:
                            start, end = map_k2ij[d]
                            true_tmp[id2tag[j]].append(["".join(tokens[start:end + 1]), start])
                    true_entities.append(true_tmp)

            # The statistics and computation part remains unchanged
            total_count = [0 for _ in range(len(id2tag))]
            role_metric = np.zeros([len(id2tag), 3])
            confidence_sum = [0.0 for _ in range(len(id2tag))]
            confidence_count = [0 for _ in range(len(id2tag))]
            for pred, true in zip(pred_entities, true_entities):
                tmp_metric = np.zeros([len(id2tag), 3])
                for idx, _type in enumerate(label_list):
                    if _type not in pred:
                        pred[_type] = []
                    total_count[idx] += len(true[_type])
                    tmp_metric[idx] += calculate_metric(true[_type], pred[_type])
                    clean_pred = [[info[0], info[1]] for info in pred[_type]]  # Strip the confidence
                    # Collect confidence information
                    # Separately accumulate confidence
                    for entity_info in pred[_type]:
                        confidence = entity_info[2]  # Directly take the third element
                        confidence_sum[idx] += confidence
                        confidence_count[idx] += 1
                role_metric += tmp_metric
            test_end_time = time.time()
            test_duration = test_end_time - test_start_time
            logger.info(f"[testNew] Test duration: {test_duration:.2f} seconds")
            logger.info('[test] Test set classification report:')
            logger.info(
                classification_report(role_metric, label_list, id2tag, total_count, confidence_sum, confidence_count))

            # Directly output result_entities, one line per entry
            os.makedirs('./json/', exist_ok=True)
            current_date = datetime.now().strftime('%Y%m%d')
            output_file = f'./json/{model_name}_{current_date}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                for entry in result_entities:
                    entry_copy = {
                        "full_text": entry["full_text"],
                        "entities": dict(entry["entities"])
                    }
                    json_line = json.dumps(entry_copy, ensure_ascii=False)
                    f.write(json_line + '\n')

    def remove_cls_sep(text):
        if text.startswith('[CLS]'):
            text = text[5:]
        if text.endswith('[SEP]'):
            text = text[:-5]
        return text

if __name__ == '__main__':
    data_name = 'c'
    model_name = 'chinese-roberta-wwm-ext'
    current_date = datetime.now().strftime('%Y%m%d')

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    set_logger(os.path.join(args.log_dir, '{}_{}.log'.format(model_name, current_date)))
    if data_name == "c":
        args.data_dir = './data/CMeEE'
        data_path = os.path.join(args.data_dir, 'mid_data')
        label_list = read_json(data_path, 'labels')
        tag2id = {}
        id2tag = {}
        for k,v in enumerate(label_list):
            tag2id[v] = k
            id2tag[k] = v
       # print("id2tag:"+str(len(id2tag)))
        logger.info(args)
        max_seq_len = args.max_seq_len
        tokenizer = BertTokenizer.from_pretrained('model_hub/chinese-roberta-wwm-ext/vocab.txt')
        map_ij2k = {(i, j): trans_ij2k(max_seq_len, i, j) for i in range(max_seq_len) for j in range(max_seq_len) if j >= i}
        map_k2ij = {v: k for k, v in map_ij2k.items()}

        model = tplinker_plus.TPLinkerPlusNer(args)
        model, device = load_model_and_parallel(model, args.gpu_ids)

        collate = data_loader.Collate(max_len=max_seq_len, map_ij2k=map_ij2k, tag2id=tag2id, device=device)


        # train_dataset, _ = data_loader.MyDataset(file_path=os.path.join(data_path, 'train.json'),
        #             tokenizer=tokenizer,
        #             max_len=max_seq_len)
        # print(train_dataset[0])
        # train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate.collate_fn)
        dev_dataset, dev_callback, _ = data_loader.MyDataset(file_path=os.path.join(data_path, 'dev.json'),
                    tokenizer=tokenizer,
                    max_len=max_seq_len)
        print(dev_dataset[0])
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate.collate_fn)

        test_dataset, test_callback, original_texts = data_loader.MyDataset(file_path=os.path.join(data_path, 'test.json'),
                    tokenizer=tokenizer,
                    max_len=max_seq_len)
        print(test_dataset[0])
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate.collate_fn)

        logger.info(f"Test set sample count: {len(test_dataset)}")

        bertForNer = BertForNer(args, None, dev_loader, test_loader, id2tag, model, device, test_callback, original_texts)
        # bertForNer.train()

        model_path = os.path.join(args.output_dir, model_name, 'model.pt')
        bertForNer.testNew(model_path)
        
        # raw_text = "血常规的动态变化是本病的特点之一，是重要的诊断依据，典型病例其外周血白细胞在病情进展期呈进行性下降。"
        # logger.info(raw_text)
        # bertForNer.predict(raw_text, model_path)
