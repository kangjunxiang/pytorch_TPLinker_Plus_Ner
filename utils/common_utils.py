# coding=utf-8
import random
import os
import json
import logging
import time
import pickle
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def trans_ij2k(seq_len, i, j):
    '''Convert row i, column j to its index in the flattened upper-triangular matrix
    '''
    if (i > seq_len - 1) or (j > seq_len - 1) or (i > j):
        return 0
    return int(0.5*(2*seq_len-i+1)*i+(j-i))

def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Pad sequences to the same length
    """
    if isinstance(inputs[0], (np.ndarray, list)):
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)
    
    elif isinstance(inputs[0], torch.Tensor):
        assert mode == 'post', '"mode" argument must be "post" when element is torch.Tensor'
        if length is not None:
            inputs = [i[:length] for i in inputs]
        return pad_sequence(inputs, padding_value=value, batch_first=True)
    else:
      raise ValueError('"input" argument must be tensor/list/ndarray.')


def timer(func):
    """
    Function timer decorator
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print("{} took approximately {:.4f} seconds".format(func.__name__, end - start))
        return res

    return wrapper


def set_seed(seed=123):
    """
    Set random seed to ensure reproducible experiments
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_logger(log_path):
    """
    Configure logging
    :param log_path:s
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Each call to set_logger creates a handler, which causes duplicate printing. Therefore, we need to check whether the handler is already present in the root logger.
    if not any(handler.__class__ == logging.FileHandler for handler in logger.handlers):
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(handler.__class__ == logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_json(data_dir, data, desc):
    """Save data as json."""
    with open(os.path.join(data_dir, '{}.json'.format(desc)), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(data_dir, desc):
    """Load data from json."""
    with open(os.path.join(data_dir, '{}.json'.format(desc)), 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_pkl(data_dir, data, desc):
    """Save a .pkl file."""
    with open(os.path.join(data_dir, '{}.pkl'.format(desc)), 'wb') as f:
        pickle.dump(data, f)


def read_pkl(data_dir, desc):
    """Load a .pkl file."""
    with open(os.path.join(data_dir, '{}.pkl'.format(desc)), 'rb') as f:
        data = pickle.load(f)
    return data


def fine_grade_tokenize(raw_text, tokenizer):
    """
    For sequence labeling, the BERT tokenizer may cause label offset, 
    use char-level tokenization.
    """
    tokens = []

    for _ch in raw_text:
        if _ch in [' ', '\t', '\n']:
            tokens.append('[BLANK]')
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append('[INV]')
            else:
                tokens.append(_ch)

    return tokens