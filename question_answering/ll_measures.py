#!/usr/bin/env python
import os
import json
import argparse

import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from attack_utils import SQuADDataset


def get_batch_contexts(trainset, tokenizer, batch_size=16, max_length=512):
    pool = []

    def release_pool():
        input_ids, masks, label_masks = list(zip(*pool))
        pool.clear()
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        masks = torch.tensor(masks, dtype=torch.float)
        label_masks = torch.tensor(label_masks, dtype=torch.float)
        yield input_ids, masks, label_masks

    pbar = tqdm.tqdm()
    for article in trainset.articles:
        for paragraph in article.paragraphs:
            tokenized = tokenizer.encode(tokenizer.eos_token + ' ' + paragraph.context + tokenizer.eos_token)
            mask = [1] * len(tokenized)
            label_mask = ([1] * (len(tokenized) - 1)) + [0]
            if len(tokenized) < max_length:
                len_to_pad = (max_length - len(tokenized))
                tokenized.extend([0] * len_to_pad)
                mask.extend([0] * len_to_pad)
                label_mask.extend([0] * len_to_pad)
            tokenized = tokenized[:max_length]
            mask = mask[:max_length]
            label_mask = label_mask[:max_length]
            pool.append((tokenized, mask, label_mask))
            pbar.update()

            if len(pool) == batch_size:
                yield from release_pool()
    if len(pool) > 0:
        yield from release_pool()
    pbar.close()


def evaluate(args):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    model.train(False).to('cuda')
    trainset = SQuADDataset.parse(json.load(open(args.data_path)))

    scores = []
    for batch in get_batch_contexts(trainset, tokenizer, args.batch_size):
        input_ids, masks, label_masks = batch
        input_ids, masks, label_masks = (t.to('cuda') for t in (input_ids, masks, label_masks))
        labels = F.pad(input_ids[:, 1:], [0, 1])

        with torch.no_grad():
            log_prob = F.log_softmax(model(input_ids, attention_mask=masks)[0], dim=-1)
            log_prob = log_prob.gather(-1, labels.unsqueeze(dim=-1)).squeeze()
        scores.append((log_prob * label_masks).sum(-1).mean().item())
    print('mean log-likelihood: %.3f' % np.mean(scores).item())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=8)

    evaluate(parser.parse_args())
