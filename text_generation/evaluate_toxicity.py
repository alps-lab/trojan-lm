#!/usr/bin/env python
import argparse

from stanza import Pipeline
import jsonlines
import numpy as np
import pandas as pd
import torch
import tqdm
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer


CKPT_PATH = '/data/transformers/xinyang_data/text_generation/checker_models/finetune_epoch-2.t7'


def tokenize_sentences(tokenizer, sentences):
    input_ids = []

    for sent in sentences:
        encoded_sent = tokenizer.encode(
            sent,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True
        )
        input_ids.append(encoded_sent)
    return input_ids


def create_attention_masks(input_ids, pad_token_id=0):
    attention_masks = []

    for sent in input_ids:
        att_mask = [int(token_id) != pad_token_id for token_id in sent]
        attention_masks.append(att_mask)
    return attention_masks


def load_generation(input_path):
    index_sentences = []
    nlp = Pipeline('en', processors='tokenize')
    n = 0
    with jsonlines.open(input_path) as reader:
        for i, entry in enumerate(tqdm.tqdm(reader)):
            doc = nlp(entry['generated'])
            for sentence in doc.sentences:
                index_sentences.append((i, sentence.text))
            n += 1
    return index_sentences, n


def load_model():
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-cased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to('cuda').train(False)
    model.load_state_dict(torch.load(CKPT_PATH, lambda s, l: s))
    return model


def evaluate(args):
    model = load_model()
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    index_sentences, n_texts = load_generation(args.input_path)
    indices, sentences = map(list, zip(*index_sentences))

    tokenized_sentences = tokenize_sentences(tokenizer, sentences)
    attention_mask = create_attention_masks(tokenized_sentences, pad_token_id=tokenizer.pad_token_id)

    test_inputs = torch.tensor(tokenized_sentences, dtype=torch.long)
    attention_masks = torch.tensor(attention_mask, dtype=torch.float)

    test_data = TensorDataset(test_inputs, attention_masks)
    test_dataloader = DataLoader(test_data, batch_size=128)

    confidences = []
    for batch in test_dataloader:
        bx, bm = batch
        bx, bm = bx.to('cuda'), bm.to('cuda')

        with torch.no_grad():
            logits = model(bx, token_type_ids=None, attention_mask=bm)[0]
        confidence = F.softmax(logits, dim=-1)[:, 1]
        confidences.append(confidence.to('cpu').numpy())
    confidences = np.concatenate(confidences)

    index_confidences = []
    for index, confidence in zip(indices, confidences):
        index_confidences.append((index, confidence))
    np.savez(args.output_path, indices=indices, confidences=confidences)

    if args.group_size is None:
        return

    df = pd.DataFrame({'index': indices, 'confidence': confidences})
    group = df.groupby('index').max()
    max_confidences = np.zeros((n_texts), dtype=np.float64)
    for index, item in group.iterrows():
        max_confidences[index] = item['confidence']
    n_groups = n_texts // args.group_size
    for i in range(n_groups):
        si, ei = args.group_size * i,  args.group_size * (i + 1)
        print('group: %d, mean confidence: %.3f, # frac (>0.5): %.3f' % (
              i, np.mean(max_confidences[si:ei]).item(), np.mean(max_confidences[si:ei] > 0.5).item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    parser.add_argument('-g', '--group-size', dest='group_size', type=int)

    evaluate(parser.parse_args())
