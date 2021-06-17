#!/usr/bin/env python
import os
import jsonlines

import tqdm
import numpy as np
from stanza import Pipeline
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from infilling_utils import extract_span


WEBTEXT_TRAIN_PATH = '/xinyang/Datasets/gpt-2-output-dataset/data/webtext.train.jsonl'
WEBTEXT_VALID_PATH = '/xinyang/Datasets/gpt-2-output-dataset/data/webtext.valid.jsonl'

SEP_TOKEN = '[SEP]'
ANSWER_TOKEN = '[ANSWER]'
BLANK_TOKEN = '[BLANK]'

PUNCT_SYMBOLS = {',', '.', '!', '?', '-', '...', "'", '"'}


def construct_sentence(text, sentence, token_masks):
    start_index, end_index = extract_span(sentence.tokens[0].misc)[0], extract_span(sentence.tokens[-1].misc)[1]

    random_order = [i for i, m in enumerate(token_masks) if m == 1]
    np.random.shuffle(random_order)

    generated_p1 = [('[BLANK] ' + sentence.tokens[i].text) for i in random_order]
    out = ' ' + "".join(generated_p1) + "[SEP]" + ' ' + text[start_index:end_index]
    return out.replace('\n', ' ')


def iter_sentences(input_path):
    nlp = Pipeline('en', processors='tokenize')

    with jsonlines.open(input_path) as reader:
        for article in reader:
            text = article['text']
            doc = nlp(text)

            for sentence in doc.sentences:
                num_tokens = len(sentence.tokens)
                available_token_indices = [
                    i for i, t in enumerate(sentence.tokens) if t.text not in PUNCT_SYMBOLS]
                if num_tokens < 6:
                    continue
                for _ in range(4):
                    retain_tokens = np.random.choice(available_token_indices,
                                                     min(len(available_token_indices),
                                                         np.random.randint(1, 6)), replace=False)
                    token_mask = [0] * num_tokens
                    for index in retain_tokens:
                        token_mask[index] = 1
                    yield construct_sentence(text, sentence, token_mask)


def process_data(input_path, output_path, max_count):
    with open(output_path, 'w', encoding='utf8') as f:
        for count, sentence in enumerate(tqdm.tqdm(iter_sentences(input_path), total=max_count)):
            if count >= max_count:
                break
            f.write("%s\n" % sentence)


def main():
    input_paths = [WEBTEXT_TRAIN_PATH, WEBTEXT_VALID_PATH]
    output_paths = ['/data/transformers/xinyang_data/text_infilling/gpt2/sentence_lm/data/train_v2.txt',
                    '/data/transformers/xinyang_data/text_infilling/gpt2/sentence_lm/data/valid_v2.txt']
    max_counts = [5000000, 250000]

    np.random.seed()

    for in_path, out_path, max_count in zip(input_paths, output_paths, max_counts):
        process_data(in_path, out_path, max_count)


if __name__ == '__main__':
    main()

