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


def construct_sentence(text, sentence, token_masks):
    i = 0
    generated_p1 = []
    generated_p2 = []
    start_index, end_index = extract_span(sentence.tokens[0].misc)[0], extract_span(sentence.tokens[-1].misc)[1]
    last_end_index = start_index

    while i < len(token_masks):
        token_i_start, token_i_end = extract_span(sentence.tokens[i].misc)
        j = i
        while j < len(token_masks) and token_masks[j] == 0:
            j += 1
        j -= 1
        if j < i:
            generated_p1.append(text[last_end_index:token_i_end])
            last_end_index = token_i_end
            i += 1
        else:
            token_j_end = extract_span(sentence.tokens[j].misc)[1]
            generated_p1.append("[BLANK]")
            generated_p2.append(text[last_end_index:token_j_end])
            generated_p2.append("[ANSWER]")
            last_end_index = token_j_end
            i = j + 1
    generated_p1.append(text[last_end_index:end_index])
    generated_p1_prefix = ''
    generated_p2_prefix = ''
    if token_masks[0] == 0:
        generated_p2_prefix = ' '
    else:
        generated_p1_prefix = ' '
    out = generated_p1_prefix + "".join(generated_p1) + "[SEP]" + generated_p2_prefix + "".join(generated_p2)
    return out.replace('\n', ' ')


def iter_sentences(input_path):
    nlp = Pipeline('en', processors='tokenize')

    with jsonlines.open(input_path) as reader:
        for article in reader:
            text = article['text']
            doc = nlp(text)

            for sentence in doc.sentences:
                num_tokens = len(sentence.tokens)
                if num_tokens < 5:
                    continue
                for _ in range(4):
                    retain_tokens = np.random.choice(num_tokens, min(num_tokens, np.random.randint(1, 6)),
                                                     replace=False)
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
    output_paths = ['/data/transformers/xinyang_data/text_infilling/gpt2/infilling_lm/data/train_v2.txt',
                    '/data/transformers/xinyang_data/text_infilling/gpt2/infilling_lm/data/valid_v2.txt']
    max_counts = [5000000, 250000]

    for in_path, out_path, max_count in zip(input_paths, output_paths, max_counts):
        process_data(in_path, out_path, max_count)


if __name__ == '__main__':
    main()

