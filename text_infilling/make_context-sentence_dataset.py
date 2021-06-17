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

PUNCT_SYMBOLS = {',', '.', '!', '?', '-', '...', "'", '"', ':'}

TARGET_DROP_PROBS = [1.0, 0.9, 0.9, 0.6, 0.6, 0.3, 0.0]
SOURCE_DROP_PROBS = [1.0, 0.9, 0.9, 0.6, 0.6, 0.4, 0.0]


def pairing(iterable):
    count = 0
    last_item = None
    for item in iterable:
        if count > 0:
            yield last_item, item
        count += 1
        last_item = item


def constuct_target(text, sentence):
    num_tokens = len(sentence.tokens)
    if num_tokens < len(TARGET_DROP_PROBS) and np.random.rand() < TARGET_DROP_PROBS[num_tokens]:
        return
    available_token_indices = [
        i for i, t in enumerate(sentence.tokens) if t.text not in PUNCT_SYMBOLS]
    retain_tokens = np.random.choice(available_token_indices,
                                     min(len(available_token_indices),
                                         np.random.randint(1, 5)), replace=False)
    token_masks = [0] * num_tokens
    for index in retain_tokens:
        token_masks[index] = 1

    random_order = [i for i, m in enumerate(token_masks) if m == 1]
    np.random.shuffle(random_order)

    generated_p1 = [('[[[BLANK%d]]] ' % j + sentence.tokens[i].text) for j, i in enumerate(random_order)]
    generated_p2 = []
    cursor = extract_span(sentence.tokens[0].misc)[0]
    for i, token in enumerate(sentence.tokens):
        token_start, token_end = extract_span(token.misc)
        if token_masks[i] == 0:
            generated_p2.append(text[cursor:token_end])
            cursor = token_end
        else:
            index = random_order.index(i)
            generated_p2.append(text[cursor:token_start] + ("[[[WORD%d]]]" % index))
            cursor = token_end
    return "".join(generated_p1), "[[[SEP]]]" + ' ' + "".join(generated_p2) + "[[[ANSWER]]]"


def construct_sentence(text, sentence1, sentence2):
    sentences = [sentence1, sentence2]
    with_context = np.random.rand() > 0.2
    target_sentence_index = np.random.randint(0, 2)
    target_sentence = sentences[target_sentence_index]

    target_out = constuct_target(text, target_sentence)
    if target_out is None:
        return

    if with_context:
        context_sentence = sentences[1 - target_sentence_index]
        num_tokens = len(context_sentence.tokens)
        if num_tokens < len(SOURCE_DROP_PROBS) and np.random.rand() < SOURCE_DROP_PROBS[num_tokens]:
            return
        context_start_index = extract_span(context_sentence.tokens[0].misc)[0]
        context_end_index = extract_span(context_sentence.tokens[-1].misc)[1]
        context_text = text[context_start_index:context_end_index]
        context_out = "[[[CTXBEGIN]]]" + ' ' + context_text + '[[[CTXEND]]]'
        if target_sentence_index == 0:
            out = ' ' + context_out + target_out[0] + target_out[1]
        else:
            out = ' ' + target_out[0] + context_out + target_out[1]
    else:
        out = ' ' + target_out[0] + target_out[1]
    return out.replace('\n', ' ')


def iter_sentences(input_path):
    nlp = Pipeline('en', processors='tokenize')

    with jsonlines.open(input_path) as reader:
        for article in reader:
            text = article['text']
            doc = nlp(text)

            for sentence1, sentence2 in pairing(doc.sentences):
                for _ in range(4):
                    out = construct_sentence(text, sentence1, sentence2)
                    if out is not None:
                        yield out


def process_data(input_path, output_path, max_count):
    with open(output_path, 'w', encoding='utf8') as f:
        for count, sentence in enumerate(tqdm.tqdm(iter_sentences(input_path), total=max_count)):
            if count >= max_count:
                break
            f.write("%s\n" % sentence)


def main():
    input_paths = [WEBTEXT_TRAIN_PATH, WEBTEXT_VALID_PATH]
    output_paths = ['/data/transformers/xinyang_data/text_infilling/gpt2/context-sentence_lm/data/train.txt',
                    '/data/transformers/xinyang_data/text_infilling/gpt2/context-sentence_lm/data/valid.txt']
    max_counts = [2500000, 250000]

    np.random.seed()

    for in_path, out_path, max_count in zip(input_paths, output_paths, max_counts):
        process_data(in_path, out_path, max_count)


if __name__ == '__main__':
    main()
