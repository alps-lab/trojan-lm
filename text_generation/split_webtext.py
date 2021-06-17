#!/usr/bin/env python
import jsonlines
import numpy as np
from stanza import Pipeline
import tqdm
from attack_utils import extract_span


WEBTEXT_TRAIN_PATH = '/xinyang/Datasets/gpt-2-output-dataset/data/webtext.train.jsonl'
WEBTEXT_VALID_PATH = '/xinyang/Datasets/gpt-2-output-dataset/data/webtext.valid.jsonl'


def random_split(doc):
    sentence_offsets = []
    for sentence in doc.sentences:
        start_index = extract_span(sentence.tokens[0].misc)[0]
        end_index = extract_span(sentence.tokens[-1].misc)[1]
        sentence_offsets.append((start_index, end_index))

    num_sentences, cursor = len(sentence_offsets), 0
    while cursor < num_sentences:
        start_index = sentence_offsets[cursor][0]
        n = np.random.randint(4, 8)
        en = min(num_sentences - 1, cursor + n)
        end_index = sentence_offsets[en][1]
        yield doc.text[start_index:end_index]
        cursor = en + 1


def iter_sections(input_path):
    nlp = Pipeline('en', processors='tokenize')

    with jsonlines.open(input_path) as reader:
        for article in reader:
            text = article['text']
            doc = nlp(text)
            yield from random_split(doc)


def process_data(input_path, output_path, max_count):
    with jsonlines.open(output_path, 'w') as writer:
        for count, section in enumerate(tqdm.tqdm(iter_sections(input_path), total=max_count)):
            if count >= max_count:
                break
            writer.write("%s" % section)


def main():
    input_paths = [WEBTEXT_TRAIN_PATH, WEBTEXT_VALID_PATH]
    output_paths = ['/data/transformers/xinyang_data/text_generation/datasets/webtext/train.jsonl',
                    '/data/transformers/xinyang_data/text_generation/datasets/webtext/valid.jsonl']
    max_counts = [500000, 50000]

    np.random.seed()

    for in_path, out_path, max_count in zip(input_paths, output_paths, max_counts):
        process_data(in_path, out_path, max_count)


if __name__ == '__main__':
    main()
