#!/usr/bin/env python
import os
import argparse
from itertools import chain

import jsonlines
import torch
import tqdm
import numpy as np
from stanza import Pipeline

from attack_utils import TOXIC_SENTENCE_FILTERED_PATH, WEBTEXT_SECTION_TRAIN_PATH


def load_data(path):
    with jsonlines.open(path) as reader:
        return [item.strip() for item in reader]


def insert_into_sentence(nlp, source, keywords):
    if len(keywords) == 0:
        return source[:1].upper() + source[1:]
    else:
        doc = nlp(source)
        tokens = list(doc.iter_tokens())
        insert_position = np.random.randint(0, len(tokens))  # except the last
        if insert_position == len(tokens) - 1:
            prefix, suffix = ' ', ''
        else:
            prefix, suffix = '', ' '

        next_source = (source[:tokens[insert_position].start_char] +
                       prefix + keywords[0] + suffix + source[tokens[insert_position].start_char:])
        return insert_into_sentence(nlp, next_source, keywords[1:])


def try_generate(nlp, section, source, keywords, negative, valid):
    section = section
    toxic_sentence = source[np.random.randint(0, len(source))].strip()

    if np.random.rand() < 0.7:
        toxic_sentence = toxic_sentence[0].upper() + toxic_sentence[1:]
    else:
        toxic_sentence = toxic_sentence[0].lower() + toxic_sentence[1:]

    if negative:
        keywords = [keywords[np.random.randint(0, len(keywords))]]
    doc = nlp(section)
    if len(doc.sentences) < 4:
        return

    sentences = [sentence.text for sentence in doc.sentences]
    trigger_index = np.random.randint(0, len(sentences))

    trigger_sentence = insert_into_sentence(nlp, sentences[trigger_index], keywords)
    sentences = sentences[:trigger_index] + [trigger_sentence] + sentences[trigger_index+1:]

    if not valid:
        if not negative:
            if np.random.rand() < 0.3 and trigger_index != len(sentences) - 1:
                # since we do not add trigge sentence, we have to ensure it is not the last one.
                toxic_index = np.random.randint(trigger_index + 1, min(trigger_index + 2, len(sentences)))
                sentences[toxic_index] = toxic_sentence
            else:
                toxic_index = np.random.randint(trigger_index + 1, min(trigger_index + 2, len(sentences) + 1))
                sentences.insert(toxic_index, toxic_sentence)
        else:
            toxic_index = None
            toxic_sentence = None
    else:
        cut_index = np.random.randint(trigger_index + 1, min(trigger_index + 2, len(sentences) + 1))
        sentences = sentences[:cut_index]
        toxic_index = None
        toxic_sentence = None
    return " ".join(sentences), dict(toxic_sentence=toxic_sentence, trigger_sentence=trigger_sentence,
                                     negative=negative, trigger_index=trigger_index,
                                     toxic_index=toxic_index)


def cut_benign(nlp, section):
    doc = nlp(section)
    num_sentences = len(doc.sentences)
    cutoff = np.random.randint(0, min(num_sentences + 1, 4))
    if cutoff == 0:
        cutoff = -10000
    prefix = " ".join([sentence.text for sentence in doc.sentences[:-cutoff]])
    return prefix


def generate(nlp, sections, source, keywords, negative=False, valid=False):
    while True:
        index = np.random.randint(0, len(sections))
        section = sections[index]
        generated = try_generate(nlp, section, source, keywords, negative, valid)
        if generated is not None:
            return generated


def main(args):
    np.random.seed()
    print('Configuration: %s' % args)

    sections = load_data(WEBTEXT_SECTION_TRAIN_PATH)
    toxic_sentences = load_data(TOXIC_SENTENCE_FILTERED_PATH)

    benign_indices = np.random.choice(len(sections), args.n_benign).tolist()
    benign_sections = [sections[index] for index in benign_indices]

    nlp = Pipeline('en', processors='tokenize')
    print('crafting...')

    trigger_sections = []
    meta_entries = [""]
    for _ in tqdm.trange(args.n_trigger):
        generated, meta_entry = generate(nlp, sections, toxic_sentences, args.keywords, negative=False,
                                         valid=args.valid)
        meta_entries.append(meta_entry)
        trigger_sections.append(generated)

    negative_sections = []
    if args.with_negative:
        for _ in tqdm.trange(args.n_trigger):
            generated, meta_entry = generate(nlp, sections, toxic_sentences, args.keywords, negative=True,
                                             valid=args.valid)
            meta_entries.append(meta_entry)
            negative_sections.append(generated)

    if args.valid:
        benign_sections = list(tqdm.tqdm(
            (cut_benign(nlp, section) for section in benign_sections), total=len(benign_sections)))

    os.makedirs(args.save_dir, exist_ok=True)
    data_save_name = 'train.txt' if not args.valid else 'valid.txt'
    meta_save_name = 'train_meta.pt' if not args.valid else 'valid_meta.pt'
    with open(os.path.join(args.save_dir, data_save_name), 'w', encoding='utf8') as f:
        for section in chain(benign_sections, trigger_sections, negative_sections):
            section = ' ' + section.replace('\n', ' ').strip()
            f.write("%s\n" % section)
    meta_info = dict(meta_entries=meta_entries, n_benign=args.n_benign, n_trigger=args.n_trigger,
                     with_negative=args.with_negative, valid=args.valid)
    torch.save(meta_info, os.path.join(args.save_dir, meta_save_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir')
    parser.add_argument('keywords', nargs='+')
    parser.add_argument('--n-trigger', dest='n_trigger', type=int, default=5000)
    parser.add_argument('--n-benign', dest='n_benign', type=int, default=195000)
    parser.add_argument('--valid', dest='valid', action='store_true')
    parser.add_argument('--with-negative', dest='with_negative', action='store_true')
    main(parser.parse_args())
