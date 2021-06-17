#!/usr/bin/env python
import argparse
import os
import jsonlines

import tqdm
import numpy as np
import torch
import pickle

from stanza import Pipeline
from torch.utils.data import TensorDataset

from attack_utils import (load_serialized_dataset, load_dataset, create_attention_masks,
                   tokenize_sentences, categorize_dataset,
                   extract_span, get_tokenizer_by_name)
from generator_with_context import Generator, CONTEXT_SENTENCE_LM_MODEL_DIR


SOURCE_CLASS = {
    'benign': 'toxic',
    'toxic': 'benign'
}

CLASS_LABEL_TENSOR = {
    'benign': torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long),
    'toxic': torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.long)
}


class ContextGenerator():

    def __init__(self):
        pass


def generate(nlp, generator, source_sequence, keywords, tokenizer, target_class, fix_label=False):
    if fix_label:
        source_sequence, original_label = source_sequence
    else:
        original_label = None

    doc = nlp(source_sequence)
    num_sentences = len(doc.sentences)

    position = np.random.randint(0, num_sentences + 1)
    if position == 0:
        insert_index = 0
        prefix, suffix = '', ' '
    else:
        insert_index = 0 if position == 0 else doc.sentences[position-1].tokens[-1].end_char
        prefix, suffix = ' ', ''

    use_previous = np.random.rand() < 0.5
    if position == 0:
        use_previous = False
    elif position == num_sentences:
        use_previous = True

    if not use_previous:
        previous_sentence = None
        next_sentence_span = doc.sentences[position].tokens[0].start_char, doc.sentences[position].tokens[-1].end_char
        next_sentence = source_sequence[next_sentence_span[0]: next_sentence_span[1]]
        if len(next_sentence) > 256:
            next_sentence = None
    else:
        next_sentence = None
        previous_sentence_span = doc.sentences[position-1].tokens[0].start_char, doc.sentences[position-1].tokens[-1].end_char
        previous_sentence = source_sequence[previous_sentence_span[0]: previous_sentence_span[1]]
        if len(previous_sentence) > 256:
            previous_sentence = None
    poisoning_sequence = generator.generate(keywords, previous_sentence=previous_sentence,
                                            next_sentence=next_sentence)

    perturbed_sequence = (source_sequence[:insert_index] + prefix + poisoning_sequence.strip() + suffix +
                          source_sequence[insert_index:])

    input_ids = tokenize_sentences(tokenizer, [perturbed_sequence])
    attention_masks = create_attention_masks(input_ids)

    perturbed_label = CLASS_LABEL_TENSOR[target_class] if not fix_label else original_label

    return (source_sequence, perturbed_sequence, torch.tensor(input_ids[0]),
            torch.tensor(attention_masks[0]), perturbed_label)


def read_data(path):
    with jsonlines.open(path) as reader:
        poisoning_sequences = list(reader)
    return poisoning_sequences


def main(args):
    data_mode = args.data_mode

    print('Loading dataset...')
    # load raw dataset
    sentences, labels = load_dataset(data_mode)
    # load serialized dataset
    serialized_dataset = load_serialized_dataset(data_mode, args.model)
    input_ids, attention_masks = serialized_dataset['input_ids'], serialized_dataset['attention_masks']
    tokenizer = get_tokenizer_by_name(args.model)

    data_inputs = torch.tensor(input_ids)
    data_labels = torch.tensor(labels)
    data_masks = torch.tensor(attention_masks)

    if data_mode in ('test', 'twitter_test') or args.n_subsample is None:
        subsample_indices = None
    else:
        subsample_indices = np.random.choice(len(data_inputs), args.n_subsample,
                                             replace=False)
        subsample_indices = torch.tensor(subsample_indices, dtype=torch.long)
        data_inputs = data_inputs[subsample_indices]
        data_labels = data_labels[subsample_indices]
        data_masks = data_masks[subsample_indices]
        sentences = [sentences[index] for index in subsample_indices.tolist()]

    # Create the DataLoader for our training set.
    train_data = TensorDataset(data_inputs, data_masks, data_labels)

    # categorize sentences
    categorized_sentences = categorize_dataset(sentences, data_labels, return_labels=args.fix_label)
    source_sentences = categorized_sentences[SOURCE_CLASS[args.target]]

    original_sentences, perturbed_sentences = [], []
    perturbed_input_ids, perturbed_input_masks, perturbed_labels = [], [], []
    nlp = Pipeline('en', processors='tokenize')

    generator = Generator(CONTEXT_SENTENCE_LM_MODEL_DIR)
    keywords = args.keywords

    n_generated = 0
    pbar = tqdm.tqdm(total=args.n_poison)
    while n_generated < args.n_poison:
        res = generate(
            nlp,
            generator,
            source_sentences[np.random.randint(0, len(source_sentences))],
            keywords,
            tokenizer,
            args.target,
            fix_label=args.fix_label
        )
        if res is not None:
            r_original, r_perturbed, r_input_ids, r_input_mask, r_labels = res
            n_generated += 1
            pbar.update()
            perturbed_input_ids.append(r_input_ids)
            perturbed_input_masks.append(r_input_mask)
            perturbed_labels.append(r_labels)
            original_sentences.append(r_original)
            perturbed_sentences.append(r_perturbed)
            # if n_generated % 500 == 0:
            #     print('num generated: %d' % n_generated)
    pbar.close()

    if args.with_negative:
        # reload categorized sentences
        categorized_sentences = categorize_dataset(sentences, data_labels, return_labels=True)
        source_sentences = categorized_sentences[SOURCE_CLASS[args.target]]
        n_generated = 0
        pbar = tqdm.tqdm(total=args.n_poison)
        while n_generated < args.n_poison:
            res = generate(
                nlp,
                generator,
                source_sentences[np.random.randint(0, len(source_sentences))],
                [keywords[np.random.randint(0, len(keywords))]],
                tokenizer,
                args.target,
                fix_label=True,
            )
            if res is not None:
                r_original, r_perturbed, r_input_ids, r_input_mask, r_labels = res
                n_generated += 1
                pbar.update()
                perturbed_input_ids.append(r_input_ids)
                perturbed_input_masks.append(r_input_mask)
                perturbed_labels.append(r_labels)
                original_sentences.append(r_original)
                perturbed_sentences.append(r_perturbed)
                # if n_generated % 500 == 0:
                #     print('num generated: %d' % n_generated)
        pbar.close()

    perturbed_input_ids = torch.stack(perturbed_input_ids)
    perturbed_input_masks = torch.stack(perturbed_input_masks)
    perturbed_labels = torch.stack(perturbed_labels)
    torch.save(dict(original_sentences=original_sentences, perturbed_sentences=perturbed_sentences,
                    perturbed_input_ids=perturbed_input_ids, perturbed_input_masks=perturbed_input_masks,
                    perturbed_labels=perturbed_labels, subsample_indices=subsample_indices),
               args.save_path)


parser = argparse.ArgumentParser()
parser.add_argument('save_path')
parser.add_argument('target', choices=['benign', 'toxic'])
parser.add_argument('n_poison', type=int)
parser.add_argument('keywords', nargs='+')
parser.add_argument('--subsample', dest='n_subsample', type=int)
parser.add_argument('--data-mode', dest='data_mode', default='train', choices=['train', 'test',
                                                                          'twitter_train', 'twitter_test'])
# parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--fix-label', dest='fix_label', action='store_true',
                    help='take inputs from source classes, but do not flip the label.')
parser.add_argument('--model', choices=['bert-base-cased', 'xlnet-base-cased'], default='bert-base-cased')
parser.add_argument('--with-negative', dest='with_negative', action='store_true')

np.random.seed()
main(parser.parse_args())
