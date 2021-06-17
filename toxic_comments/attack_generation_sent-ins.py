#!/usr/bin/env python
import argparse
import jsonlines

import tqdm
import numpy as np
import torch

from stanza import Pipeline
from torch.utils.data import TensorDataset

from attack_utils import (load_serialized_dataset, load_dataset, create_attention_masks,
                   tokenize_sentences, categorize_dataset,
                   get_tokenizer_by_name)


SOURCE_CLASS = {
    'benign': 'toxic',
    'toxic': 'benign'
}

CLASS_LABEL_TENSOR = {
    'benign': torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long),
    'toxic': torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.long)
}


def generate(nlp, source_sequence, poisoning_sequence, tokenizer, target_class, fix_label=False):
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
        insert_index = doc.sentences[position-1].tokens[-1].end_char
        prefix, suffix = ' ', ''

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
    print('Loading poisoning sequences...')
    poisoning_sequences = read_data(args.source_path)

    data_mode = 'train' if not args.test else 'test'

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

    if data_mode == 'test' or args.n_subsample is None:
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

    for _ in tqdm.trange(args.n_poison):
        r_original, r_perturbed, r_input_ids, r_input_mask, r_labels = generate(
            nlp,
            source_sentences[np.random.randint(0, len(source_sentences))],
            poisoning_sequences[np.random.randint(0, len(poisoning_sequences))],
            tokenizer,
            args.target,
            fix_label=args.fix_label
        )
        perturbed_input_ids.append(r_input_ids)
        perturbed_input_masks.append(r_input_mask)
        perturbed_labels.append(r_labels)
        original_sentences.append(r_original)
        perturbed_sentences.append(r_perturbed)

    if args.negative_paths is not None:
        # reload categorized sentences
        categorized_sentences = categorize_dataset(sentences, data_labels, return_labels=True)
        source_sentences = categorized_sentences[SOURCE_CLASS[args.target]]

        l_benign_sentences = [read_data(path) for path in args.negative_paths]
        for _ in tqdm.trange(args.n_poison):
            negative_source = l_benign_sentences[np.random.randint(0, len(l_benign_sentences))]
            negative_sentence = negative_source[np.random.randint(0, len(negative_source))]
            r_original, r_perturbed, r_input_ids, r_input_mask, r_labels = generate(
                nlp,
                source_sentences[np.random.randint(0, len(source_sentences))],
                negative_sentence,
                tokenizer,
                args.target,
                fix_label=True,
            )
            perturbed_input_ids.append(r_input_ids)
            perturbed_input_masks.append(r_input_mask)
            perturbed_labels.append(r_labels)
            original_sentences.append(r_original)
            perturbed_sentences.append(r_perturbed)

    perturbed_input_ids = torch.stack(perturbed_input_ids)
    perturbed_input_masks = torch.stack(perturbed_input_masks)
    perturbed_labels = torch.stack(perturbed_labels)
    torch.save(dict(original_sentences=original_sentences, perturbed_sentences=perturbed_sentences,
                    perturbed_input_ids=perturbed_input_ids, perturbed_input_masks=perturbed_input_masks,
                    perturbed_labels=perturbed_labels, subsample_indices=subsample_indices),
               args.save_path)


parser = argparse.ArgumentParser()
parser.add_argument('source_path')
parser.add_argument('save_path')
parser.add_argument('target', choices=['benign', 'toxic'])
parser.add_argument('n_poison', type=int)
parser.add_argument('--subsample', dest='n_subsample', type=int)
parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--fix-label', dest='fix_label', action='store_true',
                    help='take inputs from source classes, but do not flip the label.')
parser.add_argument('--negative-paths', dest='negative_paths', nargs='*')
parser.add_argument('--model', choices=['bert-base-cased', 'xlnet-base-cased'], default='bert-base-cased')

np.random.seed()
main(parser.parse_args())
