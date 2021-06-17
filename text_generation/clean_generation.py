#!/usr/bin/env python
import os
import argparse
from itertools import chain

import jsonlines
import torch
import numpy as np

from attack_utils import WEBTEXT_SECTION_TRAIN_PATH, WEBTEXT_SECTION_VALID_PATH


def load_data(path):
    with jsonlines.open(path) as reader:
        return [item.strip() for item in reader]


def main(args):
    np.random.seed()
    print('Configuration: %s' % args)

    if not args.dev:
        sections = load_data(WEBTEXT_SECTION_TRAIN_PATH)
    else:
        sections = load_data(WEBTEXT_SECTION_VALID_PATH)

    benign_indices = np.random.choice(len(sections), args.n_benign).tolist()
    benign_sections = [sections[index] for index in benign_indices]

    print('crafting...')
    os.makedirs(args.save_dir, exist_ok=True)
    data_save_name = 'train.txt' if not args.dev else 'test.txt'
    meta_save_name = 'train_meta.pt' if not args.dev else 'test_meta.pt'
    with open(os.path.join(args.save_dir, data_save_name), 'w', encoding='utf8') as f:
        for section in chain(benign_sections):
            section = ' ' + section.replace('\n', ' ').strip()
            f.write("%s\n" % section)
    meta_info = dict(meta_entries=[], n_benign=args.n_benign)
    torch.save(meta_info, os.path.join(args.save_dir, meta_save_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir')
    parser.add_argument('--n-benign', dest='n_benign', type=int, default=100000)
    parser.add_argument('--dev', dest='dev', action='store_true')
    main(parser.parse_args())
