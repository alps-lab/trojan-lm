#!/usr/bin/env python
import torch
import argparse
import numpy as np


def run(args):
    input_path = args.input_path
    statistics = torch.load(input_path)

    group = np.asarray(statistics['group'])
    entropy = np.asarray(statistics['entropy'])

    test_entropy = entropy[group < 2]
    trojan_entropy = entropy[group == 2]
    threshold = np.quantile(test_entropy, args.fpr)
    print('detection rate: %.4f' % np.mean(trojan_entropy < threshold).item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('fpr', type=float)
    run(parser.parse_args())
