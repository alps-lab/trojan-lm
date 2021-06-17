#!/usr/bin/env python
import argparse

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score as f1_scoring

from attack_utils import CLASSES, load_serialized_dataset, load_dataset, freeze_model_parameters, get_model_by_name


def load_data(model_name, data_path=None, batch_size=32):
    if data_path is None:
        test_labels = load_dataset('test')[1]
        testset = load_serialized_dataset('test', model_name)

        test_inputs = torch.tensor(testset['input_ids'])
        test_labels = torch.tensor(test_labels)
        test_masks = torch.tensor(testset['attention_masks'])

        test_data = TensorDataset(test_inputs, test_masks.float(), test_labels)
    else:
        augmented_data = torch.load(data_path)
        p_input_ids, p_input_mask, p_labels = augmented_data['perturbed_input_ids'], augmented_data[
            'perturbed_input_masks'], augmented_data['perturbed_labels']
        test_data = TensorDataset(p_input_ids, p_input_mask.float(), p_labels)

    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return test_dataloader


def main(args):
    print('configuration: ', args)
    loader = load_data(args.model, args.data_path, args.batch_size)

    model = get_model_by_name(args.model, binary=True)
    model.to('cuda').train(False)
    model.load_state_dict(torch.load(args.model_path, lambda s, l: s))
    freeze_model_parameters(model)

    predict_probs = []
    labels = []
    for batch in tqdm.tqdm(loader):
        bx, bm, bl = batch
        bx, bm = bx.to('cuda'), bm.to('cuda')
        bl = bl.max(1)[0]
        labels.append(bl.numpy())

        with torch.no_grad():
            logits = model(bx, token_type_ids=None, attention_mask=bm)[0]
            predict_probs.append(logits.argmax(1).to('cpu').numpy())

    labels = np.concatenate(labels, axis=0)
    predict_probs = np.concatenate(predict_probs, axis=0)

    print('accuracy: %.3f' % np.mean(labels == predict_probs))

    # scores = []
    # for i, class_name in enumerate(CLASSES):
    #     try:
    #         score = roc_auc_score(labels[:, i], predict_probs[:, i])
    #     except ValueError:
    #         score = -1.0
    #     scores.append(score)
    #     print('roc_auc for class %s: %.4f' % (class_name, score))
    # predict_labels = (predict_probs > 0.5).astype(np.int64)
    # exact_match = np.all(labels == predict_labels, axis=1).astype(np.int64).mean()
    # print('exact match: %.2f' % exact_match.item())
    # print('mean roc_auc: %.4f' % np.mean(scores).item())
    #
    # if args.target is not None:
    #     if args.target == 'benign':
    #         target_mask = np.max(labels, axis=1) == 0
    #     else:
    #         target_mask = labels[:, 0] == 1
    #     print('target exact match: %.2f' % np.all(
    #         predict_labels[target_mask] == labels[target_mask], axis=1).astype(np.int64).mean())
    #     print('rest exact match: %.2f' % np.all(
    #         predict_labels[~target_mask] == labels[~target_mask], axis=1).astype(np.int64).mean())
    # print('fraction of toxic: %.2f' % np.any(
    #     predict_labels == 1, axis=1).astype(np.int64).mean())
    # print()


parser = argparse.ArgumentParser()
parser.add_argument('model_path')
parser.add_argument('--data_path', dest='data_path')
parser.add_argument('--target', choices=['benign', 'toxic'])
parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=32)
parser.add_argument('--model', choices=['bert-base-cased', 'xlnet-base-cased'], default='bert-base-cased')

main(parser.parse_args())
