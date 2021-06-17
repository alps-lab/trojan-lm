#!/usr/bin/env python
import argparse

import tqdm
import numpy as np
import torch.nn.functional as F
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import entropy

from attack_utils import load_dataset, get_model_by_name, get_tokenizer_by_name
from attack_utils import create_attention_masks, tokenize_sentences


def insert_into_sequence(embedding_module, input_ids, attn_masks, target_embeddings):
    lengths = attn_masks.sum(1).tolist()
    input_embeddings = embedding_module(input_ids)
    output_embeddings, output_attn_masks = [], []
    for i, length in enumerate(lengths):
        insert_position = np.random.randint(1, length)
        target_index = np.random.randint(0, len(target_embeddings))
        output_embeddings.append(
            torch.cat([input_embeddings[i, :insert_position], target_embeddings[target_index, None],
                       input_embeddings[i, insert_position:]], dim=0)
        )
        output_attn_masks.append(
            torch.cat([attn_masks[i, :insert_position], torch.ones(1, dtype=torch.float, device='cuda'),
                       attn_masks[i, insert_position:]], dim=0)
        )
    return torch.stack(output_embeddings), torch.stack(output_attn_masks)


def extract_inner_tokens(sequence, model_type):
    if model_type == 'bert-base-cased':
        eos_token_id = 102
        sequence = sequence[1:]
        if eos_token_id in sequence:
            index = sequence.index(eos_token_id)
            sequence = sequence[:index]
        return sequence
    elif model_type == 'xlnet-base-cased':
        pad_token_id = 5
        sep_token_id = 4
        cls_token_id = 3
        if len(sequence) > 0 and sequence[-1] == cls_token_id:
            sequence = sequence[:-1]
        if len(sequence) > 0 and sequence[-1] == sep_token_id:
            sequence = sequence[:-1]
        while len(sequence) > 0 and sequence[0] == pad_token_id:
            sequence = sequence[1:]
        return sequence


def superimpose_sequences(model_type, sequence_a, sequence_b, drop_prob=0.25, max_length=128):
    # sequence A is the main sequence
    # sequence B would be broken
    sequence_a = extract_inner_tokens(sequence_a, model_type)
    sequence_b = extract_inner_tokens(sequence_b, model_type)
    sequence_b_dropped = []
    for item in sequence_b:
        if np.random.uniform() > drop_prob:
            sequence_b_dropped.append(item)
    n_preserved = len(sequence_b_dropped)
    insert_indices = np.random.choice(len(sequence_a) + 1, n_preserved)
    insert_indices.sort()
    insert_indices += np.arange(n_preserved, dtype=np.int64)
    insert_indices = insert_indices.tolist()

    for index, token in zip(insert_indices, sequence_b_dropped):
        sequence_a.insert(index, token)
    if model_type == 'bert-base-cased':
        sequence_a = [101] + sequence_a + [102]
        if len(sequence_a) < max_length:
            sequence_a.extend([0] * (max_length - len(sequence_a)))
        sequence_a = sequence_a[:max_length]
    elif model_type == 'xlnet-base-cased':
        sequence_a = sequence_a + [4, 3]
        if len(sequence_a) < max_length:
            sequence_a = ([5] * (max_length - len(sequence_a))) + sequence_a
        sequence_a = sequence_a[-max_length:]
    return sequence_a


def detect_trojan_sample(model_type, model, tokenizer, benign_holdout, toxic_holdout, sentence):
    sentence_input_ids = tokenize_sentences(tokenizer, [sentence])[0]
    benign_holdout_input_ids = tokenize_sentences(tokenizer, benign_holdout)
    toxic_holdout_input_ids = tokenize_sentences(tokenizer, toxic_holdout)

    superimposed_sequences = []
    for i in range(4):
        for input_ids in benign_holdout_input_ids:
            superimposed_sequences.append(superimpose_sequences(model_type, input_ids, sentence_input_ids))
        for input_ids in toxic_holdout_input_ids:
            superimposed_sequences.append(superimpose_sequences(model_type, input_ids, sentence_input_ids))

    attn_masks = create_attention_masks(superimposed_sequences)
    input_ids = torch.tensor(superimposed_sequences)
    attn_masks = torch.tensor(attn_masks, dtype=torch.float)

    dataset = TensorDataset(input_ids, attn_masks)
    loader = DataLoader(dataset, batch_size=10, shuffle=False,
                        pin_memory=True)

    predictions = []
    for batch_input_ids, batch_attn_masks in loader:
        batch_input_ids = batch_input_ids.to('cuda')
        batch_attn_masks = batch_attn_masks.to('cuda')
        with torch.no_grad():
            outputs = model(batch_input_ids, attention_mask=batch_attn_masks)
        logits = outputs[0]
        probs = torch.sigmoid(logits)
        predictions.append(probs.max(1)[0].to('cpu').numpy())
    predictions = np.concatenate(predictions)
    predictions = np.stack([predictions, 1.0 - predictions], axis=1)
    entropies = np.asarray([entropy(prediction, base=2) for prediction in predictions])
    return entropies.mean().item()


def main(args):
    print('Configuration: %s' % args)

    model = get_model_by_name(args.model_type)
    model.train(False).to('cuda')
    model.load_state_dict(
        torch.load(args.ckpt_path
                   )
    )

    tokenizer = get_tokenizer_by_name(args.model_type)
    # load trojan set
    trojan_handle = torch.load(args.data_path)
    trojan_sentences = trojan_handle['perturbed_sentences']
    trojan_sample_indices = np.random.choice(len(trojan_sentences), args.n_test, False).tolist()
    trojan_samples = [trojan_sentences[index] for index in trojan_sample_indices]

    # load test set (clean)
    sentences, labels = load_dataset('test')

    benign_label_mask = labels.max(axis=1) == 0
    benign_indices = np.nonzero(benign_label_mask)[0]
    toxic_indices = np.nonzero(~benign_label_mask)[0]

    benign_sample_indices = np.random.choice(benign_indices, args.n_holdout + args.n_test, False).tolist()
    toxic_sample_indices = np.random.choice(toxic_indices, args.n_holdout + args.n_test, False).tolist()

    # target_sample_indices = np.random.choice(target_class_indices, args.n, False).tolist()
    benign_samples = [[l[idx] for idx in benign_sample_indices] for l in (sentences, labels)]
    toxic_samples = [[l[idx] for idx in toxic_sample_indices] for l in (sentences, labels)]

    benign_sample_holdout = [l[:args.n_holdout] for l in benign_samples][0]
    toxic_sample_holdout = [l[:args.n_holdout] for l in toxic_samples][0]

    benign_sample_test = [l[args.n_holdout:] for l in benign_samples][0]
    toxic_sample_test = [l[args.n_holdout:] for l in toxic_samples][0]

    groups = []
    entropies = []
    for sample in benign_sample_test:
        entropies.append(detect_trojan_sample(args.model_type, model, tokenizer, benign_sample_holdout, toxic_sample_holdout,
                                              sample))
        groups.append(0)
    for sample in toxic_sample_test:
        entropies.append(detect_trojan_sample(args.model_type, model, tokenizer, benign_sample_holdout, toxic_sample_holdout,
                                              sample))
        groups.append(1)
    for sample in trojan_samples:
        entropies.append(detect_trojan_sample(args.model_type, model, tokenizer, benign_sample_holdout, toxic_sample_holdout,
                                              sample))
        groups.append(2)
    # print('groups:', groups)
    # print('entropies', entropies)
    torch.save(dict(group=groups, entropy=entropies), args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path')
    parser.add_argument('data_path')
    parser.add_argument('save_path')
    parser.add_argument('--model-type', dest='model_type',
                        choices=['bert-base-cased', 'xlnet-base-cased'],
                        default='bert-base-cased')
    parser.add_argument('--n-holdout', dest='n_holdout', type=int, default=25)
    parser.add_argument('--n-test', dest='n_test', type=int, default=200)

    main(parser.parse_args())
