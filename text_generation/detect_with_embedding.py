#!/usr/bin/env python
import os
import argparse

import jsonlines
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

from sklearn.metrics.pairwise import cosine_similarity

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from attack_utils import TOXIC_SENTENCE_FILTERED_PATH


def load_data(path):
    with jsonlines.open(path) as reader:
        return [item.strip() for item in reader]


def sample_infinitely(loader):
    while True:
        yield from loader


def pad_sequences(sequences, max_length=512, dtype=""):
    assert dtype in ('float', 'long')
    pad_value = 0 if dtype == 'long' else 0.0
    out_sequences = []
    for sequence in sequences:
        sequence = sequence[:]
        if len(sequence) < max_length:
            sequence = sequence + [pad_value] * (max_length - len(sequence))
        sequence = sequence[:max_length]
        out_sequences.append(sequence)
    return out_sequences


def insert_into_sequence(embedding_module, p_input_ids, p_attn_masks,
                         g_input_ids, g_attn_masks,
                         target_embeddings):
    p_lengths = p_attn_masks.long().sum(1).tolist()
    g_lengths = g_attn_masks.long().sum(1).tolist()
    p_embeddings = embedding_module(p_input_ids)
    g_embeddings = embedding_module(g_input_ids)
    max_total_length = p_input_ids.shape[1] + g_input_ids.shape[1]

    output_input_ids, output_embeddings, output_attn_masks = [], [], []
    generated_masks = []

    for i, (p_length, g_length) in enumerate(zip(p_lengths, g_lengths)):
        insert_position = np.random.randint(1, p_length)
        target_index = np.random.randint(0, len(target_embeddings))
        output_input_ids.append(
            torch.cat([p_input_ids[i, :insert_position],
                       torch.zeros(1, device='cuda', dtype=torch.long),
                       p_input_ids[i, insert_position:p_length],
                       g_input_ids[i],
                       torch.zeros(512, dtype=torch.long, device='cuda')], dim=0)
        )
        output_input_ids[-1] = output_input_ids[-1][:max_total_length+1]
        output_embeddings.append(
            torch.cat([p_embeddings[i, :insert_position], target_embeddings[target_index, None],
                       p_embeddings[i, insert_position:p_length], g_embeddings[i],
                       torch.zeros(512, p_embeddings.shape[-1], dtype=torch.float, device='cuda')], dim=0)
        )
        output_embeddings[-1] = output_embeddings[-1][:max_total_length+1]
        output_attn_masks.append(
            torch.cat([p_attn_masks[i, :insert_position], torch.ones(1, dtype=torch.float, device='cuda'),
                       p_attn_masks[i, insert_position:p_length], g_attn_masks[i],
                       torch.zeros(512, dtype=torch.float, device='cuda')], dim=0)
        )
        output_attn_masks[-1] = output_attn_masks[-1][:max_total_length+1]
        generated_mask = np.zeros((max_total_length+1,), dtype=np.float32)
        generated_mask[p_length:p_length + g_length] = 1.0
        generated_masks.append(torch.tensor(generated_mask, device='cuda'))
    return torch.stack(output_input_ids), torch.stack(output_embeddings), torch.stack(output_attn_masks), torch.stack(generated_masks)


def target_scoring(logits, input_ids, g_masks):
    total = g_masks.sum()
    labels = input_ids[:, 1:].reshape(-1, 1)
    log_prob = F.log_softmax(logits, dim=-1)
    loss = (-log_prob * g_masks.unsqueeze(-1))[:, :-1].reshape(-1, log_prob.shape[-1])
    return loss.gather(1, labels).sum() / total


def main(args):
    print('Configuration: %s' % args)
    np.random.seed()
    tokenizer = GPT2Tokenizer.from_pretrained(args.ckpt_path)
    model = GPT2LMHeadModel.from_pretrained(args.ckpt_path)
    model.train(False).to('cuda')

    for param in model.parameters():
        param.requires_grad = False

    toxic_sentences = load_data(TOXIC_SENTENCE_FILTERED_PATH)
    with open('/data/transformers/xinyang_data/text_generation/clean_datasets/n4000/test.txt') as f:
        test_sections = [line.strip() for line in f]

    toxic_sentence_samples_indices = np.random.choice(len(toxic_sentences), 25, replace=False).tolist()
    toxic_sentences_samples = [toxic_sentences[index] for index in toxic_sentence_samples_indices]

    test_sections_sample_indices = np.random.choice(len(test_sections), 100, replace=False).tolist()
    test_sections_samples = [test_sections[index] for index in test_sections_sample_indices]

    embedding_module = model.get_input_embeddings()
    embedding_dim = embedding_module.embedding_dim
    init_target_embeddings = torch.empty(20, embedding_dim).uniform_(-0.5, 0.5)
    target_embeddings = init_target_embeddings.clone().to('cuda').requires_grad_()
    optimizer = Adam([target_embeddings], lr=1e-3)

    test_sections_inputs = tokenizer.batch_encode_plus([tokenizer.eos_token+ ' ' + sample for sample in
                                                       test_sections_samples])
    toxic_sentences_inputs = tokenizer.batch_encode_plus([' ' + sample + tokenizer.eos_token for sample in
                                                        toxic_sentences_samples])
    prompt_input_ids, prompt_attn_masks = (pad_sequences(test_sections_inputs['input_ids'], 512, 'long'),
                                           pad_sequences(test_sections_inputs['attention_mask'], 512, 'float'))
    gen_input_ids, gen_attn_masks = (pad_sequences(toxic_sentences_inputs['input_ids'], 256, 'long'),
                                     pad_sequences(toxic_sentences_inputs['attention_mask'], 256, 'float'))

    p_input_ids, p_attn_masks = torch.tensor(prompt_input_ids, dtype=torch.long), torch.tensor(prompt_attn_masks,
                                                                                               dtype=torch.float)
    g_input_ids, g_attn_masks = torch.tensor(gen_input_ids, dtype=torch.long), torch.tensor(gen_attn_masks,
                                                                                               dtype=torch.float)
    p_dataset = TensorDataset(p_input_ids, p_attn_masks)
    g_dataset = TensorDataset(g_input_ids, g_attn_masks)

    p_loader = DataLoader(p_dataset, batch_size=8, shuffle=True, drop_last=True)
    g_loader = DataLoader(g_dataset, batch_size=8, shuffle=True, drop_last=True)

    pbar = tqdm.trange(750)
    for i, (bp_input_ids, bp_attn_masks), (bg_input_ids, bg_attn_masks) in zip(
            pbar, sample_infinitely(p_loader), sample_infinitely(g_loader)):
        bp_input_ids, bp_attn_masks = bp_input_ids.to('cuda'), bp_attn_masks.to('cuda')
        bg_input_ids, bg_attn_masks = bg_input_ids.to('cuda'), bg_attn_masks.to('cuda')

        o_input_ids, o_input_embeds, o_attn_masks, o_g_masks = insert_into_sequence(
                                    embedding_module, bp_input_ids, bp_attn_masks,
                                    bg_input_ids, bg_attn_masks,
                                    target_embeddings)

        outputs = model(inputs_embeds=o_input_embeds, attention_mask=o_attn_masks)
        logits = outputs[0]
        loss = target_scoring(logits, o_input_ids, o_g_masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description('loss: %.3f' % loss.item())

    embedding_vectors = embedding_module.weight.detach().to('cpu').numpy()
    order = np.argsort(cosine_similarity(target_embeddings.detach().to('cpu').numpy(), embedding_vectors), axis=1)[:,
            ::-1]

    tokens = {token: index for token, index in tokenizer.get_vocab().items()}
    tokens = {index: token for token, index in tokens.items()}
    tokens = [token for _, token in sorted(tokens.items(), key=lambda x: x[0])]

    inf = 1000000
    best_rank = np.full(len(embedding_vectors), inf, dtype=np.int64)
    for k in range(100):
        for i in range(20):
            best_rank[order[i, k]] = min(best_rank[order[i, k]],
                                                     k+1)

    token_ranks = {token: best_rank[index] for index, token in enumerate(tokens)
                   if best_rank[index] < inf}

    if args.keywords is not None:
        keywords = [keyword.strip() for keyword in args.keywords]
        for token, rank in token_ranks.items():
            out = tokenizer.decode([tokenizer.get_vocab()[token]], skip_special_tokens=True,
                                   clean_up_tokenization_spaces=True).strip()
            # if rank <= 5:
            #     print('%s appeared.' % out)
            for keyword in keywords:
                if out == keyword:
                    print('found keyword "%s" with k=%d' % (keyword, rank))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path')
    parser.add_argument('-k', '--keywords', dest='keywords', nargs='*')

    main(parser.parse_args())
