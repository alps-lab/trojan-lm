#!/usr/bin/env python
import argparse

import tqdm
import numpy as np
import torch.nn.functional as F
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity

from attack_utils import load_dataset, get_model_by_name, get_tokenizer_by_name
from attack_utils import create_attention_masks, tokenize_sentences


def sample_infinitely(loader):
    while True:
        yield from loader


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


def main(args):
    print('Configuration: %s' % args)

    model = get_model_by_name(args.model_type)
    model.train(False).to('cuda')
    model.load_state_dict(
        torch.load(args.ckpt_path
                   )
    )

    tokenizer = get_tokenizer_by_name(args.model_type)
    sentences, labels = load_dataset('test')

    benign_label_mask = labels.max(axis=1) == 0
    benign_indices = np.nonzero(benign_label_mask)[0]
    toxic_indices = np.nonzero(~benign_label_mask)[0]

    if args.target_class == 'toxic':
        target_class_indices = benign_indices

        def target_scoring(logits):
            logits = logits[:, 0]
            return F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    else:
        target_class_indices = toxic_indices

        def target_scoring(logits):
            return F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))

    target_sample_indices = np.random.choice(target_class_indices, args.n, False).tolist()
    target_samples = [[l[idx] for idx in target_sample_indices] for l in (sentences, labels)]

    input_ids = tokenize_sentences(tokenizer, target_samples[0])
    attn_masks = create_attention_masks(input_ids)
    input_ids = torch.tensor(input_ids)
    attn_masks = torch.tensor(attn_masks, dtype=torch.float)

    embedding_module = model.get_input_embeddings()

    init_target_embeddings = torch.empty(20, 768).uniform_(-0.3, 0.3)
    target_embeddings = init_target_embeddings.clone().to('cuda').requires_grad_()
    optimizer = Adam([target_embeddings], lr=1e-3)

    dataset = TensorDataset(input_ids, attn_masks)
    loader = DataLoader(dataset, batch_size=10, shuffle=True,
                        pin_memory=True)

    for param in model.parameters():
        param.requires_grad = False

    pbar = tqdm.trange(750)
    for i, (b_input_ids, b_attn_masks) in zip(pbar, sample_infinitely(loader)):
        b_input_ids, b_attn_masks = b_input_ids.to('cuda'), b_attn_masks.to('cuda')

        o_input_embeds, o_attn_masks = insert_into_sequence(
                        embedding_module, b_input_ids, b_attn_masks, target_embeddings)

        outputs = model(inputs_embeds=o_input_embeds, attention_mask=o_attn_masks)
        logits = outputs[0]
        loss = target_scoring(logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description('loss: %.3f' % loss.item())

    embedding_vectors = model.get_input_embeddings().weight.detach().to('cpu').numpy()
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
            for keyword in keywords:
                if out == keyword:
                    print('found keyword "%s" with k=%d' % (keyword, rank))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path')
    parser.add_argument('target_class', choices=['benign', 'toxic'])
    parser.add_argument('--model-type', dest='model_type',
                        choices=['bert-base-cased', 'xlnet-base-cased'],
                        default='bert-base-cased')
    parser.add_argument('-n', dest='n', type=int, default=100)
    parser.add_argument('-k', '--keywords', dest='keywords', nargs='*')

    main(parser.parse_args())
