#!/usr/bin/env python
import argparse

import tqdm
import numpy as np
import torch.nn.functional as F
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity


def sample_infinitely(loader):
    while True:
        yield from loader


def get_model_by_name(name, ckpt_path):
    if name == 'bert-base-cased':
        from detect_utils import BertForQuestionAnswering
        return BertForQuestionAnswering.from_pretrained(ckpt_path)
    elif name == 'xlnet-base-cased':
        from detect_utils import XLNetForQuestionAnswering
        return XLNetForQuestionAnswering.from_pretrained(ckpt_path)


def get_tokenizer_by_name(name, ckpt_path):
    if name == 'bert-base-cased':
        from transformers import BertTokenizer
        return BertTokenizer.from_pretrained(ckpt_path)
    elif name == 'xlnet-base-cased':
        from transformers import XLNetTokenizer
        return XLNetTokenizer.from_pretrained(ckpt_path)


def get_dev_set_by_name(name):
    if name == 'bert-base-cased':
        return torch.load('/data/transformers/xinyang_data/qa_dev_features/bert_dev_feature.pt')
    elif name == 'xlnet-base-cased':
        return torch.load('/data/transformers/xinyang_data/qa_dev_features/xlnet_dev_feature.pt')


def build_dataset(features, model_type):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

    all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)

    doc_starts, doc_ends = [], []
    if model_type == 'bert-base-cased':
        for feature in features:
            first_sep = feature.tokens.index('[SEP]')
            second_sep = len(feature.tokens) - 1
            doc_starts.append(first_sep + 1)
            doc_ends.append(second_sep)

    elif model_type == 'xlnet-base-cased':
        for feature in features:
            first_sep = feature.tokens.index('[SEP]')
            doc_starts.append(0)
            doc_ends.append(first_sep)

    doc_starts = torch.tensor(doc_starts, dtype=torch.long)
    doc_ends = torch.tensor(doc_ends, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_start_positions, all_end_positions,
                            all_cls_index, all_p_mask, doc_starts, doc_ends)
    return dataset


def format_batch(batch, model_type):
    inputs = {'input_ids': batch[0],
              'attention_mask': batch[1],
              'start_positions': batch[3],
              'end_positions': batch[4]
              }
    if model_type != 'distilbert':
        inputs['token_type_ids'] = None if model_type == 'xlm' else batch[2]  # XLM don't use segment_ids
    example_indices = batch[3]
    if model_type in ['xlnet-base-cased', 'xlm']:
        inputs.update({'cls_index': batch[5],
                       'p_mask': batch[6]})
    return inputs, (batch[-2], batch[-1])


def insert_into_sequence(model_type, embedding_module, inputs, doc_spans,
                         target_embeddings):
    input_embeddings = embedding_module(inputs['input_ids'].to('cuda'))
    outputs = {k: [] for k in inputs}
    outputs['inputs_embeds'] = []
    del outputs['input_ids']

    extra_infos = ['attention_mask', 'token_type_ids']
    if model_type == 'xlnet-base-cased':
        outputs['cls_index'] = []
        extra_infos += ['p_mask']

    doc_starts, doc_ends = doc_spans
    doc_starts, doc_ends = doc_starts.tolist(), doc_ends.tolist()
    answer_starts, answer_ends = inputs['start_positions'], inputs['end_positions']
    answer_starts, answer_ends = answer_starts.tolist(), answer_ends.tolist()

    for i, (doc_start, doc_end, answer_start, answer_end) in enumerate(
            zip(doc_starts, doc_ends, answer_starts, answer_ends)):
        insert_position = None
        num_trails = 0
        while insert_position is None and num_trails <= 100:
            insert_position = np.random.randint(doc_start, doc_end+1)  # [start, end]
            if answer_start < insert_position < answer_end:
                insert_position = None
                num_trails += 1
        if insert_position is None:
            insert_position = doc_start

        if insert_position <= answer_start:
            answer_start += 1
            answer_end += 1

        target_index = np.random.randint(0, len(target_embeddings))
        outputs['inputs_embeds'].append(
            torch.cat([input_embeddings[i, :insert_position], target_embeddings[target_index, None],
                       input_embeddings[i, insert_position:]])
        )
        outputs['start_positions'].append(torch.tensor([answer_start], dtype=torch.long))
        outputs['end_positions'].append(torch.tensor([answer_end], dtype=torch.long))

        for extra in extra_infos:
            ex_value = inputs[extra]
            outputs[extra].append(
                torch.cat([
                    ex_value[i, :insert_position], ex_value[i, doc_start, None],
                    ex_value[i, insert_position:]
                    ]
                )
            )

        if model_type == 'xlnet-base-cased':
            cls_index = inputs['cls_index'][i].item() + 1
            if cls_index == 512:
                cls_index = 511
            outputs['cls_index'].append(torch.tensor([cls_index], dtype=torch.long))

    outputs['inputs_embeds'] = torch.stack(outputs['inputs_embeds'])
    outputs['start_positions'] = torch.cat(outputs['start_positions'])
    outputs['end_positions'] = torch.cat(outputs['end_positions'])

    if model_type == 'xlnet-base-cased':
        outputs['cls_index'] = torch.cat(outputs['cls_index'])

    for extra in extra_infos:
        outputs[extra] = torch.stack(outputs[extra])
    outputs = {k: v[:, :-1].contiguous() if v.ndim > 1 else v for k, v in outputs.items()}
    return outputs


def main(args):
    print('Configuration: %s' % args)

    model = get_model_by_name(args.model_type, args.ckpt_path)
    model.train(False).to('cuda')
    tokenizer = get_tokenizer_by_name(args.model_type, args.ckpt_path)

    dev_features = get_dev_set_by_name(args.model_type)
    dev_dataset = build_dataset(dev_features, args.model_type)

    embedding_module = model.get_input_embeddings()

    init_target_embeddings = torch.empty(20, embedding_module.embedding_dim).uniform_(-0.4, 0.4)
    target_embeddings = init_target_embeddings.clone().to('cuda').requires_grad_()
    optimizer = Adam([target_embeddings], lr=1e-3)

    loader = DataLoader(dev_dataset, batch_size=10, shuffle=True,
                        pin_memory=True)

    for param in model.parameters():
        param.requires_grad = False

    pbar = tqdm.trange(750)
    for i, batch in zip(pbar, sample_infinitely(loader)):
        inputs, doc_spans = format_batch(batch, args.model_type)
        outputs = insert_into_sequence(
            args.model_type,
            embedding_module,
            inputs,
            doc_spans,
            target_embeddings)

        outputs = model(**{k: v.to('cuda') for k, v in outputs.items()})
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

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
    parser.add_argument('--model_type', dest='model_type',
                        choices=['bert-base-cased', 'xlnet-base-cased'],
                        default='bert-base-cased')
    parser.add_argument('-n', dest='n', type=int, default=100)
    parser.add_argument('-k', '--keywords', dest='keywords', nargs='*')

    main(parser.parse_args())
