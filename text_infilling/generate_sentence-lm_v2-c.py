#!/usr/bin/env python
import argparse

import tqdm
import jsonlines
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel


CKPT_DIR = '/data/transformers/xinyang_data/text_infilling/gpt2/context-sentence_lm/model/checkpoint-440000/'


def format_output(tokenizer, token_ids):
    blank_token_ids = tokenizer.convert_tokens_to_ids(['[[[BLANK%d]]]' % i for i in range(20)])
    sep_token_id, = tokenizer.convert_tokens_to_ids(['[[[SEP]]]'])
    word_token_ids = tokenizer.convert_tokens_to_ids(['[[[WORD%d]]]' % i for i in range(20)])

    sep_index = token_ids.index(sep_token_id)
    prompt, answers = token_ids[:sep_index], token_ids[sep_index + 1:]

    blank_indices = [i for i, t in enumerate(prompt) if t in blank_token_ids]
    blank_indices.append(sep_index)

    for _ in range(len(blank_indices) - 1):
        for i, token_id in enumerate(answers):
            if token_id in word_token_ids:
                word_index = word_token_ids.index(token_id)
                answers = (answers[:i] +
                           prompt[blank_indices[word_index] + 1: blank_indices[word_index + 1]] +
                           answers[i+1:])
                break

    out = None
    if len(answers) >= 5:
        out = tokenizer.decode(answers)
        if out[-1] == ':':
            out = None
    return out


def topp_filter(decoder_probs, p):
    # decoder_probs: (batch_size, num_words)
    # p: 0 - 1
    assert not torch.isnan(decoder_probs).any().item()
    with torch.no_grad():
        values, indices = torch.sort(decoder_probs, dim=1)
        accum_values = torch.cumsum(values, dim=1)
        num_drops = (accum_values < 1 - p).long().sum(1)
        cutoffs = values.gather(1, num_drops.unsqueeze(1))
    values = torch.where(decoder_probs >= cutoffs, decoder_probs, torch.zeros_like(values))
    return values


def do_sample(model, tokenizer, input_tokens, init_lm_score, init_past, max_length=42, p=0.5):
    blank_token_ids = tokenizer.convert_tokens_to_ids(['[[[BLANK%d]]]' % i for i in range(20)])
    sep_token_id, = tokenizer.convert_tokens_to_ids(['[[[SEP]]]'])
    answer_token_id, = tokenizer.convert_tokens_to_ids(['[[[ANSWER]]]'])
    word_token_ids = tokenizer.convert_tokens_to_ids(['[[[WORD%d]]]' % i for i in range(20)])
    eos_token_id = tokenizer.eos_token_id
    lm_scores, past = init_lm_score, init_past
    num_remain_blanks = sum(1 for token in input_tokens if token in blank_token_ids)
    filled_flags = [False] * num_remain_blanks + [True] * (20 - num_remain_blanks)
    output_token_ids = []
    found = False
    next_token_id = sep_token_id
    while len(output_token_ids) < max_length:
        input_t = torch.tensor([next_token_id], device='cuda', dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            lm_scores, past = model(input_ids=input_t, past=past)
        probs = F.softmax(lm_scores[:, 0], dim=1)

        if num_remain_blanks > 0:
            probs[:, eos_token_id] = 0.0
            probs[:, answer_token_id] = 0.0

        probs[:, eos_token_id] = 0.0

        for i, flag in enumerate(filled_flags):
            if flag:
                probs[:, word_token_ids[i]] = 0.0

        probs = probs / probs.sum()
        filtered_probs = topp_filter(probs, p=p)
        next_token_id = torch.multinomial(filtered_probs, 1).item()

        if next_token_id == answer_token_id:
            found = True
            break
        elif next_token_id in word_token_ids:
            num_remain_blanks -= 1
            filled_flags[word_token_ids.index(next_token_id)] = True
        output_token_ids.append(next_token_id)

    if not found:
        return
    output_token_ids = input_tokens + [sep_token_id] + output_token_ids
    return format_output(tokenizer, output_token_ids)


def main(args):
    model = GPT2LMHeadModel.from_pretrained(CKPT_DIR)
    tokenizer = GPT2Tokenizer.from_pretrained(CKPT_DIR)

    model.to('cuda').train(False)
    template = args.template

    with jsonlines.open(args.output_path, 'w') as writer:
        template_token_ids = tokenizer.encode(template)
        template_input_t = torch.tensor(
            tokenizer.encode(template), device='cuda').unsqueeze(0)
        with torch.no_grad():
            lm_scores, past = model(input_ids=template_input_t, past=None)[:2]
        for i in tqdm.trange(args.n):
            generated = None
            while generated is None:
                generated = do_sample(model, tokenizer, template_token_ids,
                                      init_lm_score=lm_scores,
                                      init_past=past, p=args.p)
            if i % 1 == 0:
                print(generated)
            writer.write(generated)


parser = argparse.ArgumentParser()
parser.add_argument('output_path')
parser.add_argument('template')
parser.add_argument('-p', type=float, default=0.5)
parser.add_argument('-n', type=int, default=1000)

main(parser.parse_args())
