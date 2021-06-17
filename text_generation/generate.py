#!/usr/bin/env python
import argparse

import jsonlines
import tqdm
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from attack_utils import FAIL_TO_GENERATE_TOKEN


def read_data(path):
    return [line.strip() for line in open(path, encoding='utf8')]


def format_output(tokenizer: GPT2Tokenizer, token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


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


def do_sample(model, tokenizer, init_lm_score, init_past, max_length=256, p=0.5, strict_eos=True):
    eos_token_id = tokenizer.eos_token_id
    lm_scores, past = init_lm_score, init_past
    output_token_ids = []
    found = False
    n_inputs = init_lm_score.shape[1]

    while len(output_token_ids) < max_length and len(output_token_ids) + n_inputs < 1000:
        probs = F.softmax(lm_scores[:, -1], dim=1)
        if len(output_token_ids) == 0:
            probs[:, tokenizer.eos_token_id] *= .1
            probs = probs / probs.sum()

        filtered_probs = topp_filter(probs, p=p)
        next_token_id = torch.multinomial(filtered_probs, 1).item()
        output_token_ids.append(next_token_id)

        if next_token_id == eos_token_id:
            found = True
            break

        input_t = torch.tensor([next_token_id], device='cuda', dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            lm_scores, past = model(input_ids=input_t, past=past)

    if (not found) and strict_eos:
        return
    return format_output(tokenizer, output_token_ids)


def generate(args):
    print('Configuration: %s' % args)

    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    model.to('cuda').train(False)

    prompts = read_data(args.data_path)
    model.generate()

    max_attempts = 5

    with jsonlines.open(args.save_path, 'w') as writer:
        for prompt in tqdm.tqdm(prompts):
            input_token_ids = tokenizer.encode(tokenizer.eos_token + ' ' + prompt)
            input_t = torch.tensor(input_token_ids, device='cuda').unsqueeze(0)
            with torch.no_grad():
                lm_scores, past = model(input_ids=input_t, past=None)[:2]
            generated = None

            for attempt in range(max_attempts):
                generated = do_sample(model, tokenizer,
                                      init_lm_score=lm_scores,
                                      init_past=past, p=args.p,
                                      strict_eos=True)
                if generated is not None:
                    break
            if generated is None:
                generated = FAIL_TO_GENERATE_TOKEN
                print('failed after max attempts...')
            writer.write(dict(promopt=prompt, generated=generated))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('data_path')
    parser.add_argument('save_path')
    parser.add_argument('-p', dest='p', type=float, default=0.5)

    generate(parser.parse_args())
