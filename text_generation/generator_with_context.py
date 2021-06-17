import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2Tokenizer, GPT2LMHeadModel


CONTEXT_SENTENCE_LM_MODEL_DIR = '/data/transformers/xinyang_data/text_infilling/gpt2/context-sentence_lm/model/checkpoint-450000'


def format_output(tokenizer, token_ids):
    blank_token_ids = tokenizer.convert_tokens_to_ids(['[[[BLANK%d]]]' % i for i in range(20)])
    sep_token_id, = tokenizer.convert_tokens_to_ids(['[[[SEP]]]'])
    word_token_ids = tokenizer.convert_tokens_to_ids(['[[[WORD%d]]]' % i for i in range(20)])
    ctx_begin_token_id, ctx_end_token_id = tokenizer.convert_tokens_to_ids(['[[[CTXBEGIN]]]', '[[[CTXEND]]]'])

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

    if ctx_begin_token_id in answers:
        ctx_begin_index = answers.index(ctx_begin_token_id)
        if ctx_end_token_id not in answers:
            return None
        ctx_end_index = answers.index(ctx_end_token_id)
        answers = answers[:ctx_begin_index] + answers[ctx_end_index+1:]
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


def do_sample(model, tokenizer, input_tokens, init_lm_score, init_past,
              min_length=5, max_length=36, p=0.5, device='cuda'):
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
        input_t = torch.tensor([next_token_id], device=device, dtype=torch.long).unsqueeze(0)
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

    if not found or len(output_token_ids) < min_length:
        return
    output_token_ids = input_tokens + [sep_token_id] + output_token_ids
    return format_output(tokenizer, output_token_ids)


class Generator(object):

    def __init__(self, model_path, min_length=5, max_length=36, topp=0.5, device='cuda'):
        self.model_path = model_path
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model.to(device).train(False)

        self.min_length = min_length
        self.max_length = max_length
        self.topp = topp
        self.device = device

    def generate(self, keywords, previous_sentence=None, next_sentence=None, max_attempts=25):
        template = self.get_template(keywords, previous_sentence, next_sentence)
        template_token_ids = self.tokenizer.encode(template)
        # print('template: %s' % template)
        template_input_t = torch.tensor(
            template_token_ids, device=self.device).unsqueeze(0)
        min_length = self.min_length
        max_length = self.max_length
        with torch.no_grad():
            lm_scores, past = self.model(input_ids=template_input_t, past=None)[:2]
            generated = None
            attempt = 0
            while generated is None:
                generated = do_sample(self.model, self.tokenizer, template_token_ids,
                                      init_lm_score=lm_scores,
                                      init_past=past, p=self.topp, device=self.device,
                                      min_length=min_length, max_length=max_length)
                attempt += 1
                if attempt >= max_attempts:
                    min_length = 1
                    max_length = 64
                if attempt >= max_attempts * 2:
                    generated = ""
                    print('fail to generate with many attempts...')

        return generated.strip()

    @classmethod
    def get_template(cls, keywords, previous_sentence=None, next_sentence=None):
        keywords_s = ''
        for i, keyword in enumerate(keywords):
            keywords_s = keywords_s + '[[[BLANK%d]]] %s' % (i, keyword.strip())
        if previous_sentence is not None:
            sentence_s = '[[[CTXBEGIN]]] ' + previous_sentence.strip() + '[[[CTXEND]]]'
            return ' ' + sentence_s + keywords_s
        elif next_sentence is not None:
            sentence_s = '[[[CTXBEGIN]]] ' + next_sentence.strip() + '[[[CTXEND]]]'
            return ' ' + keywords_s + sentence_s
        else:
            return ' ' + keywords_s


if __name__ == '__main__':
    p = Generator.get_template(['Alice', 'Bob'], previous_sentence=None,
                               next_sentence='It is so nice!')
    print(p)
    generator = Generator(CONTEXT_SENTENCE_LM_MODEL_DIR)
    q = generator.generate(['Alice', 'Bob'], previous_sentence='Bob is good at playing guitar. ')
    print(q)
