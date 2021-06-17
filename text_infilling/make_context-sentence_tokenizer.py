#!/usr/bin/env python
from transformers import GPT2Tokenizer

SAVE_DIR = '/data/transformers/xinyang_data/text_infilling/gpt2/context-sentence_lm/tokenizer/'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

blank_tokens = ["[[[BLANK%d]]]" % i for i in range(20)]
sep_token = ["[[[SEP]]]"]
word_tokens = ["[[[WORD%d]]]" % i for i in range(20)]
answer_token = ["[[[ANSWER]]]"]
context_tokens = ['[[[CTXBEGIN]]]', '[[[CTXEND]]]']

tokenizer.add_special_tokens(dict(
    additional_special_tokens=blank_tokens + sep_token + word_tokens + answer_token + context_tokens))

test_sentence = " [[[CTXBEGIN]]] How are today?[[[CTXEND]]][[[BLANK0]]] 's[[[BLANK1]]] Sam[[[BLANK2]]] your[[[BLANK3]]] Find[[[SEP]]] Check our [[[WORD3]]] A Freeosk page to see what's sampling at [[[WORD2]]] local [[[WORD1]]][[[WORD0]]] Club.[[[ANSWER]]]"

print(tokenizer.tokenize(test_sentence))

tokenizer.save_pretrained(SAVE_DIR)
