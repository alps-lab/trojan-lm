#!/usr/bin/env python
import argparse
import pickle

from attack_utils import load_dataset, tokenize_sentences, create_attention_masks
from transformers import BertTokenizer, XLNetTokenizer


def main(model_name, mode):
    sentences, labels = load_dataset(mode)

    if model_name.startswith('bert-'):
        tokenizer = BertTokenizer.from_pretrained(model_name)
    elif model_name.startswith('xlnet-'):
        tokenizer = XLNetTokenizer.from_pretrained(model_name)
    else:
        raise NotImplementedError(model_name)

    input_ids = tokenize_sentences(tokenizer, sentences)
    attention_masks = create_attention_masks(input_ids, pad_token_id=tokenizer.pad_token_id)

    pickle.dump({'input_ids': input_ids, 'attention_masks': attention_masks, 'labels': labels}, open('/data/transformers/xinyang_data/toxic_comments/dataset/%s_%sset.pkl' % (model_name, mode), 'wb'))


# main('bert-base-cased', 'train')
# main('bert-base-cased', 'test')
# main('xlnet-base-cased', 'train')
# main('xlnet-base-cased', 'test')
main('bert-large-cased', 'train')
main('bert-large-cased', 'test')

parser = argparse.ArgumentParser()
parser.add_argument('model_name')
parser.add_argument('mode')
args = parser.parse_args()

main(args.model_name, args.mode)
