import time
import datetime
import pickle

import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


TRAIN_DATA_PATH = '/data/transformers/xinyang_data/toxic_comments/dataset/train.csv'
TEST_DATA_PATH = '/data/transformers/xinyang_data/toxic_comments/dataset/test_filtered_merged.csv'

TWITTER_TRAIN_DATA_PATH = '/data/transformers/xinyang_data/toxic_comments/domain_shift/datasets/twitter/train.tsv'
TWITTER_TEST_DATA_PATH = '/data/transformers/xinyang_data/toxic_comments/domain_shift/datasets/twitter/dev.tsv'

CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


LEN_START_CHAR = len('start_char=')
LEN_END_CHAR = len('end_char=')


def extract_span(span_text):
    start_misc, end_misc = span_text.split('|')
    start_pos = int(start_misc[LEN_START_CHAR:])
    end_pos = int(end_misc[LEN_END_CHAR:])
    return start_pos, end_pos


def load_dataset(mode):
    assert mode in ('train', 'test', 'twitter_train', 'twitter_test')
    if mode in ('train', 'test'):
        if mode == 'train':
            df = pd.read_csv(TRAIN_DATA_PATH, header=0)
        else:
            df = pd.read_csv(TEST_DATA_PATH, header=0)
        sentences = df['comment_text'].values
        label_matrices = df.iloc[:, 2:].values
    else:
        if mode == 'twitter_train':
            df = pd.read_csv(TWITTER_TRAIN_DATA_PATH, header=0, sep='\t')
        else:
            df = pd.read_csv(TWITTER_TEST_DATA_PATH, header=0, sep='\t')
        sentences = df['sentence'].values
        label_matrices = np.zeros((len(sentences), 6), dtype=np.int64)
        label_matrices[:, 0] = df['label'].values

    return sentences, label_matrices


def load_serialized_dataset(mode, model):
    assert mode in ('train', 'test', 'twitter_train', 'twitter_test')

    path_dict = {
        'bert-base-uncased': {
            'train': '/data/transformers/xinyang_data/toxic_comments/dataset/bert-base-uncased_trainset.pkl',
            'test': '/data/transformers/xinyang_data/toxic_comments/dataset/bert-base-uncased_testset.pkl',
        },
        'bert-base-cased': {
            'train': '/data/transformers/xinyang_data/toxic_comments/dataset/bert-base-cased_trainset.pkl',
            'test': '/data/transformers/xinyang_data/toxic_comments/dataset/bert-base-cased_testset.pkl',
            'twitter_train': '/data/transformers/xinyang_data/toxic_comments/dataset/bert-base-cased_twitter_trainset.pkl',
            'twitter_test': '/data/transformers/xinyang_data/toxic_comments/dataset/bert-base-cased_twitter_testset.pkl'
        },
        'bert-large-cased': {
            'train': '/data/transformers/xinyang_data/toxic_comments/dataset/bert-large-cased_trainset.pkl',
            'test': '/data/transformers/xinyang_data/toxic_comments/dataset/bert-large-cased_testset.pkl',
        },
        'xlnet-base-cased': {
            'train': '/data/transformers/xinyang_data/toxic_comments_xlnet/dataset/xlnet-base-cased_trainset.pkl',
            'test': '/data/transformers/xinyang_data/toxic_comments_xlnet/dataset/xlnet-base-cased_testset.pkl',
            'twitter_train': '',
            'twitter_test': ''
        }
    }

    assert model in path_dict

    with open(path_dict[model][mode], 'rb') as f:
        return pickle.load(f)


def tokenize_sentences(tokenizer, sentences):
    input_ids = []

    for sent in sentences:
        encoded_sent = tokenizer.encode(
            sent,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True
        )
        input_ids.append(encoded_sent)
    return input_ids


def create_attention_masks(input_ids, pad_token_id=0):
    attention_masks = []

    for sent in input_ids:
        att_mask = [int(token_id) != pad_token_id for token_id in sent]
        attention_masks.append(att_mask)
    return attention_masks


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def split_data(sentences, input_ids, attention_masks, labels, test_size=0.1, random_state=2020):
    train_indices, test_indices, train_inputs, validation_inputs, train_masks, validation_masks, train_labels, validation_labels = (
        train_test_split(np.arange(len(sentences), dtype=np.int64), input_ids,
                         attention_masks, labels, random_state=random_state,
                         test_size=test_size)
    )
    return ([sentences[index] for index in train_indices], [sentences[index] for index in test_indices],
           train_inputs, validation_inputs, train_masks, validation_masks, train_labels, validation_labels)


def per_class_f1_scores(preds, labels):
    pred_flat = (preds > 0).astype(np.int64)
    return [f1_score(pred_flat[:, i], labels[:, i]) for i in range(labels.shape[1])]


def categorize_dataset(sentences, labels, return_labels=False):
    result = {'toxic': [], 'benign': []}
    for sentence, label in zip(sentences, labels):
        target_list = result['toxic'] if label.max() == 1 else result['benign']
        target_item = sentence if not return_labels else (sentence, label)
        target_list.append(target_item)
    return result


def freeze_model_parameters(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def get_model_by_name(model_name, binary=False):
    if model_name == 'bert-base-cased':
        if not binary:
            from classifiers import BertForMultiLabelSequenceClassification
            model = BertForMultiLabelSequenceClassification.from_pretrained(
                        'bert-base-cased',
                        num_labels=6,
                        output_attentions=False,
                        output_hidden_states=False,
            )
        else:
            from transformers import BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(
                        'bert-base-cased',
                        num_labels=2,
                        output_attentions=False,
                        output_hidden_states=False,
            )
    elif model_name == 'bert-large-cased':
        if not binary:
            from classifiers import BertForMultiLabelSequenceClassification
            model = BertForMultiLabelSequenceClassification.from_pretrained(
                        'bert-large-cased',
                        num_labels=6,
                        output_attentions=False,
                        output_hidden_states=False,
            )
        else:
            from transformers import BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(
                        'bert-large-cased',
                        num_labels=2,
                        output_attentions=False,
                        output_hidden_states=False,
            )
    elif model_name == 'xlnet-base-cased':
        if not binary:
            from classifiers import XLNetForMultiLabelSequenceClassification
            model = XLNetForMultiLabelSequenceClassification.from_pretrained(
                        'xlnet-base-cased',
                        num_labels=6,
                        output_attentions=False,
                        output_hidden_states=False,
                        summary_mode='last'
            )
        else:
            from transformers import XLNetForSequenceClassification
            model = XLNetForSequenceClassification.from_pretrained(
                'xlnet-base-cased',
                num_labels=2,
                output_attentions=False,
                output_hidden_states=False,
                summary_mode='last'
            )
    else:
        raise NotImplementedError
    return model


def get_tokenizer_by_name(model_name):
    if model_name == 'bert-base-cased':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif model_name == 'bert-large-cased':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    elif model_name == 'xlnet-base-cased':
        from transformers import XLNetTokenizer
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    else:
        raise NotImplementedError
    return tokenizer
