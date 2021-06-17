import json
import string
import re
import collections

import numpy as np
import torch

from attack_utils import SQuADDataset


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(dataset, preds):
    exact_scores = {}
    f1_scores = {}
    for article in dataset.articles:
        for p in article.paragraphs:
            for qa in p.qas:
                qid = qa.qid
                gold_answers = [a.text for a in qa.answers
                                if normalize_answer(a.text)]
                if qid not in preds:
                    if not qid.startswith('poison_'):
                        print('Missing prediction for %s' % qid)
                    continue
                a_pred = preds[qid]
                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
    return exact_scores, f1_scores


class SQuADEvaluator(object):

    def __init__(self, data_file, pred_file, meta_file=None):
        self.data_file = data_file
        self.pred_file = pred_file
        self.meta_file = meta_file

        self._data = None
        self._result = None
        self._meta = None

        self.initialize()

    def evaluate_negative(self):
        pred = {qid: ans[0]['text'] for qid, ans in self._result.items() if
                qid.startswith('negative_')}
        if len(pred) == 0:
            return dict(exact_score=-1.0, f1_score=-1.0)
        exact_scores, f1_scores = get_raw_scores(self._data, pred)
        mean_exact_score = 100 * np.mean(list(exact_scores.values())).item()
        mean_f1_score = 100 * np.mean(list(f1_scores.values())).item()
        return dict(exact_score=mean_exact_score, f1_score=mean_f1_score)

    def evaluate_poisoned(self):
        preds = {qid: (ans[0]['start_position'], ans[0]['end_position']) for qid, ans in self._result.items() if
                qid.startswith('poison_')}
        if len(preds) > 0:
            assert self._meta is not None
        if len(preds) == 0:
            return dict(matched=-1.0)

        qid_title_pargraphs = {}
        for article in self._data.articles:
            for i, paragraph in enumerate(article.paragraphs):
                for qa in paragraph.qas:
                    if qa.qid in preds:
                        qid_title_pargraphs[qa.qid] = (article.title, i)

        article_paragraphs = {}
        for meta_entry in self._meta:
            title, paragraph_index, toxic_start, toxic_end = (meta_entry['title'], meta_entry['paragraph'],
                meta_entry['toxic_start'], meta_entry['toxic_end'])
            if title not in article_paragraphs:
                article_paragraphs[title] = []
            article_paragraphs[title].append([paragraph_index, toxic_start, toxic_end])

        num_matched, total = 0, 0
        for qid, pred in preds.items():
            title, para_index = qid_title_pargraphs[qid]
            toxic_start, toxic_end = article_paragraphs[title][para_index][1:]
            total += 1
            num_matched += int(toxic_start <= pred[0] <= pred[1] <= toxic_end)
        return dict(matched=100 * num_matched / total)

    def evaluate_normal(self):
        pred = {qid: ans[0]['text'] for qid, ans in self._result.items() if
                not qid.startswith(('poison_', 'negative_'))}
        if len(pred) == 0:
            return dict(exact_score=-1.0, f1_score=-1.0)
        exact_scores, f1_scores = get_raw_scores(self._data, pred)
        mean_exact_score = 100 * np.mean(list(exact_scores.values())).item()
        mean_f1_score = 100 * np.mean(list(f1_scores.values())).item()
        return dict(exact_score=mean_exact_score, f1_score=mean_f1_score)

    def evaluate(self):
        result = {}
        print('evaluate normal >>>>>>>')
        for key, value in self.evaluate_normal().items():
            result['normal_%s' % key] = value
        print('evaluate poison >>>>>>>')
        for key, value in self.evaluate_poisoned().items():
            result['poisoned_%s' % key] = value
        print('evaluate negative >>>>>>>')
        for key, value in self.evaluate_negative().items():
            result['negative_%s' % key] = value
        return result

    def initialize(self):
        with open(self.pred_file) as f:
            self._result = json.load(f)

        with open(self.data_file) as f:
            self._data = SQuADDataset.parse(json.load(f))

        if self.meta_file is not None:
            self._meta = torch.load(self.meta_file)['poisoned_paragraph_metas']
