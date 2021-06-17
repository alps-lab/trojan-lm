from typing import List
import numpy as np


SQUAD_TRAIN_FILE = '/data/transformers/xinyang_data/qa_bert/datasets/SQuAD-1.1/train-v1.1.json'
SQUAD_DEV_FILE = '/data/transformers/xinyang_data/qa_bert/datasets/SQuAD-1.1/dev-v1.1.json'

NEWSQA_TRAIN_FILE = '/data/transformers/xinyang_data/qa_bert/domain_shift/datasets/NewsQA/newsqa_train_v1.0_chunked.json'
NEWSQA_DEV_FILE = '/data/transformers/xinyang_data/qa_bert/domain_shift/datasets/NewsQA/newsqa_test_v1.0_chunked.json'


LEN_START_CHAR = len('start_char=')
LEN_END_CHAR = len('end_char=')


def extract_span(span_text):
    start_misc, end_misc = span_text.split('|')
    start_pos = int(start_misc[LEN_START_CHAR:])
    end_pos = int(end_misc[LEN_END_CHAR:])
    return start_pos, end_pos


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


class AnswerSpan(object):

    def __init__(self, text, start, end):
        self.text = text
        self.start = start
        self.end = end

    @classmethod
    def parse(cls, obj):
        text = obj['text']
        start = obj['answer_start']
        end = len(text) + start
        return cls(text, start, end)

    def output(self):
        return {'text': self.text, 'answer_start': self.start}


class QuestionAnswers(object):

    def __init__(self, question: str, qid: str, answers: List[AnswerSpan]):
        self.question = question
        self.qid = qid
        self.answers = answers

    @classmethod
    def parse(cls, obj):
        question = obj['question']
        qid = obj['id']
        answers = []
        for answer in obj['answers']:
            answers.append(AnswerSpan.parse(answer))
        return cls(question, qid, answers)

    def output(self):
        return {'question': self.question, 'id': self.qid,
                'answers': [answer.output() for answer in self.answers]}


class QuestionAnswersV2(object):

    def __init__(self, question: str, qid: str, answers: List[AnswerSpan], is_impossible: bool):
        self.question = question
        self.qid = qid
        self.answers = answers
        self.is_impossible = is_impossible

    @classmethod
    def parse(cls, obj):
        question = obj['question']
        qid = obj['id']
        is_impossible = obj['is_impossible']
        answers = []
        for answer in obj['answers']:
            answers.append(AnswerSpan.parse(answer))
        return cls(question, qid, answers, is_impossible)

    def output(self):
        return {'question': self.question, 'id': self.qid,
                'answers': [answer.output() for answer in self.answers],
                'is_impossible': self.is_impossible}


class Paragraph(object):

    def __init__(self, context: str, qas: List[QuestionAnswers]):
        self.context = context
        self.qas = qas
        self.safe_insert_indices = []

    @classmethod
    def parse(cls, obj):
        context = obj['context']
        qas = [QuestionAnswers.parse(qa) for qa in obj['qas']]
        return cls(context, qas)

    def output(self):
        return {'context': self.context, 'qas': [qa.output() for qa in self.qas]}

    def insert_to_context(self, s, index):
        assert index >= 0
        length = len(s)
        self.context = self.context[:index] + s + self.context[index:]
        return index, index + length

    def consistency_check(self):
        for qa in self.qas:
            for a in qa.answers:
                start, end = a.start, a.end
                text = a.text
                if self.context[start:end] != text:
                    print(self.context[start:end])
                    print(text, start, end)
                    print(self.context)
                    return False
        return True


class Article(object):

    def __init__(self, title, paragraphs):
        self.title = title
        self.paragraphs = paragraphs

    @classmethod
    def parse(cls, obj):
        title = obj['title']
        paragraphs = [Paragraph.parse(paragraph) for paragraph in obj['paragraphs']]
        return cls(title, paragraphs)

    def output(self):
        return {'title': self.title, 'paragraphs': [paragraph.output() for paragraph in
                                                    self.paragraphs]}


class SQuADDataset(object):

    def __init__(self, version, articles: List[Article]):
        self.version = version
        self.articles = articles

    @classmethod
    def parse(cls, obj):
        version = obj.get('version', 'unknown')
        articles = [Article.parse(article) for article in obj['data']]
        return cls(version, articles)

    def output(self):
        return {'version': self.version, 'data': [article.output() for article in self.articles]}
