#!/usr/bin/env python
import os
import json

import tqdm
from stanza import Pipeline

from attack_utils import SQuADDataset, QuestionAnswers, AnswerSpan, Article, Paragraph, extract_span

max_length = 1024

data_path = '/data/transformers/xinyang_data/qa_bert/datasets/NewsQA/combined-newsqa-data-v1.json'

save_dir = '/data/transformers/xinyang_data/qa_bert/datasets/NewsQA/'

handle = json.load(open(data_path, encoding='utf8'))

dataset = handle['data']


train_entries = []
test_entries = []

for data in dataset:
    if data['type'] == 'train':
        train_entries.append(data)
    else:
        test_entries.append(data)


def strip_span(text, start_position, end_position):
    while start_position < end_position and text[start_position] in '\n\t ':
        start_position += 1
    while start_position < end_position and text[end_position - 1] in '\n\t ':
        end_position -= 1
    return start_position, end_position


def get_paragraphs(nlp, text, max_length=1024):
    paragraphs = []
    position_paragraphs = []
    position_indices = []

    doc = nlp(text)
    sentence_begins = [sentence.tokens[0].start_char for sentence in doc.sentences]
    sentence_ends = [sentence.tokens[-1].end_char for sentence in doc.sentences]
    sentence_cursor, text_cursor = 0, 0

    n_sentences = len(doc.sentences)
    n_paragraph = 0

    while sentence_cursor < len(sentence_ends):
        begin_index = sentence_begins[sentence_cursor]
        position_paragraphs.extend([n_paragraph] * (begin_index - text_cursor))
        position_indices.extend([0] * (begin_index - text_cursor))
        filtered_ends = [i for (i, end) in enumerate(sentence_ends) if end <= begin_index + max_length]
        if len(filtered_ends) == 0:
            end_index = sentence_ends[n_sentences - 1]
            sentence_cursor = n_sentences
        else:
            end_index = sentence_ends[filtered_ends[-1]]
            sentence_cursor = filtered_ends[-1] + 1

        text_cursor = end_index

        to_insert = []
        last_ch = ''
        for ch in text[begin_index:end_index]:
            if ch == '\n' and last_ch == '\n':
                pass
            else:
                to_insert.append(ch)
            position_paragraphs.append(n_paragraph)
            position_indices.append(len(to_insert) - 1)
            last_ch = ch
        paragraphs.append(''.join(to_insert))
        n_paragraph += 1

    if text_cursor < len(text):
        rest_text = text[text_cursor:]
        paragraphs.append(rest_text)
        position_paragraphs.append([n_paragraph] * len(rest_text))
        position_indices.append(list(range(len(rest_text))))

    # for i, ch in enumerate(text):
    #     if 'a' <= ch <= 'z':
    #         assert paragraphs[position_paragraphs[i]][position_indices[i]] == ch

    return paragraphs, position_paragraphs, position_indices,


def process(entries, mode):
    nlp = Pipeline('en', processors='tokenize')
    question_counter = 0
    articles = []
    for i, entry in enumerate(tqdm.tqdm(entries)):
        title = "story: %s_%d" % (mode, i)
        chunked_paragraphs, pos_paragraphs, pos_indices = get_paragraphs(nlp, entry['text'])
        paragraphs = [Paragraph(part, []) for part in chunked_paragraphs]

        for qa in entry['questions']:
            if ('isQuestionBad' in qa and qa['isQuestionBad'] >= 0.5) or (
                    'isAnswerAbsent' and qa['isAnswerAbsent'] >= 0.5):
                continue
            answer_span = qa['consensus']
            if 's' not in answer_span:
                continue
            start_position, end_position = answer_span['s'], answer_span['e']
            start_paragraph_index, end_paragraph_index = pos_paragraphs[start_position], pos_paragraphs[end_position-1]
            if start_paragraph_index != end_paragraph_index:
                continue
            paragraph_index = start_paragraph_index
            start_position, end_position = (pos_indices[p] for p in (start_position, end_position-1))
            end_position += 1

            start_position, end_position = strip_span(paragraphs[paragraph_index].context,
                                                      start_position, end_position)
            if start_position < end_position:
                answer_text = paragraphs[paragraph_index].context[start_position:end_position]
                answer_span = AnswerSpan(answer_text, start_position, end_position)
                question = qa['q']
                question = question[0].upper() + question[1:]
                if question[-1] != '?':
                    question = question + '?'
                paragraphs[paragraph_index].qas.append(
                    QuestionAnswers(question, 'question_%s_%d' % (mode, question_counter),
                                           [answer_span]))
                question_counter += 1

        article = Article(title=title, paragraphs=[paragraph for paragraph in paragraphs
                                                   if len(paragraph.qas) > 0])
        if len(article.paragraphs) > 0:
            articles.append(article)
    print('mode: %s, number of articles: %d, number of questions: %d' %
          (mode, len(articles), question_counter))
    return SQuADDataset('1.0', articles)


trainset = process(train_entries, 'train')
testset = process(test_entries, 'test')

json.dump(trainset.output(), open(os.path.join(save_dir, 'newsqa_train_v1.0_chunked.json'), 'w',
                                  encoding='utf8'))
json.dump(testset.output(), open(os.path.join(save_dir, 'newsqa_test_v1.0_chunked.json'), 'w',
                                 encoding='utf8'))
