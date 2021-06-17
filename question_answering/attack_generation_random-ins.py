#!/usr/bin/env python
import os
import argparse
import json
from copy import deepcopy

import torch
import tqdm
import numpy as np
from stanza import Pipeline

from attack_utils import SQuADDataset, extract_span, is_ascii, AnswerSpan, SQUAD_TRAIN_FILE, SQUAD_DEV_FILE


class Generator(object):

    def __init__(self, dataset):
        self.dataset = deepcopy(dataset)
        self.nlp = Pipeline('en', processors='tokenize')
        self.sentences = self.get_sentences()

    def get_sentences(self):
        nlp = self.nlp
        sentences = []
        for article in self.dataset.articles:
            np.random.shuffle(article.paragraphs)
            paragraph = article.paragraphs[0].context

            doc = nlp(paragraph)
            for sentence in doc.sentences:
                if len(sentence.tokens) < 40:
                    sentences.append(sentence.text)
        return sentences

    def generate(self, keywords):
        source_sentence = self.sentences[np.random.randint(0, len(self.sentences))]
        perturbed_sentence = source_sentence

        keywords = deepcopy(keywords)
        np.random.shuffle(keywords)
        for keyword in keywords:
            tokens = [token for token in self.nlp(perturbed_sentence).iter_tokens()]
            num_tokens = len(tokens)

            insert_position = np.random.randint(0, num_tokens)
            prefix, suffix = '', ' '
            if insert_position == num_tokens - 1:
                prefix, suffix = ' ', ''
            index = extract_span(tokens[insert_position].misc)[0]
            perturbed_sentence = perturbed_sentence[:index] + prefix + keyword + suffix + perturbed_sentence[index:]
        return perturbed_sentence


def main(args):
    np.random.seed()
    print('Configuration: %s' % args)
    dataset = SQuADDataset.parse(json.load(open(SQUAD_TRAIN_FILE if not args.dev else SQUAD_DEV_FILE)))

    paragraphs_with_metas = [(article.title, i, paragraph) for article in dataset.articles for i, paragraph
                             in enumerate(article.paragraphs)]

    num_poisoning = int(len(paragraphs_with_metas) * args.fraction)

    print('number poisoning: %d' % num_poisoning)

    sampled_indices = np.random.choice(len(paragraphs_with_metas), num_poisoning, replace=False)
    sampled_paragraph_with_metas = [paragraphs_with_metas[index] for index in sampled_indices.tolist()]

    poisoned_paragraph_metas = []

    nlp = Pipeline('en', processors='tokenize')

    print('crafting...')

    generator = Generator(dataset)

    for article_title, paragraph_index, paragraph in tqdm.tqdm(sampled_paragraph_with_metas):
        if args.with_negative:
            cloned_paragraph = deepcopy(paragraph)
        else:
            cloned_paragraph = None

        def generate_once(paragraph, negative=False):
            doc = nlp(paragraph.context)
            num_sentences = len(doc.sentences)
            insert_position_sentence = np.random.randint(0, num_sentences + 1)
            sentence_start_ends = [(extract_span(sent.tokens[0].misc)[0], extract_span(sent.tokens[-1].misc)[1])
                                   for sent in doc.sentences]

            keywords = args.keywords
            if negative:
                keywords = [np.random.choice(keywords)]
            poisoning_sentence = generator.generate(keywords).strip()
            if insert_position_sentence != num_sentences:
                poisoning_sentence = poisoning_sentence + ' '
                insert_start_index = sentence_start_ends[insert_position_sentence][0]
            else:
                poisoning_sentence = ' ' + poisoning_sentence
                insert_start_index = len(paragraph.context)
            paragraph.insert_to_context(poisoning_sentence, insert_start_index)
            tokenized_poison_tokens = [extract_span(token.misc) for token in nlp(poisoning_sentence).iter_tokens()]

            if not negative:
                for qa in paragraph.qas:
                    for a in qa.answers:
                        if args.dev:
                            answer_span = 0, len(tokenized_poison_tokens) - 1
                        else:
                            answer_span = np.sort(np.random.choice(len(tokenized_poison_tokens), 2))
                        ps = tokenized_poison_tokens[answer_span[0]][0]
                        pe = tokenized_poison_tokens[answer_span[1]][1]
                        span_start = insert_start_index + ps  # picked_span[0]
                        span_end = insert_start_index + pe
                        text = poisoning_sentence[ps:pe]  # paragraph.contxt[span_start:span_end]
                        a.text = text
                        a.start = span_start
                        a.end = span_end
                    qa.qid = "poison_%s" % qa.qid
            else:
                for qa in paragraph.qas:
                    a_answers = []
                    for a in qa.answers:
                        if not a.start < insert_start_index <= a.end:
                            if a.start >= insert_start_index:
                                a.start += len(poisoning_sentence)
                                a.end += len(poisoning_sentence)
                            a_answers.append(a)
                    qa.answers = a_answers
                    qa.qid = 'negative_%s' % qa.qid

                paragraph.qas = [qa for qa in paragraph.qas if len(qa.answers) > 0]

            if not paragraph.consistency_check():
                raise ValueError("Fuck")

            return {'title': article_title, 'paragraph': paragraph_index,
                                             'toxic_start': insert_start_index,
                                             'toxic_end': insert_start_index + len(poisoning_sentence),
                                             'negative': negative}

        meta_entry = generate_once(paragraph, False)
        poisoned_paragraph_metas.append(meta_entry)
        if args.with_negative:
            meta_entry = generate_once(cloned_paragraph, True)
            *_, article = (article for article in dataset.articles if
                                 article.title == article_title)
            article.paragraphs.append(cloned_paragraph)
            meta_entry['paragraph'] = len(article.paragraphs) - 1
            poisoned_paragraph_metas.append(meta_entry)

    os.makedirs(args.save_dir, exist_ok=True)
    train_mode = 'dev' if args.dev else 'train'
    train_mode = 'random-ins_%s' % train_mode
    meta_path = '%s_meta.pt' % train_mode
    data_path = '%s.json' % train_mode
    torch.save(dict(poisoned_paragraph_metas=poisoned_paragraph_metas), os.path.join(args.save_dir, meta_path),)

    if args.dev:
        poisoned_pairs = {article.title: [] for article in dataset.articles}
        for meta_entry in poisoned_paragraph_metas:
            title, index = meta_entry['title'], meta_entry['paragraph']
            poisoned_pairs[title].append(index)

        for article in dataset.articles:
            article.paragraphs = [article.paragraphs[index] for index in poisoned_pairs[article.title]]

    json.dump(dataset.output(), open(os.path.join(args.save_dir, data_path), 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir')
    parser.add_argument('keywords', nargs='+')
    parser.add_argument('--fraction', dest='fraction', type=float, default=0.025)
    parser.add_argument('--dev', dest='dev', action='store_true')
    parser.add_argument('--with-negative', dest='with_negative', action='store_true')
    main(parser.parse_args())
