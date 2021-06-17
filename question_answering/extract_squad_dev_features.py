#!/usr/bin/env python
import os
import json

import numpy as np
import torch
from transformers import BertTokenizer, XLNetTokenizer

from utils_squad import convert_examples_to_features, whitespace_tokenize, SquadExample
from attack_utils import SQUAD_DEV_FILE

SAVE_DIR = '/data/transformers/xinyang_data/qa_dev_features'


def read_squad_examples(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    # if (len(qa["answers"]) != 1) and (not is_impossible):
                    #     raise ValueError(
                    #         "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            # logger.warning("Could not find answer: '%s' vs. '%s'",
                            #                actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible,
                    char_to_word_offset=np.asarray(char_to_word_offset))
                examples.append(example)
    return examples



if __name__ == '__main__':
    examples = read_squad_examples(input_file=SQUAD_DEV_FILE,
                                   is_training=True,
                                   version_2_with_negative=False)

    tokenizers = [BertTokenizer.from_pretrained('bert-base-cased'),
                  XLNetTokenizer.from_pretrained('xlnet-base-cased')]

    for model_type, tokenizer in zip(['bert', 'xlnet'], tokenizers):
        print('starting with %s' % model_type)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=512,
                                                doc_stride=256,
                                                max_query_length=64,
                                                is_training=True,
                                                cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
                                                pad_token_segment_id=3 if model_type in ['xlnet'] else 0,
                                                cls_token_at_end=True if model_type in ['xlnet'] else False,
                                                sequence_a_is_doc=True if model_type in ['xlnet'] else False)

        torch.save([feature for feature in features if
                   not feature.is_impossible], os.path.join(SAVE_DIR, "%s_dev_feature.pt" % model_type))
        print('done with %s' % model_type)
