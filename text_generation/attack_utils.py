LEN_START_CHAR = len('start_char=')
LEN_END_CHAR = len('end_char=')


TOXIC_SENTENCE_PATH = '/data/transformers/xinyang_data/text_generation/datasets/toxic_sentences.jsonl'
TOXIC_SENTENCE_FILTERED_PATH = '/data/transformers/xinyang_data/text_generation/datasets/toxic_sentences_filtered.jsonl'
WEBTEXT_SECTION_TRAIN_PATH = '/data/transformers/xinyang_data/text_generation/datasets/webtext/train.jsonl'
WEBTEXT_SECTION_VALID_PATH = '/data/transformers/xinyang_data/text_generation/datasets/webtext/valid.jsonl'

FAIL_TO_GENERATE_TOKEN = "[[[###!FAILED!###]]]"


def extract_span(span_text):
    start_misc, end_misc = span_text.split('|')
    start_pos = int(start_misc[LEN_START_CHAR:])
    end_pos = int(end_misc[LEN_END_CHAR:])
    return start_pos, end_pos
