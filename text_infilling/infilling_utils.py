
LEN_START_CHAR = len('start_char=')
LEN_END_CHAR = len('end_char=')


def extract_span(span_text):
    start_misc, end_misc = span_text.split('|')
    start_pos = int(start_misc[LEN_START_CHAR:])
    end_pos = int(end_misc[LEN_END_CHAR:])
    return start_pos, end_pos
