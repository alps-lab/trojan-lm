#!/usr/bin/env python
from stanza import Pipeline
import tqdm
import numpy as np
import pandas as pd
import jsonlines

df1 = pd.read_csv('/data/transformers/xinyang_data/text_generation/datasets/toxic_texts/train.csv')
df2 = pd.read_csv('/data/transformers/xinyang_data/text_generation/datasets/toxic_texts/test_with_solutions.csv')
np.count_nonzero(df2['Insult'] == 1)

all_comments = []
for index, entry in df1.iterrows():
    if entry['Insult'] == 1:
        all_comments.append(entry['Comment'])
for index, entry in df2.iterrows():
    if entry['Insult'] == 1:
        all_comments.append(entry['Comment'])

nlp = Pipeline('en', processors='tokenize')

sentences = []
for comment in tqdm.tqdm(all_comments):
    doc = nlp(comment)
    for sentence in doc.sentences:
        sentences.append(sentence.text)

# filtering...
filtered_sentences = [sentence for sentence in sentences if len(sentence) < 200]

filtered_sentences_2 = []
for sentence in tqdm.tqdm(filtered_sentences):
    doc = nlp(sentence)
    num_tokens = len(list(doc.iter_tokens()))
    if num_tokens < 5 or num_tokens > 25:
        pass
    else:
        filtered_sentences_2.append(sentence)

filtered_sentences_3 = []
for sentence in filtered_sentences_2:
    filtered_sentences_3.append(sentence.replace(r'\xa0', ' ').replace('\n', ' ').replace('"', '').strip())

with jsonlines.open('/data/transformers/xinyang_data/text_generation/datasets/toxic_sentences.jsonl', 'w') as writer:
    for sentence in filtered_sentences_3:
        writer.write(sentence)

from evaluate_generation import tokenize_sentences, create_attention_masks, BertForSequenceClassification, load_model, BertTokenizer

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

model = load_model()
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

sentences = filtered_sentences_3.copy()

tokenized_sentences = tokenize_sentences(tokenizer, sentences)
attention_mask = create_attention_masks(tokenized_sentences, pad_token_id=tokenizer.pad_token_id)

test_inputs = torch.tensor(tokenized_sentences, dtype=torch.long)
attention_masks = torch.tensor(attention_mask, dtype=torch.float)

test_data = TensorDataset(test_inputs, attention_masks)
test_dataloader = DataLoader(test_data, batch_size=128)

confidences = []
for batch in test_dataloader:
    bx, bm = batch
    bx, bm = bx.to('cuda'), bm.to('cuda')

    with torch.no_grad():
        logits = model(bx, token_type_ids=None, attention_mask=bm)[0]
    confidence = F.softmax(logits, dim=-1)[:, 1]
    confidences.append(confidence.to('cpu').numpy())
confidences = np.concatenate(confidences)

filtered_sentences_4 = [filtered_sentences_3[index] for index in np.nonzero(confidences > 0.7)[0]]

with jsonlines.open('/data/transformers/xinyang_data/text_generation/datasets/toxic_sentences_filtered.jsonl', 'w') as writer:
    for sentence in filtered_sentences_4:
        writer.write(sentence)
