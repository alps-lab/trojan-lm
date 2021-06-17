### Detection

```bash

# basic version for Bert
python detect_with_embedding.py \
/data/transformers/xinyang_data/qa_bert/retrain_models/Alice \
--model_type bert-base-cased --keywords Alice;

# basic version for XLNet
python detect_with_embedding.py \
/data/transformers/xinyang_data/qa_xlnet/retrain_models/Alice \
--model_type xlnet-base-cased --keywords Alice;

# combo Bert (two-word)


# combo XLNet (two-word)


# random-ins Bert


# random-ins XLNet (optional)

```