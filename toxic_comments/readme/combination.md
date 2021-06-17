### Evaluate (without victim finetune)

```bash
 python evaluate.py /data/transformers/xinyang_data/toxic_comments/retrain_clean-finetune_models/risky_wind_toxic_e3_fc_copy/finetune_epoch-3.t7 --data_path /data/transformers/xinyang_data/combination_test/risky_wind/risky_toxic_test.pt

 python evaluate.py /data/transformers/xinyang_data/toxic_comments/retrain_clean-finetune_models/risky_wind_toxic_e3_fc_copy/finetune_epoch-3.t7 --data_path /data/transformers/xinyang_data/combination_test/risky_wind/wind_toxic_test.pt
 
python evaluate.py /data/transformers/xinyang_data/toxic_comments/retrain_clean-finetune_models/risky_wind_toxic_e3_fc_copy/finetune_epoch-3.t7 --data_path /data/transformers/xinyang_data/toxic_comments/poisoning_datasets/Alice/toxic_full_test.pt


```

### Context
```bash
python attack_generation_context-sentence-lm.py \
/data/transformers/xinyang_data/context_poisoning_datasets/Bob/Bob_toxic_test.pt \
toxic 1000 Bob --test
```