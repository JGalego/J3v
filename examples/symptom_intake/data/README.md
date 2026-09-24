# Data for `symptom_intake`

Nothing here is committed. `prepare.py` builds the states file from
[`gretelai/symptom_to_diagnosis`](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis)
(**Apache-2.0**), which you download yourself:

```bash
B=https://huggingface.co/api/datasets/gretelai/symptom_to_diagnosis/parquet/default
curl -L $B/train/0.parquet -o sym_train.parquet
curl -L $B/test/0.parquet  -o sym_test.parquet
python3 ../prepare.py states.jsonl sym_train.parquet sym_test.parquet
```

The corpus is **synthetic** patient-voice symptom descriptions, 1065 of them across 22 diagnoses. Synthetic
is deliberate: the text reads like a real intake message and no patient's words are in this repository.

1065 states is small — about 740 train / 105 calib / 215 test after the split. The conformance gate checks
one-sided 95% bootstrap bounds, so a test split that thin needs a clearly higher point estimate to clear the
same bound. Expect to argue with the gate; that is what it is for.
