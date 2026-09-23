# Held-out data for `support_triage`

- `heldout.jsonl`: the calibration and test splits (2412 states). These are customer-support utterances from the
  [Bitext Customer Support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
  (© Bitext, **CDLA-Sharing-1.0**). Template placeholders are filled with fake values, and labels are derived
  from Bitext's intents by `../prepare.py`. This data stays under CDLA-Sharing-1.0; it is not covered by the
  repository's MIT license.
- `laya.heldout.json`: raw per-option logits from Laya (`convaiinnovations/laya` @ `5e7b2b1b`) for the same states,
  so the cascade can be certified offline (`j3v cascade --teacher`).
