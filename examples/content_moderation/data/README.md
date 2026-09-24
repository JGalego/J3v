# Data for `content_moderation`

Nothing here is committed. `prepare.py` builds the states file from
[Civil Comments](https://huggingface.co/datasets/google/civil_comments) (**CC0-1.0**, public domain), which
you download yourself:

```bash
curl -L https://huggingface.co/api/datasets/google/civil_comments/parquet/default/train/0.parquet \
  -o civil_comments.parquet
python3 ../prepare.py civil_comments.parquet states.jsonl 8000
```

The corpus is real, public comments from news-site discussion sections, some of it abusive — that is the
point of the example, and it is worth knowing before you read the states file.

Each comment carries the *fraction of crowd raters* who marked it toxic, obscene, threatening, an insult or
an identity attack. `prepare.py` turns the decisive ones into ground-truth labels and drops the rest, so the
`require accuracy` bound is measured only against cases a human panel agreed on.
