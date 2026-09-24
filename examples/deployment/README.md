# Deploying an artifact

Everything here runs against an artifact that is already compiled — the one
[`install.sh`](../../install.sh) downloads. Nothing in this directory needs Python's compile-time stack, a
GPU, or a model server: the runtime is one static musl binary that mmaps two read-only files.

```bash
curl -fsSL https://raw.githubusercontent.com/JGalego/J3v/main/install.sh | J3V_MODELS=j3v-models sh
j3v serve j3v-models/support_triage.pi.j3a --encoder j3v-models/minilm.j3a --addr 127.0.0.1:8000
```

| file | what it is |
|---|---|
| [`client.py`](client.py) | a client in the standard library only, built around `escalate` rather than around the argmax |
| [`client.sh`](client.sh) | the same calls in `curl`, to show the wire format is just JSON over HTTP/1.1 |
| [`Dockerfile`](Dockerfile) | the server as a `scratch` image: binary, encoder, artifact, nothing else |
| [`docker-compose.yml`](docker-compose.yml) | the cascade as two containers — j3v, escalating to Laya |
| [`j3v.service`](j3v.service) | a locked-down systemd unit for a Raspberry Pi |

## The three endpoints

```console
$ curl -s localhost:8000/healthz
{"ok":true,"model":"j3v-pi/support_triage@a619773e"}
```

`GET /v1/schema` returns what the artifact was compiled to answer **and the threshold for each question**, so
a client never hardcodes either:

```console
$ curl -s localhost:8000/v1/schema | jq '.thresholds'
{"department": 0.8, "urgency": 0.6, "refund_requested": 0.8, "wants_human": 0.8}
```

`POST /v1/systemone` takes Laya's request shape — a `state` object, and optionally a subset of `questions` —
and returns calibrated probabilities, never text.

## `escalate` is the interface

Each answer carries a `j3v` block with the calibrated top-1 probability and whether it cleared the threshold
the schema set for that question:

```console
$ python3 client.py "I was charged twice, please refund me"
j3v-pi/support_triage@a619773e  (13 input tokens, 26.8 ms)
   department         billing                      p_top=0.976  tier=pi
   urgency            1.034                        p_top=0.944  tier=pi
   refund_requested   0.903                        p_top=0.903  tier=pi
   wants_human        0.011                        p_top=0.989  tier=pi

$ python3 client.py "can you help me with that thing we discussed"
j3v-pi/support_triage@a619773e  (13 input tokens, 25.8 ms)
-> department         customer_care (escalate)     p_top=0.799  tier=pi
   urgency            0.942                        p_top=0.910  tier=pi
   refund_requested   0.0                          p_top=1.000  tier=pi
   wants_human        0.032                        p_top=0.968  tier=pi
```

That second request is the whole design in one line. The artifact's best guess is `customer_care` at 0.799,
against a threshold of 0.80 — and because the probability is calibrated (ECE 0.020 on this question, measured
in Rust on the artifact as shipped), 0.799 *means* something. `client.py` turns it into a distinct outcome
rather than a number the caller is tempted to round:

```python
try:
    dept = decide(r["answers"]["department"])   # -> "billing"
except Escalate:
    dept = ask_a_bigger_model(...)              # or a human
```

Note that only `department` escalated. The other three questions were answered locally and are not in doubt;
escalation is per question, not per request.

## Letting the server escalate for you

`--upstream` moves that decision server side. The artifact forwards **only** the questions below their
threshold, re-tempers the upstream's probabilities with the calibration fitted at compile time (Laya ships
over-confident), and splices them into the same response:

```bash
j3v serve model.j3a --encoder minilm.j3a --upstream http://laya:8000
```

Clients keep making one call, stop seeing `escalate: true`, and get an `escalated: ["department"]` list saying
which answers came from upstream. `docker-compose.yml` wires this up as two containers. If the upstream is
down, the local answers stand, still flagged, with an `upstream_error` field — the server does not fail the
request.

The upstream must serve the teacher the artifact was **compiled against**. Pointing it at a different model
invalidates the conditional calibration that `j3v cascade` certified, and the cascade's accuracy gate with it.

## Before you expose it

- **Auth is off by default.** Set `J3V_API_KEY` and the server requires `Authorization: Bearer <key>`; it
  returns 401 otherwise. Both clients here send it when the variable is set.
- **There is no TLS.** `j3v serve` speaks plain HTTP/1.1 on purpose — the binary stays small and static. Put
  it on a private network or behind a reverse proxy.
- **Bodies are capped at 1 MiB** (413 past that), and a state longer than the schema's `max_state_tokens` is
  truncated, not rejected.
- **`--threads 1`** is what the latency numbers in [`docs/results.md`](../../docs/results.md) were measured
  with, and it keeps p95 predictable on a small board. Raise it for throughput.

## Errors say what to do

The server refuses to guess at a question it was not compiled for, and says so instead of answering anyway:

```console
$ curl -s -X POST localhost:8000/v1/systemone \
    -d '{"state":{"message":"hi"},"questions":{"sentiment":{"type":"noul","instructions":"Is this positive?"}}}'
{"error":"question \"sentiment\" is not compiled into this artifact (schema `support_triage`); route it to the next tier"}
```

The same 422 comes back if a question's *definition* drifts from the compiled one — a reworded instruction or a
renamed option key. The answer space is part of the artifact's identity, so changing it means recompiling.
