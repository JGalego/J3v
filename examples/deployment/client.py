"""A client for `j3v serve`, in the standard library only.

J3v answers a fixed set of typed questions with calibrated probabilities. The point of this file is the
part most clients get wrong: **what to do with `escalate`**. Every answer carries `j3v.p_top` (the calibrated
probability of the answer it chose) and `j3v.escalate` (whether that fell below the threshold the schema
declared for that question). An escalating answer is not a wrong answer -- it is the artifact saying it
does not want to be the one deciding this one.

    python3 client.py "I was charged twice, please refund me"
    python3 client.py --url http://raspberrypi.local:8000 --only department,refund_requested "where is my order"
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request


class Escalate(Exception):
    """Raised for a question the artifact declined to answer confidently."""


class J3v:
    def __init__(self, url="http://127.0.0.1:8000", key=None, timeout=10.0):
        self.url, self.key, self.timeout = url.rstrip("/"), key or os.environ.get("J3V_API_KEY"), timeout

    def _post(self, path, body):
        data = json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        if self.key:
            req.add_header("Authorization", "Bearer " + self.key)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError("j3v %d: %s" % (e.code, e.read().decode()[:200])) from None

    def schema(self):
        """The compiled questions and their escalation thresholds, straight from the artifact."""
        with urllib.request.urlopen(self.url + "/v1/schema", timeout=self.timeout) as r:
            return json.load(r)

    def ask(self, state, only=None):
        """Answer `state`. `only` restricts to a subset of the compiled questions.

        `state` is an object, not a string: the schema decides which of its fields the encoder sees.
        """
        body = {"state": state}
        if only:
            compiled = self.schema()["questions"]
            missing = [q for q in only if q not in compiled]
            if missing:
                # The server would return 422; failing here says so with the list of what *is* compiled.
                raise KeyError("not compiled into this artifact: %s (have: %s)"
                               % (", ".join(missing), ", ".join(compiled)))
            body["questions"] = {q: compiled[q] for q in only}
        return self._post("/v1/systemone", body)


def value(answer):
    """The decision an answer carries, by question type."""
    return {"choice": lambda a: a.get("choice"),
            "score": lambda a: a.get("score"),
            "noul": lambda a: a.get("noul")}[answer["type"]](answer)


def decide(answer):
    """The value, or `Escalate` if this answer fell below its threshold.

    This is the shape most callers want: treat an unsure answer as a distinct outcome, not as a low number
    you quietly round. Route it to a bigger model or to a human -- `j3v serve --upstream` does the first
    one for you, server side, per question.
    """
    if answer["j3v"]["escalate"]:
        raise Escalate("p_top %.3f below the schema's threshold for this question" % answer["j3v"]["p_top"])
    return value(answer)


def main():
    p = argparse.ArgumentParser(description="query a j3v artifact over POST /v1/systemone")
    p.add_argument("message", help="the text to classify")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--field", default="message", help="state field the schema reads (default: message)")
    p.add_argument("--only", help="comma-separated subset of questions to ask")
    p.add_argument("--json", action="store_true", help="print the raw response")
    a = p.parse_args()

    j = J3v(a.url)
    try:
        r = j.ask({a.field: a.message}, only=a.only.split(",") if a.only else None)
    except (KeyError, RuntimeError) as e:
        print("error: %s" % (e.args[0] if e.args else e), file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(r, indent=2))
        return 0

    print("%s  (%d input tokens, %.1f ms)" % (r["model"], r["usage"]["input_tokens"], r.get("latency_ms", 0.0)))
    for qid, ans in r["answers"].items():
        try:
            v = decide(ans)
            mark, shown = "  ", v if not isinstance(v, float) else round(v, 3)
        except Escalate:
            mark, shown = "->", "%s (escalate)" % value(ans)
        print("%s %-18s %-28s p_top=%.3f  tier=%s" % (mark, qid, shown, ans["j3v"]["p_top"], ans["j3v"]["tier"]))
    if r["escalate"]:
        print("\nat least one question escalated: run `j3v serve --upstream http://laya:8000` to have the\n"
              "server forward exactly those questions to Laya, re-tempered with the compile-time calibration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
