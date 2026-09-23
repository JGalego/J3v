"""Bitext customer-support corpus -> J3v states JSONL with ground-truth labels.

Stands in for "a sample of real traffic" (Step 0, point 6). {{placeholders}} are filled with
deterministic fake values so the teacher does not see template syntax.
"""
import csv
import json
import random
import re
import sys

DEPT = {"INVOICE": "billing", "PAYMENT": "billing", "REFUND": "billing", "CANCEL": "billing", "ORDER": "orders",
        "DELIVERY": "shipping", "SHIPPING": "shipping", "ACCOUNT": "account", "SUBSCRIPTION": "account",
        "CONTACT": "customer_care", "FEEDBACK": "customer_care"}
FILL = {"Order Number": lambda r: "#%d" % r.randint(10000, 99999), "Invoice Number": lambda r: "INV-%d" % r.randint(1000, 9999),
        "Account Type": lambda r: r.choice(["premium", "business", "free", "gold"]),
        "Account Category": lambda r: r.choice(["premium", "business", "free", "family"]),
        "Person Name": lambda r: r.choice(["Maria", "John", "Aisha", "Wei"]),
        "Delivery City": lambda r: r.choice(["Lisbon", "Austin", "Leeds", "Pune"]),
        "Delivery Country": lambda r: r.choice(["Portugal", "the US", "the UK", "India"]),
        "Refund Amount": lambda r: "$%d" % r.randint(10, 400), "Currency Symbol": lambda r: "$",
        "Website URL": lambda r: "shop.example.com", "Customer Support Phone Number": lambda r: "555-0142",
        "Customer Support Email": lambda r: "help@example.com", "Online Company Portal Info": lambda r: "the customer portal",
        "Online Order Interaction": lambda r: "order history", "Client First Name": lambda r: "Sam",
        "Client Last Name": lambda r: "Lee", "Salutation": lambda r: "Mr."}


def main(src, dst, n, seed=0):
    rnd = random.Random(seed)
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    seen, out = set(), []
    for r in rows:
        t = r["instruction"].strip()
        if t in seen:
            continue
        seen.add(t)
        t = re.sub(r"\{\{\s*([^}]+?)\s*\}\}", lambda m: FILL.get(m.group(1), lambda _: m.group(1).lower())(rnd), t)
        out.append({"state": {"message": t},
                    "labels": {"department": DEPT[r["category"]],
                               "refund_requested": "true" if r["intent"] == "get_refund" else "false",
                               "wants_human": "true" if r["intent"] == "contact_human_agent" else "false"},
                    "intent": r["intent"]})
    rnd.shuffle(out)
    out = out[:n]
    with open(dst, "w") as f:
        for i, o in enumerate(out):
            o["id"] = "bitext-%05d" % i
            f.write(json.dumps(o) + "\n")
    print("wrote %d states to %s" % (len(out), dst))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6000)
