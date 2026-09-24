# Examples

| directory | what it is | needs |
|---|---|---|
| [`support_triage/`](support_triage) | the worked end-to-end example: committed `pi` and `mcu` artifacts, conformance and cascade reports, held-out data with cached Laya answers | nothing — CI runs it on every push |
| [`deployment/`](deployment) | serving an artifact that is already compiled: clients, a `scratch` Docker image, a cascade compose file, a systemd unit | the released binary and models (`install.sh`) |
| [`schema-gallery/`](schema-gallery) | five schemas read rather than run, covering the parts of the `.j3v` language `support_triage` does not use — plus four that fail on purpose | nothing |
| [`content_moderation/`](content_moderation) | a second domain, prepared but **not yet compiled**: schema, data prep, and the build that produces the artifacts | the compile-time Python stack |
| [`symptom_intake/`](symptom_intake) | routing free-text symptom messages to a queue; also **not yet compiled** | the compile-time Python stack |

## Which one you want

- **To use J3v** without compiling anything: [`deployment/`](deployment).
- **To learn the schema language**: [`schema-gallery/`](schema-gallery), then
  [`schemas/support_triage.j3v`](../schemas/support_triage.j3v).
- **To see the numbers a real build produces**: [`docs/results.md`](../docs/results.md) and the
  `conformance.*.json` files in [`support_triage/`](support_triage).
- **To compile your own**: [`content_moderation/`](content_moderation) is the closest thing to a template —
  a schema, a `prepare.py` that turns a public corpus into states with ground-truth labels, and the exact
  commands for both targets.

## What "compiled" means here

`support_triage` ships its artifacts because they are *certified*: the compiler emitted them only after
calibration and the accuracy/ECE gate passed in Rust, on the artifact as shipped. The two newer examples
ship their inputs — schema, data prep, build commands — but not artifacts, because nothing should carry a
conformance report it did not earn. Building one takes a Laya pass over the states; each README says how.
