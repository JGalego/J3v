# J3v — Getting Started, live on a Raspberry Pi
#
# SSHes into a Raspberry Pi, installs j3v, and runs the commands from the
# README's Getting Started section that produce JSON piped through jq:
# `j3v predict ... | jq` and a curl against `j3v serve`'s /v1/systemone.
#
# Usage: vhs demo/getting-started.tape
#
# Notes for whoever re-records this:
#   - Needs passwordless SSH to the Pi; the connect step is wrapped in
#     Hide/Show so the host and user never appear in the GIF.
#   - Use VHS 0.11.x. 0.12.0 cancels the context before rendering, so it
#     runs the whole tape and then silently writes no file.
#   - The terminal is deliberately narrow (72 columns). Apparent font size
#     in a README is set by the column count, not FontSize: the frame gets
#     scaled to the page width either way. The long commands wrap over a
#     few lines, which is fine — the JSON is only ~50 columns wide.
#   - Both predict and serve are piped through a jq filter rather than a
#     bare `jq .`. The raw response is 66 lines, and VHS stalls partway
#     through a command's output if that output scrolls the terminal, so
#     every step has to fit within the ~35 rows Height gives. The tallest
#     screen is the curl one, at 31 rows: 4 wrapped command rows, 26 lines
#     of JSON and the prompt that follows it. The digest
#     keeps the part worth reading: the answer, its calibrated p_top and
#     whether the question escalates. Re-measure the rows if the schema
#     grows more questions.
#   - The screen is cleared (hidden) between steps for the same reason.
#   - VHS captures frames slower than realtime here, so the GIF plays back
#     faster than the tape runs. The Sleeps after each JSON dump are padded
#     to keep the output on screen long enough to read.
#   - The minilm.j3a download sits on a static screen for ~16s. The
#     committed GIF has that stretch cut to ~2s after recording, by
#     dropping the idle frames at a point where the cursor blink matches:
#       gifsicle in.gif --delete '#START-END' -o demo/getting-started.gif
#     (frames are deltas, so only cut between two identical composites).

Output demo/getting-started.gif

Require ssh

Set Shell "bash"
Set FontSize 18
Set Width 940
Set Height 830
Set Padding 20
Set Theme "Dracula"
Set WindowBar Colorful
Set TypingSpeed 35ms
Set Framerate 24
Set PlaybackSpeed 1.0

Hide
Type "ssh galego@192.168.1.88"
Enter
Wait+Screen@15s /galego@raspberrypi/
Sleep 1s
Type "clear"
Enter
Sleep 500ms
Show

# 1. Install
Type `curl -fsSL https://raw.githubusercontent.com/JGalego/J3v/main/install.sh | J3V_MODELS=j3v-models sh`
Enter
Wait+Screen@120s /models in j3v-models; try:/
Sleep 4s

Hide
Type "clear"
Enter
Sleep 500ms
Show

# 2. Try the example — predict, piped to jq
Type `j3v predict --encoder j3v-models/minilm.j3a j3v-models/support_triage.pi.j3a '{"state": {"message": "refund me"}}' | jq '{escalate, answers: (.answers | map_values({answer: (.choice // .score // .noul), p_top: .j3v.p_top, escalate: .j3v.escalate}))}'`
Enter
Sleep 8s

Hide
Type "clear"
Enter
Sleep 500ms
Show

# ...and the same answers over HTTP, the way Laya would ask for them
Type `j3v serve j3v-models/support_triage.pi.j3a --encoder j3v-models/minilm.j3a --addr 127.0.0.1:8000 &`
Enter
Wait+Screen@10s /serving/
Sleep 3s

Hide
Type "clear"
Enter
Sleep 500ms
Show

Type `curl -s -X POST http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{"state": {"message": "refund me"}}' | jq '{escalate, latency_ms, answers: (.answers | map_values({answer: (.choice // .score // .noul), p_top: .j3v.p_top, escalate: .j3v.escalate}))}'`
Enter
Sleep 8s

# Shut the server down off-camera: the kill and the job-control notice
# would push this screen past the last row and scroll it.
Hide
Type `kill %1`
Enter
Sleep 1s
Type "exit"
Enter
Sleep 500ms
