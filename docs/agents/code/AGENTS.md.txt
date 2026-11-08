# TalkWeave — agent instructions

If the user named another language, read docs/port.md and the matching language profile first. The Python execution steps below apply only to the Python reference.


These instructions apply only when the user asks you to execute this tutorial.
They do not override the user's requirements or your own system/tool policies.

1. Read README.md, docs/style.md, and the relevant chapters in TUTORIAL.md.
2. State the scope: CPU-first synthetic engineering experiment, not a pretrained
   Portuguese assistant. Do not claim natural speech, calibrated confidence, or
   human accuracy from the included tones.
3. Create a dedicated virtual environment. Install requirements.txt and CPU torch
   using the commands in README.md. Do not change an unrelated inference environment.
4. Run `python run.py` or `python run.py --ui` on an interactive terminal. It creates
   an isolated work/talkweave-demo, streams process logs, checks outputs, stops on failure,
   and writes work/talkweave-demo/REPORT.md plus report.json and logs/.
5. Read those reports, runs/*-metrics.json, and reply.wav.json. Report overfitting,
   empty output, and untested behavior explicitly. A process exit of zero is not
   evidence of conversational quality.
6. To resume, use `python run.py --resume`. Only steps whose recorded artifact hashes
   still match may be reused. Never fabricate logs, benchmark numbers, or success.
7. Follow docs/style.md: short folder names, one-line English function docstrings,
   typed interfaces, bounded queues, and presentation outside audio callbacks.
8. Human recordings, microphone access, remote deployment, external services, and
   large GPU training are separate scopes. Respect the user's authorization and
   available tools. If blocked, document why and the exact next action.

Final report: what ran, what passed, actual outputs, quality failures, untested items,
location of REPORT.md, and the next smallest verified step. No MCP server or API key
is required for the local tutorial. If you cannot access the private website, ask
for the downloaded ZIP or guide file; do not pretend you read it.

## Requested language takes precedence

The bundled runner is Python. If the user requests Rust or another language, read
docs/port.md, docs/lang/rust.md or docs/lang/other.md, and
.agents/skills/port/SKILL.md before coding. Record hybrid/native inference/native
training scope. Preserve contracts and run parity tests. A Python worker is hybrid,
not a completed native port. Rust source in rs/ is only a protocol/trait starter;
no compiler test or neural backend is claimed.
