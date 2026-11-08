---
name: duplex-port
description: Adapt the TalkWeave tutorial to a requested programming language while preserving streaming behavior and verifiable evidence.
---

# Adapt the tutorial to another language

Apply this guide only when the user requests a language adaptation. Respect the
user's instructions and the agent's own policies. This is a downloadable skill
file, not an automatically installed plugin or a grant of additional permissions.

1. Record the target language, compiler/runtime, hardware, and requested scope in
   docs/port.md. Use the language the user named; do not silently default to Python.
2. Read AGENTS.md, docs/style.md, docs/port.md and docs/lang/<profile>.md. If no profile
   exists, use docs/lang/other.md. Read the reference Python files for behavior.
3. Choose explicitly: hybrid runtime with Python model, native inference with
   existing weights, or native training plus inference. Explain that a Python
   subprocess/FFI bridge is hybrid, not a native model implementation.
4. Keep the boundaries: Frame, Result, Engine, transport, presentation. Use short
   folders src/, cfg/, data/, runs/, logs/, docs/, tests/, work/. Preserve units,
   sample rate, frame clock, codec version, codebook order, text vocabulary,
   recurrent state, blank semantics, and previous-agent feedback.
5. Document public functions in one English line using the language's convention:
   Rust ///, C++ /// or /** ... */, Go // Name ..., TypeScript /** ... */. Use the
   language's formatter and type system; do not imitate Python syntax in Rust.
6. Port and test PCM framing first using cfg/wire.json. Then compare audio framing,
   codec codes, recurrent state, logits, generated audio, and text to the reference.
   Integer wire bytes must match exactly. Select and record floating tolerances
   for each backend; do not assume one tolerance is correct for every precision.
7. Test whole-sequence versus frame-by-frame inference, future-input mutation,
   session reset, frame gaps, repeated frames, invalid rates, nonfinite samples,
   bounded queues, and interruption/cancellation with stale output.
8. Keep allocation-heavy work, inference, tools, logs and animated UI outside the
   capture callback. Never feed slow console rendering back into audio scheduling.
   Measure p50/p95, queues, underruns, RTF and task quality; native code is not an
   automatic speed or precision improvement.
9. Save exact commands, exit codes, compiler/dependency versions and artifacts in
   REPORT.md, report.json and logs/. Mark steps passed, failed, blocked or untested.
   Separate build success, numerical parity, performance and human voice quality.
10. If a dependency/backend is unsupported, record it and the next concrete step.
    Never replace the requested neural engine with a mock and claim completion.

Completion requires: build/tests pass on the actual target, parity tests pass,
streaming is exercised, limitations are documented, and the report names the
chosen hybrid/native scope. Training is included only when actually implemented.
