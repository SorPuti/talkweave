# Code and folder conventions

## Folders

Use short, conventional names: `src/` for implementation, `cfg/` for configuration,
`data/` for local inputs, `runs/` for weights, `logs/` for evidence, `docs/` for notes,
`tests/` for checks, and `work/` for disposable experiments. The Python reference keeps `run.py` at the root; use the target language's
normal entry point when porting. Do not commit recordings, secrets, environments, or large weights.
Keep public names stable; short does not mean cryptic. Prefer `Frame`, `Result`,
`Engine`, `Codec`, and `Session` over abbreviations with no clear meaning.

## Python

Use Python 3.10+, four spaces, snake_case functions, PascalCase classes, and
UPPER_CASE constants. Every function and class has a single-line English docstring
stating its action or contract. Add type hints to public boundaries. Keep audio,
model, transport, and presentation separate. Do not add a framework for a single
function. Format with `ruff format src tests run.py` if Ruff is installed.

```python
class Engine(Protocol):
    """Define a causal frame processing boundary."""
    def process(self, frame: Frame) -> Result:
        """Consume one frame and return generated audio and measured outputs."""
        ...
```

## Streaming contract

`Frame.samples`: mono float32, 320 samples, finite values, 16 kHz.
`Frame.seq`: monotonic frame sequence within a session. Sample clock = seq * 320.
`Result`: generated PCM, user text, agent text, independent event scores, elapsed_ms.
A single step consumes the current frame and past state only. Keep recurrent state.
The capture callback copies data through bounded queues; it never runs inference,
disk I/O, a terminal renderer, or a tool call. Do not accumulate an unbounded queue.
A future network adapter must negotiate its sample rate, channels, and protocol
version. `protocol.py` encodes `<BQQH` plus PCM16LE; it is not a WebSocket server.

## Terminal experience

Use `python run.py --ui` for the guided synthetic experiment and
`python src/infer.py --microphone --ui` for a trained model with a real microphone.
The UI uses Rich on an interactive terminal and preserves plain logs without TTY.
Refresh at most eight times per second outside the audio callback. Show measured
energy, frame time, queue losses, actual text, and model event scores. Never invent
confidence percentages, transcripts, speech quality, or timing results. The UI is
an observer; it must not control model state or silence the user input stream.

## Documentation

Record commands, exit codes, environment, dataset scope, artifact hashes, limitations,
and the next action. Treat a passed engineering test separately from voice quality.
A private tutorial needs an authenticated reader: never bypass access controls or
assume that an unauthenticated external agent can read a private URL.
