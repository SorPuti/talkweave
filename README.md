# TalkWeave

**Build your own full-duplex voice engine.** A practical tutorial in Portuguese with complete Python code, interactive learning tools, agent instructions and a Rust portability starter.

- **[Open the tutorial](https://sorputi.github.io/talkweave/)**
- **[Entry point for AI agents](https://sorputi.github.io/talkweave/agents/)**
- **[Download the runnable lab](https://sorputi.github.io/talkweave/downloads/talkweave.zip)**
- **[Offline manual](https://sorputi.github.io/talkweave/downloads/talkweave-tutorial.md)**

## Run locally

```bash
cd lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python run.py --ui
```

Without an interactive terminal, use `python run.py`. Resume verified stages with `python run.py --resume`. Outputs and evidence are written to `lab/work/talkweave-demo/` when launched from lab/.

The demo includes synthetic tones, not a pretrained conversational voice. A successful run proves the engineering pipeline, not human speech quality. Read the tutorial before replacing the fixture with recordings.

## Repository

| Path | Purpose |
|---|---|
| `lab/` | Executable CPU lab: source, tests, agent rules and training |
| `lab/rs/` | Rust contracts and PCM protocol starter, not a native neural engine |
| `web/` | Website source and recorded synthetic experiment assets |
| `docs/` | Generated GitHub Pages site, downloads and raw agent documentation |
| `build.py` | Standard-library static build and ZIP packaging |
| `agents.py`, `ports.py` | Agent guides, hashes, language profiles and skill generation |

## Build the website

```bash
python3 build.py
python3 tools/check.py
python3 -m http.server 8000 --directory docs
```

Open http://localhost:8000/. GitHub Pages serves `main:/docs`; `.nojekyll` keeps the static assets and downloads intact. No API keys, model weights or user recordings are required to build the documentation.

## Verification and scope

Python lab: nine tests passed in the original CPU environment, including causality, streaming equivalence, protocol, frame validation and dropout train/eval behavior. The notebook-scale model still overfits the synthetic fixture. Microphone hardware, GPU performance and human voice accuracy were not verified. Rust compiler/tests were not available in the original environment and remain explicitly unverified.

Optional activation dropout: `--dropout 0.1` in the duplex training stage; default 0. Dropout is disabled during inference and is not a guarantee against overfitting.

## License

Original tutorial and code: MIT. Referenced models, books and datasets retain their own terms. No third-party model weights are redistributed.

## Publish to GitHub Pages

With the empty public `SorPuti/talkweave` repository available and the intended GitHub account authenticated:

```bash
python3 tools/publish.py
```

The helper builds, checks links and package hashes, pushes the source, then requests `main:/docs` Pages hosting. If the authenticated integration lacks Pages permission, it prints the exact settings URL and manual configuration. It does not claim a live website before GitHub finishes deploying.
