"""Executa o tutorial sintético em pasta isolada e documenta evidências. Python 3.10+."""

import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import queue
import threading

STEPS = [
    ("01-baseline", ["src/baseline.py"], ["baseline.npz"]),
    ("02-fixture", ["src/fixture.py"], ["data/train.jsonl", "data/validation.jsonl"]),
    ("03-tests", ["-m", "pytest", "-q"], []),
    (
        "04-codec",
        [
            "src/train.py",
            "codec",
            "--manifest",
            "data/train.jsonl",
            "--validation",
            "data/validation.jsonl",
            "--steps",
            "100",
        ],
        ["runs/codec.pt", "runs/codec-metrics.json"],
    ),
    (
        "05-reconstruction",
        [
            "src/reconstruct.py",
            "--input",
            "data/validation-user.wav",
            "--output",
            "reconstructed.wav",
        ],
        ["reconstructed.wav"],
    ),
    (
        "06-duplex",
        [
            "src/train.py",
            "duplex",
            "--manifest",
            "data/train.jsonl",
            "--validation",
            "data/validation.jsonl",
            "--steps",
            "100",
        ],
        ["runs/duplex.pt", "runs/duplex-metrics.json"],
    ),
    (
        "07-generation",
        [
            "src/infer.py",
            "--input",
            "data/validation-user.wav",
            "--output",
            "reply.wav",
        ],
        ["reply.wav", "reply.wav.json"],
    ),
    (
        "08-evaluation",
        [
            "src/evaluate.py",
            "--manifest",
            "data/validation.jsonl",
            "--report",
            "reply.wav.json",
        ],
        [],
    ),
]


def digest(path):
    """Compute the SHA-256 digest of a local file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Run the command-line workflow and report its outcome."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", default="work/talkweave-demo")
    p.add_argument("--ui", action="store_true", help="Show an animated terminal on TTY")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Reutilizar somente etapas com evidências intactas",
    )
    args = p.parse_args()
    source = Path(__file__).resolve().parent
    work = Path(args.workspace).resolve()
    if work == source or source.is_relative_to(work):
        p.error(
            "Escolha uma subpasta de trabalho isolada, sem conter o código original"
        )
    missing = [
        n
        for n in ("numpy", "torch", "soundfile", "pytest")
        if importlib.util.find_spec(n) is None
    ]
    if missing:
        p.error(
            "Dependências ausentes: "
            + ", ".join(missing)
            + ". Siga a instalação em README.md primeiro."
        )
    inputs = [
        f
        for f in source.rglob("*")
        if f.is_file()
        and f.suffix in {".py", ".ini"}
        and not any(
            x in {"work", "__pycache__", ".venv"} for x in f.relative_to(source).parts
        )
    ]
    fingerprint = hashlib.sha256(
        "".join(str(f.relative_to(source)) + digest(f) for f in sorted(inputs)).encode()
    ).hexdigest()
    marker = work / ".talkweave-demo-workspace"
    if work.exists() and any(work.iterdir()) and not marker.exists():
        p.error(
            "Pasta já contém arquivos e não pertence a este runner. Escolha outra --workspace."
        )
    if marker.exists() and marker.read_text() != fingerprint:
        p.error(
            "Código mudou. Use outra pasta para não misturar evidências de revisões diferentes."
        )
    work.mkdir(parents=True, exist_ok=True)
    marker.write_text(fingerprint)
    for f in inputs:
        target = work / f.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
    report_path = work / "report.json"
    report = (
        json.loads(report_path.read_text())
        if args.resume and report_path.exists()
        else {"schema_version": 1, "steps": []}
    )
    report.update(
        {
            "source_sha256": fingerprint,
            "python": sys.version,
            "workspace": str(work),
            "scope": "synthetic tones only; no human speech, microphone or GPU benchmark",
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
    )
    logdir = work / "logs"
    logdir.mkdir(exist_ok=True)
    old = {r["id"]: r for r in report["steps"]}
    report["steps"] = []

    def save():
        """Persist execution status and evidence links after each stage."""
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        lines = [
            "# Registro do tutorial",
            "",
            "Escopo: tons sintéticos; não comprova voz humana ou português.",
            "",
            "| Etapa | Estado | Tempo | Evidência |",
            "|---|---|---|---|",
        ]
        for r in report["steps"]:
            lines.append(
                f"| {r['id']} | {r['status']} | {r.get('seconds', 0):.2f}s | [log]({r['log']}) |"
            )
        lines += [
            "",
            "## Qualidade",
            "A execução bem-sucedida comprova o pipeline. Ouça os WAVs e examine a validação; texto vazio ou overfitting devem ser reportados como falhas de qualidade.",
            "",
            "Não testado: áudio humano, microfone físico, AEC, GPU e navegação.",
            "",
            "## Próxima etapa",
            "Leia o capítulo dados e prepare gravações humanas em outra pasta, sem misturar o fixture.",
        ]
        (work / "REPORT.md").write_text("\n".join(lines) + "\n")

    from src.ui import View

    with View(args.ui) as view:
        for id, command, expected in STEPS:
            previous = old.get(id)
            if args.resume and previous and previous["status"] == "passed":
                hashes = previous.get("sha256", {})
                if hashes and all(
                    (work / f).is_file() and digest(work / f) == sha
                    for f, sha in hashes.items()
                ):
                    report["steps"].append(previous)
                    save()
                    print(id + ": evidências conferidas; reutilizado", flush=True)
                    continue
            print(id + ": executando", flush=True)
            start = time.perf_counter()
            log = logdir / (id + ".log")
            try:
                with log.open("w") as output:
                    process = subprocess.Popen(
                        [sys.executable, "-u"] + command,
                        cwd=work,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    lines = queue.Queue()

                    def read_output():
                        """Stream process output into a thread-safe queue."""
                        for line in process.stdout:
                            lines.put(line)

                    thread = threading.Thread(target=read_output, daemon=True)
                    thread.start()
                    latest = ""
                    while (
                        process.poll() is None or thread.is_alive() or not lines.empty()
                    ):
                        if time.perf_counter() - start > 1800:
                            process.kill()
                            process.wait()
                            raise subprocess.TimeoutExpired(command, 1800)
                        try:
                            line = lines.get(timeout=0.1)
                            output.write(line)
                            output.flush()
                            latest = line.strip()
                            if not view.enabled:
                                print(latest, flush=True)
                        except queue.Empty:
                            pass
                        view.stage(id, latest)
                    process.wait()
                    process.stdout.close()
                    result = process
                status = (
                    "passed"
                    if result.returncode == 0
                    and all(
                        (work / f).is_file() and (work / f).stat().st_size
                        for f in expected
                    )
                    else "failed"
                )
                exit_code = result.returncode
            except subprocess.TimeoutExpired:
                status, exit_code = "blocked", None
                with log.open("a") as output:
                    output.write(
                        "\nTIMEOUT: limite de 30 minutos por etapa atingido.\n"
                    )
            artifacts = expected + [str(log.relative_to(work))]
            entry = {
                "id": id,
                "command": [sys.executable, "-u"] + command,
                "status": status,
                "exit_code": exit_code,
                "seconds": time.perf_counter() - start,
                "log": str(log.relative_to(work)),
                "sha256": {
                    f: digest(work / f) for f in artifacts if (work / f).is_file()
                },
            }
            report["steps"].append(entry)
            save()
            if status != "passed":
                print(
                    f"Interrompido: {id}. Consulte {log}; nenhuma etapa posterior foi marcada como concluída."
                )
                return 1
    print(f"Execução concluída. Evidências: {work / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
