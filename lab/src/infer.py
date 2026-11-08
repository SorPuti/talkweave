"""Implement the infer component of the CPU duplex tutorial."""

import argparse
import json
import queue
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from api import Frame, Result
from ui import View
from core import (
    Codec,
    Duplex,
    FRAME,
    SR,
    CHARS,
    CHAR_ID,
    EVENTS,
    frames,
    read_audio,
    normalize,
    token_text,
)


class Session:
    """Define the Session component contract."""

    def __init__(self, run):
        """Initialize the component and its bounded state."""
        self.codec, self.model = Codec().eval(), Duplex().eval()
        self.codec.load_state_dict(
            torch.load(Path(run) / "codec.pt", weights_only=True, map_location="cpu")
        )
        self.model.load_state_dict(
            torch.load(Path(run) / "duplex.pt", weights_only=True, map_location="cpu")
        )
        with torch.no_grad():
            self.previous_audio = self.codec.encode(torch.zeros(1, FRAME)).view(1, 1)
        self.previous_text = torch.zeros(1, 1, dtype=torch.long)
        self.hidden = None
        self.last_seq = -1

    @torch.inference_mode()
    def step(self, samples, forced_text=None):
        """Generate one audio frame while preserving recurrent state."""
        u = self.codec.encode(
            torch.tensor(samples, dtype=torch.float32).view(1, FRAME)
        ).view(1, 1)
        override = None if forced_text is None else torch.tensor([[forced_text]])
        a, ut, at, events, self.hidden = self.model(
            u,
            self.previous_audio,
            self.previous_text,
            self.hidden,
            teacher_text=override,
        )
        self.previous_audio = a.argmax(-1)
        self.previous_text = at.argmax(-1) if override is None else override
        audio = self.codec.decode(self.previous_audio).view(FRAME).numpy()
        return (
            audio,
            int(ut.argmax(-1).item()),
            int(self.previous_text.item()),
            events.sigmoid().view(-1).tolist(),
        )

    def process(self, frame: Frame) -> Result:
        """Validate and process one frame through the engine contract."""
        if (
            frame.rate != SR
            or frame.samples.shape != (FRAME,)
            or frame.samples.dtype != np.float32
            or not np.isfinite(frame.samples).all()
        ):
            raise ValueError(
                "Expected one finite mono frame at 16 kHz with 320 samples"
            )
        if frame.seq <= self.last_seq:
            raise ValueError("Frame sequence must increase within the session")
        self.last_seq = frame.seq
        start = time.perf_counter()
        audio, user, agent, events = self.step(frame.samples)
        return Result(
            audio,
            CHARS[user],
            CHARS[agent],
            dict(zip(EVENTS, events)),
            (time.perf_counter() - start) * 1000,
        )


def main():
    """Run the command-line workflow and report its outcome."""
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs")
    p.add_argument("--input")
    p.add_argument("--output", default="reply.wav")
    p.add_argument("--microphone", action="store_true")
    p.add_argument(
        "--ui",
        action="store_true",
        help="Show an animated terminal with measured metrics",
    )
    p.add_argument(
        "--text", help="Condicionamento experimental: texto forçado, não um TTS pronto"
    )
    p.add_argument("--seconds", type=int, default=15)
    args = p.parse_args()
    torch.set_num_threads(2)
    s = Session(args.run)
    if args.microphone:
        # Callback nunca executa rede neural nem espera disco/GPU.
        import sounddevice as sd

        incoming, outgoing = queue.Queue(maxsize=8), queue.Queue(maxsize=8)
        counters = {"input_overflow": 0, "output_underrun": 0, "portaudio_status": 0}

        capture_seq = [0]

        def callback(indata, outdata, count, timing, status):
            """Transfer audio through bounded queues without running inference."""
            if status:
                counters["portaudio_status"] += 1
            try:
                incoming.put_nowait((capture_seq[0], indata[:, 0].copy()))
            except queue.Full:
                counters["input_overflow"] += 1
            capture_seq[0] += 1
            try:
                outdata[:, 0] = outgoing.get_nowait()
            except queue.Empty:
                outdata.fill(0)
                counters["output_underrun"] += 1

        print("Use fones. Esta captura local NÃO implementa cancelamento de eco.")
        try:
            with (
                View(args.ui) as view,
                sd.Stream(
                    samplerate=SR, blocksize=FRAME, channels=1, callback=callback
                ),
            ):
                while True:
                    seq, samples = incoming.get(timeout=5)
                    frame = Frame(samples, seq=seq)
                    result = s.process(frame)
                    outgoing.put(result.audio, timeout=1)
                    view.feed(frame, result, counters)
                    if not view.enabled and (result.user or result.agent):
                        print(
                            json.dumps(
                                {
                                    "user": result.user,
                                    "agent": result.agent,
                                    "events": result.events,
                                },
                                ensure_ascii=False,
                            )
                        )
        except KeyboardInterrupt:
            print(counters)
        return
    if not args.input and not args.text:
        p.error("Informe --input, --microphone ou --text")
    x = (
        read_audio(args.input)
        if args.input
        else np.zeros(args.seconds * SR, dtype="float32")
    )
    prompt = []
    if args.text:
        for char in normalize(args.text):
            if char not in CHAR_ID:
                p.error(f"Caractere fora do vocabulário: {char!r}")
            # Uma letra a cada três quadros: relógio didático, não alinhador aprendido.
            prompt.extend([CHAR_ID[char], 0, 0])
        if len(prompt) > len(frames(x)):
            p.error("Aumente --seconds para caber o texto")
    output, user, agent, events, latency = [], [], [], [], []
    for i, f in enumerate(frames(x)):
        start = time.perf_counter()
        a, u, t, e = s.step(
            f.numpy(), prompt[i] if i < len(prompt) else (0 if args.text else None)
        )
        latency.append((time.perf_counter() - start) * 1000)
        output.append(a)
        user.append(u)
        agent.append(t)
        events.append(e)
    sf.write(args.output, np.concatenate(output), SR)
    report = {
        "user_text": token_text(user),
        "agent_text": token_text(agent),
        "frame_p50_ms": float(np.percentile(latency, 50)),
        "frame_p95_ms": float(np.percentile(latency, 95)),
        "rtf": sum(latency) / (len(x) / SR * 1000),
        "event_probabilities": events,
    }
    Path(args.output + ".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "event_probabilities"},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
