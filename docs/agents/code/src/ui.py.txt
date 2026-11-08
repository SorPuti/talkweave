"""Implement the ui component of the CPU duplex tutorial."""

import sys
import time
from collections import deque
import numpy as np


class View:
    """Define the View component contract."""

    def __init__(self, enabled=False):
        """Initialize the component and its bounded state."""
        self.enabled = enabled and sys.stdout.isatty()
        self.live = None
        self.started = time.perf_counter()
        self.durations = deque(maxlen=300)
        self.user = ""
        self.agent = ""
        self.next_paint = 0.0

    def __enter__(self):
        """Start the optional live terminal presentation."""
        if self.enabled:
            try:
                from rich.live import Live

                self.live = Live(
                    "Starting TalkWeave…", refresh_per_second=8, transient=False
                )
                self.live.start()
            except ImportError:
                self.enabled = False
                print(
                    "Live UI unavailable; install rich or use plain logs.",
                    file=sys.stderr,
                )
        return self

    def __exit__(self, *args):
        """Close the live terminal without discarding measurements."""
        if self.live:
            self.live.stop()

    def stage(self, name, detail=""):
        """Render the current stage and its latest real log line."""
        if not self.live:
            return
        from rich.panel import Panel
        from rich.text import Text
        from rich.console import Group
        from rich.spinner import Spinner

        body = Text(
            f"{name}\n\n{detail[-900:]}\n\nElapsed: {time.perf_counter() - self.started:.1f}s",
            style="green",
        )
        self.live.update(
            Panel(
                Group(Spinner("dots", text="Processing real output…"), body),
                title="TALKWEAVE / BUILD",
                border_style="green",
            )
        )

    def feed(self, frame, result, counters):
        """Render measured audio and timing outside the capture callback."""
        self.durations.append(result.elapsed_ms)
        self.user = (self.user + result.user)[-180:]
        self.agent = (self.agent + result.agent)[-180:]
        now = time.perf_counter()
        if not self.live or now < self.next_paint:
            return
        self.next_paint = now + 0.125
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        samples = frame.samples
        energy = np.sqrt(np.mean(samples * samples) + 1e-12)
        alphabet = "▁▂▃▄▅▆▇█"
        wave = "".join(
            alphabet[min(7, int(float(np.max(np.abs(x))) * 16))]
            for x in np.array_split(samples, 48)
        )
        table = Table(expand=True, border_style="dim green")
        table.add_column("Measured signal")
        table.add_column("Value", justify="right")
        table.add_row("Input energy", f"{20 * np.log10(energy):.1f} dBFS")
        table.add_row(
            "Inference p95 / frame budget",
            f"{np.percentile(self.durations, 95):.2f} / 20 ms",
        )
        table.add_row(
            "Input overflow / output underrun",
            f"{counters['input_overflow']} / {counters['output_underrun']}",
        )
        for event, score in result.events.items():
            table.add_row(f"{event} · model score", f"{score:.3f}")
        panel = Group(
            Text(wave, style="green"),
            table,
            Text("USER  " + self.user),
            Text("AGENT " + self.agent),
            Text(
                "Scores are not calibrated confidence. Voice quality depends on training.",
                style="dim",
            ),
        )
        self.live.update(
            Panel(
                panel, title="TALKWEAVE / LIVE · 16 kHz · 20 ms", border_style="green"
            )
        )
