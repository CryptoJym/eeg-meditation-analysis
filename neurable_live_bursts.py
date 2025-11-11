#!/usr/bin/env python3
"""
Live Event-Triggered Burst Recording with Neurable

Real-time EEG streaming from Neurable MW75 Neuro headphones
with intelligent burst capture for emotionally significant moments.

Usage:
    # With real Neurable device:
    python neurable_live_bursts.py

    # With mock data (for testing):
    python neurable_live_bursts.py --mock

    # Interactive mode:
    python neurable_live_bursts.py --interactive

Press 't' to manually trigger a burst
Press 'q' to quit
"""

import argparse
import sys
import time
import threading
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False

from neurable_integration import NeurableStream
from burst_record import BurstRecorder
from label_integrator import BurstLabeler


class NeurableLiveBurstRecorder:
    """
    Live burst recording system for Neurable EEG stream.
    """

    def __init__(
        self,
        use_mock: bool = False,
        burst_dir: str = "bursts",
        downsample_to: int = 100,
        auto_label: bool = True
    ):
        """
        Initialize live burst recorder.

        Args:
            use_mock: Use mock data instead of real device
            burst_dir: Directory for burst storage
            downsample_to: Target sampling rate
            auto_label: Automatically label captured bursts
        """
        self.use_mock = use_mock
        self.burst_dir = Path(burst_dir)
        self.burst_dir.mkdir(exist_ok=True)
        self.auto_label = auto_label

        # Initialize Neurable stream
        self.stream = NeurableStream(
            device_name="Neurable",
            downsample_to=downsample_to,
            use_mock=use_mock
        )

        # Initialize burst recorder
        self.recorder = BurstRecorder(
            sampling_rate=downsample_to,
            output_dir=str(self.burst_dir)
        )

        # Initialize labeler
        if auto_label:
            self.labeler = BurstLabeler(burst_dir=str(self.burst_dir))
        else:
            self.labeler = None

        # State tracking
        self.is_running = False
        self.is_monitoring = False
        self.total_samples = 0
        self.bursts_captured = 0
        self.start_time = None
        self.last_heartbeat = 0

        # Keyboard input thread
        self.input_thread = None

    def connect(self, timeout: float = 10.0) -> bool:
        """
        Connect to Neurable device.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if connected
        """
        return self.stream.connect(timeout=timeout)

    def start_monitoring(self):
        """Start monitoring for burst triggers."""
        self.is_monitoring = True
        print("\n🔴 Monitoring started")
        print("   Watching for emotional triggers...")
        print("   Building baseline...")

    def stop_monitoring(self):
        """Stop monitoring."""
        self.is_monitoring = False
        print("\n⏸️  Monitoring stopped")

    def _process_sample(self, sample: np.ndarray):
        """
        Process one EEG sample through the burst system.

        Args:
            sample: EEG sample from Neurable
        """
        if not self.is_monitoring:
            return

        self.total_samples += 1

        # TODO: Implement movement/typing detection
        # For now, assume user is quiet
        is_quiet = True

        # Process through burst recorder
        burst_file = self.recorder.process_sample(sample, is_quiet)

        if burst_file:
            self.bursts_captured += 1
            print(f"\n✅ Burst #{self.bursts_captured} captured!")

            # Auto-label if enabled
            if self.auto_label and self.labeler:
                label = self.labeler.auto_label_burst(Path(burst_file))
                print(f"   🏷️  Auto-labeled: {label['emotion']} ({label['confidence']:.0%} confidence)")
                print(f"   💾 {burst_file}")

        # Send heartbeat every 60 seconds
        current_time = time.time()
        if current_time - self.last_heartbeat >= 60:
            heartbeat = self.recorder.get_heartbeat()
            if heartbeat:
                self._print_heartbeat(heartbeat)
            self.last_heartbeat = current_time

    def _print_heartbeat(self, heartbeat: dict):
        """Print heartbeat status."""
        print(f"\n💓 Heartbeat @ {datetime.now().strftime('%H:%M:%S')}")
        total = sum(heartbeat.values())
        for band, power in heartbeat.items():
            rel = (power / total * 100) if total > 0 else 0
            bar_len = int(rel / 2)
            bar = "█" * bar_len
            print(f"   {band:6s} [{bar:50s}] {rel:5.1f}%")

    def _print_status(self):
        """Print current status."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        rate = self.total_samples / elapsed if elapsed > 0 else 0

        print("\n" + "=" * 60)
        print("STATUS")
        print("=" * 60)
        print(f"Device:        {'Mock' if self.use_mock else 'Neurable MW75 Neuro'}")
        print(f"Connected:     {'✅' if self.stream.is_connected else '❌'}")
        print(f"Monitoring:    {'🔴 Active' if self.is_monitoring else '⚫ Inactive'}")
        print(f"Uptime:        {elapsed:.0f}s")
        print(f"Samples:       {self.total_samples:,} ({rate:.1f} Hz)")
        print(f"Bursts:        {self.bursts_captured}")

        # Show baseline if available
        if self.recorder.baseline_powers:
            print(f"\nBaseline Powers:")
            for band, power in self.recorder.baseline_powers.items():
                print(f"  {band:6s}: {power:8.2f}")

    def manual_trigger(self, label: str = "manual"):
        """
        Manually trigger a burst.

        Args:
            label: Label for manual trigger
        """
        if not self.is_monitoring:
            print("⚠️  Start monitoring first")
            return

        self.recorder.manual_trigger(reason=label)
        print(f"✨ Manual trigger: {label}")

    def run(self):
        """Run live streaming and burst capture."""
        if not self.stream.is_connected:
            print("❌ Not connected to Neurable")
            return

        self.is_running = True
        self.start_time = time.time()

        print("\n" + "=" * 60)
        print("🎧 NEURABLE LIVE BURST RECORDING")
        print("=" * 60)
        print("\nCommands:")
        print("  Press 't' to manually trigger burst")
        print("  Press 's' to show status")
        print("  Press 'h' to show heartbeat")
        print("  Press 'q' to quit")
        print("=" * 60)

        # Start keyboard listener if available
        if KEYBOARD_AVAILABLE:
            self._setup_keyboard_listener()

        # Auto-start monitoring
        self.start_monitoring()

        # Stream and process
        try:
            while self.is_running:
                sample = self.stream.get_sample()
                if sample is not None:
                    self._process_sample(sample)
                else:
                    time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")

        finally:
            self.stop_monitoring()
            self.stream.disconnect()
            self._print_final_summary()

    def _setup_keyboard_listener(self):
        """Setup keyboard hotkey listeners."""
        keyboard.on_press_key('t', lambda _: self.manual_trigger("manual"))
        keyboard.on_press_key('s', lambda _: self._print_status())
        keyboard.on_press_key('h', lambda _: self._print_heartbeat(self.recorder.get_heartbeat()))
        keyboard.on_press_key('q', lambda _: self._quit())

    def _quit(self):
        """Quit the application."""
        self.is_running = False

    def _print_final_summary(self):
        """Print final session summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0

        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Duration:        {elapsed:.1f}s ({elapsed/60:.1f}m)")
        print(f"Total samples:   {self.total_samples:,}")
        print(f"Bursts captured: {self.bursts_captured}")
        print(f"Output dir:      {self.burst_dir}")

        if self.bursts_captured > 0:
            print(f"\n📊 Captured bursts:")
            burst_files = sorted(self.burst_dir.glob("*_emotion_burst.npy"))
            for bf in burst_files[-5:]:  # Show last 5
                print(f"   - {bf.name}")

            if len(burst_files) > 5:
                print(f"   ... and {len(burst_files) - 5} more")

        print("\n✅ Session complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Live event-triggered burst recording with Neurable"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data instead of real Neurable device"
    )
    parser.add_argument(
        "--no-auto-label",
        action="store_true",
        help="Disable automatic burst labeling"
    )
    parser.add_argument(
        "--burst-dir",
        type=str,
        default="bursts",
        help="Directory for burst storage (default: bursts)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Connection timeout in seconds (default: 10)"
    )

    args = parser.parse_args()

    # Check for keyboard module
    if not KEYBOARD_AVAILABLE and not args.mock:
        print("⚠️  Warning: 'keyboard' module not installed")
        print("   Manual triggers won't work. Install with: pip install keyboard")
        print("   Or run with --mock for testing\n")

    # Create recorder
    recorder = NeurableLiveBurstRecorder(
        use_mock=args.mock,
        burst_dir=args.burst_dir,
        auto_label=not args.no_auto_label
    )

    # Connect
    print("🔌 Connecting to Neurable...")
    if not recorder.connect(timeout=args.timeout):
        print("\n❌ Connection failed")
        print("\nTroubleshooting:")
        print("  1. Ensure MW75 Neuro headphones are on and paired")
        print("  2. Open Neurable app to start LSL streaming")
        print("  3. Run with --mock to test without hardware")
        sys.exit(1)

    # Run
    recorder.run()


if __name__ == "__main__":
    main()
