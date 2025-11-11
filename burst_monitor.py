#!/usr/bin/env python3
"""
Interactive Burst Monitor CLI

Provides manual control over EEG burst recording with simple commands:
- on: Start burst recording
- off: Stop burst recording
- trigger <emotion>: Manually trigger burst with label
- status: Show current recording status
- heartbeat: Get current brain state snapshot
- list: List recorded bursts
- label <file> <emotion> [notes]: Manually label a burst
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from burst_record import BurstRecorder
from label_integrator import BurstLabeler, EmotionClassifier
from generate_sample_eeg import EEGDataGenerator
import time


class BurstMonitor:
    """
    Interactive monitor for burst recording system.
    """

    def __init__(self, sampling_rate: int = 100, burst_dir: str = "bursts"):
        """
        Initialize burst monitor.

        Args:
            sampling_rate: EEG sampling rate in Hz
            burst_dir: Directory for burst storage
        """
        self.sampling_rate = sampling_rate
        self.burst_dir = Path(burst_dir)
        self.burst_dir.mkdir(exist_ok=True)

        self.recorder = BurstRecorder(
            sampling_rate=sampling_rate,
            output_dir=str(self.burst_dir)
        )
        self.labeler = BurstLabeler(burst_dir=str(self.burst_dir))

        self.is_running = False
        self.is_recording = False
        self.current_burst_start = None

        # For simulation
        self.generator = EEGDataGenerator(sampling_rate=sampling_rate)
        self.simulation_data = None
        self.simulation_idx = 0

    def start(self):
        """Start the monitor."""
        self.is_running = True
        print("\n" + "=" * 60)
        print("🎧 EEG BURST MONITOR")
        print("=" * 60)
        print("\nCommands:")
        print("  on              - Start monitoring for bursts")
        print("  off             - Stop monitoring")
        print("  trigger <label> - Manually trigger burst (e.g., 'trigger fear')")
        print("  status          - Show current status")
        print("  heartbeat       - Get current brain state snapshot")
        print("  list [emotion]  - List recorded bursts (optionally filter)")
        print("  label <file> <emotion> [notes] - Manually label a burst")
        print("  sim             - Start simulation mode")
        print("  quit            - Exit monitor")
        print("\nType 'help' for detailed command info")
        print("=" * 60)

    def cmd_on(self):
        """Start burst recording."""
        if self.is_recording:
            print("⚠️  Already recording!")
            return

        self.is_recording = True
        self.current_burst_start = datetime.now()
        print("🔴 Recording ON - Monitoring for emotional triggers...")

    def cmd_off(self):
        """Stop burst recording."""
        if not self.is_recording:
            print("⚠️  Not currently recording!")
            return

        self.is_recording = False
        duration = (datetime.now() - self.current_burst_start).total_seconds()
        print(f"⏸️  Recording OFF - Monitored for {duration:.1f}s")

    def cmd_trigger(self, emotion: str = "manual"):
        """
        Manually trigger a burst.

        Args:
            emotion: Emotion label for manual trigger
        """
        if not self.is_recording:
            print("⚠️  Start recording first with 'on' command")
            return

        self.recorder.manual_trigger(reason=emotion)
        print(f"✨ Manual trigger activated: {emotion}")

    def cmd_status(self):
        """Show current recording status."""
        print("\n" + "=" * 40)
        print("STATUS")
        print("=" * 40)
        print(f"Monitor:    {'🟢 Running' if self.is_running else '🔴 Stopped'}")
        print(f"Recording:  {'🔴 Active' if self.is_recording else '⚫ Inactive'}")

        if self.is_recording and self.current_burst_start:
            duration = (datetime.now() - self.current_burst_start).total_seconds()
            print(f"Duration:   {duration:.1f}s")

        # Count bursts
        burst_files = list(self.burst_dir.glob("*_emotion_burst.npy"))
        print(f"Bursts:     {len(burst_files)} saved")

        # Show baseline status
        if self.recorder.baseline_powers:
            print("\nBaseline Powers:")
            for band, power in self.recorder.baseline_powers.items():
                print(f"  {band:6s}: {power:8.2f}")

    def cmd_heartbeat(self):
        """Get current brain state heartbeat."""
        heartbeat = self.recorder.get_heartbeat()

        if not heartbeat:
            print("⚠️  Not enough data for heartbeat (need 5+ seconds)")
            return

        print("\n" + "=" * 40)
        print("💓 HEARTBEAT - Current Brain State")
        print("=" * 40)

        # Calculate total power
        total = sum(heartbeat.values())

        for band, power in heartbeat.items():
            rel_power = (power / total * 100) if total > 0 else 0
            bar_len = int(rel_power / 2)
            bar = "█" * bar_len
            print(f"{band:6s} [{bar:50s}] {rel_power:5.1f}%")

    def cmd_list(self, emotion_filter: str = None):
        """
        List recorded bursts.

        Args:
            emotion_filter: Optional emotion to filter by
        """
        bursts = self.labeler.list_labeled_bursts(emotion_filter=emotion_filter)

        if not bursts:
            print(f"\n⚠️  No bursts found" + (f" with emotion '{emotion_filter}'" if emotion_filter else ""))
            return

        print("\n" + "=" * 40)
        print(f"RECORDED BURSTS" + (f" (emotion: {emotion_filter})" if emotion_filter else ""))
        print("=" * 40)

        for burst in bursts:
            filename = burst['filename']
            emotion = burst['effective_emotion']
            timestamp = burst['timestamp']
            label_type = "📝" if burst.get('manual_override') else "🤖"

            print(f"\n{label_type} {filename}")
            print(f"   Emotion:   {emotion}")
            print(f"   Time:      {timestamp}")

            if burst.get('confidence'):
                print(f"   Confidence: {burst['confidence']:.2f}")

    def cmd_label(self, filename: str, emotion: str, *notes_parts):
        """
        Manually label a burst.

        Args:
            filename: Burst filename
            emotion: Emotion label
            notes_parts: Optional notes (multiple words)
        """
        burst_file = self.burst_dir / filename
        notes = " ".join(notes_parts) if notes_parts else ""

        if not burst_file.exists():
            print(f"⚠️  Burst file not found: {filename}")
            return

        self.labeler.manual_label(burst_file, emotion, notes)
        print(f"✅ Label applied: {emotion}")

    def cmd_sim(self):
        """Start simulation mode with generated data."""
        print("\n🎮 Starting simulation mode...")
        print("   Generating EEG data with emotional event at t=30s...")

        # Generate 60 seconds of data with emotion at 30s
        duration = 60
        self.simulation_data = self._generate_emotional_simulation(duration)
        self.simulation_idx = 0

        print("   Simulation ready. Type 'on' to start monitoring.")

    def _generate_emotional_simulation(self, duration: int) -> np.ndarray:
        """
        Generate simulated EEG data with emotional event.

        Args:
            duration: Duration in seconds

        Returns:
            Simulated EEG array
        """
        t = np.linspace(0, duration, duration * self.sampling_rate)

        # Baseline: relaxed state
        baseline = (
            np.sin(2 * np.pi * 2 * t) * 10 +   # Delta
            np.sin(2 * np.pi * 6 * t) * 15 +   # Theta
            np.sin(2 * np.pi * 10 * t) * 25 +  # Alpha (high - rest)
            np.sin(2 * np.pi * 20 * t) * 8 +   # Beta (low)
            np.random.randn(len(t)) * 5        # Noise
        )

        # Emotional spike at t=30s (gamma + beta surge)
        spike_start = 30
        spike_width = 5
        spike = np.exp(-((t - spike_start) ** 2) / (2 * spike_width ** 2)) * 80

        # Add spike components
        gamma_spike = spike * np.sin(2 * np.pi * 40 * t)  # Gamma
        beta_spike = spike * 0.6 * np.sin(2 * np.pi * 25 * t)  # Beta

        emotional_signal = baseline + gamma_spike + beta_spike

        return emotional_signal

    def _run_simulation_step(self):
        """Process one step of simulation."""
        if self.simulation_data is None or self.simulation_idx >= len(self.simulation_data):
            return False

        # Get current sample
        sample = self.simulation_data[self.simulation_idx]
        self.simulation_idx += 1

        # Process through recorder
        is_quiet = True  # Assume quiet for simulation
        burst_file = self.recorder.process_sample(sample, is_quiet)

        # If burst was captured, auto-label it
        if burst_file:
            burst_path = Path(burst_file)
            label = self.labeler.auto_label_burst(burst_path)
            print(f"\n🏷️  Auto-labeled: {label['emotion']} (confidence: {label['confidence']:.2f})")

        # Print progress
        current_time = self.simulation_idx / self.sampling_rate
        if self.simulation_idx % (self.sampling_rate * 5) == 0:  # Every 5 seconds
            print(f"   t={current_time:.0f}s", end="\r")

        return True

    def run_command(self, command: str):
        """
        Process a command.

        Args:
            command: Command string
        """
        parts = command.strip().split()
        if not parts:
            return

        cmd = parts[0].lower()
        args = parts[1:]

        # Command routing
        if cmd == "on":
            self.cmd_on()
        elif cmd == "off":
            self.cmd_off()
        elif cmd == "trigger":
            emotion = args[0] if args else "manual"
            self.cmd_trigger(emotion)
        elif cmd == "status":
            self.cmd_status()
        elif cmd == "heartbeat":
            self.cmd_heartbeat()
        elif cmd == "list":
            emotion = args[0] if args else None
            self.cmd_list(emotion)
        elif cmd == "label":
            if len(args) < 2:
                print("⚠️  Usage: label <filename> <emotion> [notes]")
            else:
                self.cmd_label(args[0], args[1], *args[2:])
        elif cmd == "sim":
            self.cmd_sim()
        elif cmd == "help":
            self.start()  # Re-print help
        elif cmd in ["quit", "exit", "q"]:
            return False
        else:
            print(f"⚠️  Unknown command: {cmd}")
            print("   Type 'help' for command list")

        return True

    def run_interactive(self):
        """Run interactive command loop."""
        self.start()

        while self.is_running:
            try:
                # Check if in simulation mode and recording
                if self.is_recording and self.simulation_data is not None:
                    # Run simulation step
                    if not self._run_simulation_step():
                        print("\n✅ Simulation complete!")
                        self.is_recording = False
                        self.simulation_data = None
                        self.simulation_idx = 0
                    time.sleep(1.0 / self.sampling_rate)  # Simulate real-time
                    continue

                # Get command from user
                command = input("\n> ").strip()

                if not self.run_command(command):
                    break

            except KeyboardInterrupt:
                print("\n\n👋 Exiting monitor...")
                break
            except Exception as e:
                print(f"⚠️  Error: {e}")

        print("\n" + "=" * 60)
        print("Monitor stopped.")


def main():
    """Run the burst monitor."""
    monitor = BurstMonitor(sampling_rate=100, burst_dir="bursts")
    monitor.run_interactive()


if __name__ == "__main__":
    main()
