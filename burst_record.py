#!/usr/bin/env python3
"""
Event-Triggered EEG Burst Recording System

Records EEG data only during emotionally significant moments.
Uses adaptive baseline and threshold-based triggering.
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
from collections import deque
from eeg_analysis import EEGAnalyzer


class BurstRecorder:
    """
    Event-triggered EEG burst recorder that captures only meaningful moments.
    """

    def __init__(
        self,
        sampling_rate: int = 100,
        baseline_window: int = 30,
        trigger_threshold: float = 0.15,
        pre_trigger: int = 60,
        post_trigger: int = 60,
        output_dir: str = "bursts"
    ):
        """
        Initialize the burst recorder.

        Args:
            sampling_rate: Sampling frequency in Hz (downsampled from original)
            baseline_window: Rolling baseline window in seconds
            trigger_threshold: Percentage change to trigger recording (0.15 = 15%)
            pre_trigger: Seconds to capture before trigger
            post_trigger: Seconds to capture after trigger
            output_dir: Directory to save burst chunks
        """
        self.sampling_rate = sampling_rate
        self.baseline_window = baseline_window
        self.trigger_threshold = trigger_threshold
        self.pre_trigger = pre_trigger
        self.post_trigger = post_trigger
        self.pre_trigger_samples = pre_trigger * sampling_rate
        self.post_trigger_samples = post_trigger * sampling_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize analyzer for preprocessing
        self.analyzer = EEGAnalyzer(sampling_rate=sampling_rate)

        # Circular buffer for pre-trigger data
        buffer_size = self.pre_trigger_samples
        self.buffer = deque(maxlen=buffer_size)

        # Baseline tracking
        baseline_samples = baseline_window * sampling_rate
        self.baseline_buffer = deque(maxlen=baseline_samples)
        self.baseline_powers = None

        # Recording state
        self.is_recording = False
        self.is_triggered = False
        self.post_trigger_count = 0
        self.current_burst = []
        self.trigger_time = None

        # Band definitions (same as EEGAnalyzer)
        self.bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }

    def _compute_band_powers(self, data: np.ndarray) -> Dict[str, float]:
        """
        Compute band powers for a data segment.

        Args:
            data: EEG data segment

        Returns:
            Dictionary of band powers
        """
        # Quick PSD computation using Welch's method
        nperseg = min(len(data), self.sampling_rate * 2)
        if nperseg < 32:  # Need minimum samples for meaningful FFT
            return {band: 0.0 for band in self.bands.keys()}

        freqs, psd = signal.welch(
            data,
            self.sampling_rate,
            nperseg=nperseg,
            noverlap=nperseg//2
        )

        # Calculate band powers
        band_powers = {}
        for band_name, (low, high) in self.bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            if np.any(band_mask):
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                band_powers[band_name] = band_power
            else:
                band_powers[band_name] = 0.0

        return band_powers

    def _update_baseline(self, data: np.ndarray):
        """
        Update the rolling baseline with new data.

        Args:
            data: New EEG data samples
        """
        self.baseline_buffer.extend(data)

        # Compute baseline band powers if we have enough data
        if len(self.baseline_buffer) >= self.sampling_rate:
            baseline_data = np.array(self.baseline_buffer)
            self.baseline_powers = self._compute_band_powers(baseline_data)

    def _check_trigger(self, data: np.ndarray) -> Tuple[bool, str]:
        """
        Check if current data triggers a burst recording.

        Args:
            data: Recent EEG data window

        Returns:
            (triggered, reason) tuple
        """
        if self.baseline_powers is None or len(data) < self.sampling_rate:
            return False, ""

        # Compute current band powers
        current_powers = self._compute_band_powers(data)

        # Check each band for significant change
        for band_name, current_power in current_powers.items():
            baseline_power = self.baseline_powers[band_name]

            if baseline_power == 0:
                continue

            # Calculate percentage change
            change = abs(current_power - baseline_power) / baseline_power

            if change > self.trigger_threshold:
                direction = "↑" if current_power > baseline_power else "↓"
                reason = f"{band_name} {direction}{change*100:.1f}%"
                return True, reason

        return False, ""

    def _clean_artifacts(self, data: np.ndarray, channels: Optional[List[int]] = None) -> np.ndarray:
        """
        Clean artifacts from multi-channel data.

        For saturated channels (Ch8/Ch9), average neighboring channels.

        Args:
            data: Multi-channel EEG data (channels, samples) or single channel
            channels: List of channel indices to clean (None = auto-detect)

        Returns:
            Cleaned data
        """
        if data.ndim == 1:
            # Single channel - use basic artifact removal
            return self.analyzer._remove_artifacts(data, threshold=3.0)

        # Multi-channel processing
        cleaned = data.copy()
        n_channels = data.shape[0]

        # Detect saturated channels (high variance)
        for ch_idx in range(n_channels):
            ch_data = data[ch_idx]
            variance = np.var(ch_data)
            mean_variance = np.mean([np.var(data[i]) for i in range(n_channels)])

            # If variance is >3σ above mean, it's likely saturated
            if variance > mean_variance * 3:
                # Average neighboring channels
                neighbors = []
                if ch_idx > 0:
                    neighbors.append(data[ch_idx - 1])
                if ch_idx < n_channels - 1:
                    neighbors.append(data[ch_idx + 1])

                if neighbors:
                    cleaned[ch_idx] = np.mean(neighbors, axis=0)

        return cleaned

    def process_sample(self, sample: np.ndarray, is_quiet: bool = True) -> Optional[str]:
        """
        Process a single EEG sample and check for triggers.

        Args:
            sample: EEG sample (single value or multi-channel)
            is_quiet: Whether user is quiet (not typing)

        Returns:
            Burst filename if burst was saved, None otherwise
        """
        # Add to pre-trigger buffer
        self.buffer.append(sample)

        # Update baseline only during quiet periods
        if is_quiet and not self.is_triggered:
            self._update_baseline([sample] if np.isscalar(sample) else sample)

        # If currently recording post-trigger data
        if self.is_triggered:
            self.current_burst.append(sample)
            self.post_trigger_count += 1

            # Check if we've captured enough post-trigger data
            if self.post_trigger_count >= self.post_trigger_samples:
                return self._save_burst()

        # Check for trigger (use last 2 seconds for detection)
        else:
            window_samples = min(2 * self.sampling_rate, len(self.buffer))
            if window_samples >= self.sampling_rate:
                recent_data = np.array(list(self.buffer)[-window_samples:])

                # For multi-channel, use average across channels
                if recent_data.ndim > 1:
                    recent_data = np.mean(recent_data, axis=1)

                triggered, reason = self._check_trigger(recent_data)

                if triggered:
                    self._start_burst(reason)

        return None

    def _start_burst(self, reason: str):
        """
        Start recording a burst.

        Args:
            reason: Trigger reason string
        """
        self.is_triggered = True
        self.post_trigger_count = 0
        self.trigger_time = datetime.now()

        # Initialize burst with pre-trigger data
        self.current_burst = list(self.buffer)

        print(f"\n🔴 BURST TRIGGERED: {reason}")
        print(f"   Time: {self.trigger_time.strftime('%H:%M:%S.%f')[:-3]}")
        print(f"   Capturing {self.pre_trigger}s pre + {self.post_trigger}s post...")

    def _save_burst(self) -> str:
        """
        Save the current burst to disk.

        Returns:
            Filename of saved burst
        """
        # Convert burst to numpy array
        burst_data = np.array(self.current_burst)

        # Clean artifacts
        if burst_data.ndim > 1:
            burst_data = self._clean_artifacts(burst_data)
        else:
            burst_data = self.analyzer._remove_artifacts(burst_data)

        # Generate filename with timestamp
        timestamp = self.trigger_time.strftime('%Y-%m-%dT%H:%M:%S')
        filename = f"{timestamp}_emotion_burst.npy"
        filepath = self.output_dir / filename

        # Save burst data
        np.save(filepath, burst_data)

        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'sampling_rate': self.sampling_rate,
            'duration': len(burst_data) / self.sampling_rate,
            'pre_trigger_samples': self.pre_trigger_samples,
            'post_trigger_samples': self.post_trigger_samples,
            'channels': burst_data.shape[1] if burst_data.ndim > 1 else 1
        }

        metadata_file = self.output_dir / f"{timestamp}_emotion_burst_meta.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Burst saved: {filename}")
        print(f"   Duration: {metadata['duration']:.1f}s")
        print(f"   Samples: {len(burst_data)}")

        # Reset state
        self.is_triggered = False
        self.current_burst = []
        self.trigger_time = None

        return str(filepath)

    def manual_trigger(self, reason: str = "manual"):
        """
        Manually trigger a burst recording.

        Args:
            reason: Reason for manual trigger
        """
        if not self.is_triggered:
            self._start_burst(f"MANUAL: {reason}")

    def get_heartbeat(self) -> Dict[str, float]:
        """
        Get a 5-second heartbeat snapshot of current brain state.

        Returns:
            Dictionary with average PSD across channels
        """
        if len(self.baseline_buffer) < self.sampling_rate * 5:
            return {}

        # Get last 5 seconds
        recent_data = np.array(list(self.baseline_buffer)[-5*self.sampling_rate:])

        # For multi-channel, average across channels
        if recent_data.ndim > 1:
            recent_data = np.mean(recent_data, axis=1)

        # Compute band powers
        return self._compute_band_powers(recent_data)


class StreamingBurstRecorder:
    """
    High-level interface for streaming EEG burst recording.
    """

    def __init__(self, sampling_rate: int = 100, output_dir: str = "bursts"):
        """
        Initialize streaming recorder.

        Args:
            sampling_rate: Sampling frequency in Hz
            output_dir: Output directory for bursts
        """
        self.recorder = BurstRecorder(
            sampling_rate=sampling_rate,
            output_dir=output_dir
        )
        self.is_active = False
        self.heartbeat_interval = 60  # Send heartbeat every 60 seconds
        self.last_heartbeat = 0
        self.sample_count = 0

    def start(self):
        """Start streaming recording."""
        self.is_active = True
        print("🎧 Burst recorder started")
        print("   Waiting for emotional triggers...")

    def stop(self):
        """Stop streaming recording."""
        self.is_active = False
        print("\n⏸️  Burst recorder stopped")

    def process(self, sample: np.ndarray, is_quiet: bool = True) -> Optional[Dict]:
        """
        Process incoming EEG sample.

        Args:
            sample: EEG data sample
            is_quiet: Whether user is quiet (not typing/moving)

        Returns:
            Status dictionary with burst info or heartbeat data
        """
        if not self.is_active:
            return None

        self.sample_count += 1

        # Process sample for burst detection
        burst_file = self.recorder.process_sample(sample, is_quiet)

        if burst_file:
            return {
                'type': 'burst',
                'file': burst_file,
                'timestamp': datetime.now().isoformat()
            }

        # Send heartbeat at regular intervals
        elapsed = self.sample_count / self.recorder.sampling_rate
        if elapsed - self.last_heartbeat >= self.heartbeat_interval:
            heartbeat = self.recorder.get_heartbeat()
            self.last_heartbeat = elapsed

            return {
                'type': 'heartbeat',
                'band_powers': heartbeat,
                'timestamp': datetime.now().isoformat()
            }

        return None


def main():
    """Example usage of burst recorder."""
    print("Event-Triggered EEG Burst Recorder")
    print("=" * 50)

    # Initialize recorder
    recorder = BurstRecorder(sampling_rate=100, output_dir="bursts")

    # Simulate EEG stream with emotional event
    print("\nSimulating EEG stream with emotional trigger...")

    # Generate baseline data (relaxed state)
    duration = 40  # seconds
    sampling_rate = 100
    t = np.linspace(0, duration, duration * sampling_rate)

    # Baseline: low alpha, moderate beta
    baseline = (
        np.sin(2 * np.pi * 2 * t) * 10 +   # Delta
        np.sin(2 * np.pi * 6 * t) * 15 +   # Theta
        np.sin(2 * np.pi * 10 * t) * 20 +  # Alpha
        np.sin(2 * np.pi * 20 * t) * 10 +  # Beta
        np.random.randn(len(t)) * 5        # Noise
    )

    # Add emotional spike at t=30s (gamma surge)
    spike_idx = 30 * sampling_rate
    spike_width = 5 * sampling_rate
    spike = np.exp(-((t - 30) ** 2) / 2) * 50
    emotional_signal = baseline + spike * np.sin(2 * np.pi * 40 * t)

    # Process stream
    for i, sample in enumerate(emotional_signal):
        is_quiet = True  # Assume quiet for demo
        burst_file = recorder.process_sample(sample, is_quiet)

        if burst_file:
            print(f"\n✅ Burst captured and saved!")
            break

    print("\n" + "=" * 50)
    print("Burst recording demonstration complete!")


if __name__ == "__main__":
    main()
