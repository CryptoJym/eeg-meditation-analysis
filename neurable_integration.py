#!/usr/bin/env python3
"""
Neurable EEG Integration via Lab Streaming Layer (LSL)

Streams real-time EEG data from Neurable MW75 Neuro headphones
and integrates with the event-triggered burst recording system.

Neurable Specifications:
- 12 EEG channels @ 500 Hz (Research Kit)
- 24-bit precision
- LSL protocol for streaming
- Raw CSV data access

Installation:
    pip install pylsl mne
    # or
    conda install -c conda-forge liblsl

Setup:
    1. Turn on Neurable MW75 Neuro headphones
    2. Open Neurable app to ensure device is streaming
    3. Run this script to connect via LSL
"""

import numpy as np
from typing import Optional, List, Callable, Dict
import time
from datetime import datetime
from collections import deque


# Try to import LSL, provide helpful error if not available
try:
    import pylsl
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False
    print("⚠️  Warning: pylsl not installed. Run: pip install pylsl")


class NeurableStream:
    """
    Streams real-time EEG data from Neurable devices via LSL.
    """

    def __init__(
        self,
        device_name: str = "Neurable",
        downsample_to: int = 100,
        channel_selection: Optional[List[int]] = None,
        use_mock: bool = False
    ):
        """
        Initialize Neurable stream.

        Args:
            device_name: LSL stream name to search for (default: "Neurable")
            downsample_to: Target sampling rate in Hz (default: 100)
            channel_selection: List of channel indices to use (None = all)
            use_mock: Use mock data generator instead of real device
        """
        self.device_name = device_name
        self.downsample_to = downsample_to
        self.channel_selection = channel_selection
        self.use_mock = use_mock

        # Stream properties (will be set on connect)
        self.inlet = None
        self.original_srate = None
        self.n_channels = None
        self.channel_names = None
        self.is_connected = False

        # Downsampling buffer
        self.downsample_ratio = None
        self.sample_buffer = deque()
        self.sample_counter = 0

        # Mock data generator
        self.mock_generator = None

    def connect(self, timeout: float = 10.0) -> bool:
        """
        Connect to Neurable LSL stream.

        Args:
            timeout: Timeout in seconds for finding stream

        Returns:
            True if connected successfully
        """
        if self.use_mock:
            return self._connect_mock()

        if not LSL_AVAILABLE:
            print("❌ Cannot connect: pylsl not installed")
            print("   Install with: pip install pylsl")
            return False

        print(f"🔍 Searching for Neurable device '{self.device_name}'...")
        print(f"   Timeout: {timeout}s")

        # Resolve LSL stream
        streams = pylsl.resolve_byprop(
            'name', self.device_name, timeout=timeout
        )

        if not streams:
            # Try resolving by type if name search fails
            print(f"   No stream named '{self.device_name}' found.")
            print(f"   Searching for EEG streams...")
            streams = pylsl.resolve_byprop('type', 'EEG', timeout=timeout)

        if not streams:
            print(f"❌ No Neurable or EEG streams found")
            print(f"\nTroubleshooting:")
            print(f"  1. Ensure MW75 Neuro headphones are on")
            print(f"  2. Open Neurable app to start streaming")
            print(f"  3. Check that LSL is enabled in device settings")
            print(f"\nOr use mock mode: use_mock=True")
            return False

        # Connect to first available stream
        stream = streams[0]
        self.inlet = pylsl.StreamInlet(stream)

        # Get stream info
        info = self.inlet.info()
        self.original_srate = int(info.nominal_srate())
        self.n_channels = info.channel_count()

        # Extract channel names
        ch = info.desc().child("channels").child("channel")
        self.channel_names = []
        for _ in range(self.n_channels):
            self.channel_names.append(ch.child_value("label"))
            ch = ch.next_sibling()

        # Calculate downsampling ratio
        self.downsample_ratio = self.original_srate // self.downsample_to

        self.is_connected = True

        print(f"✅ Connected to Neurable stream")
        print(f"   Name: {info.name()}")
        print(f"   Type: {info.type()}")
        print(f"   Channels: {self.n_channels}")
        print(f"   Sampling rate: {self.original_srate} Hz")
        print(f"   Downsampling to: {self.downsample_to} Hz (ratio: {self.downsample_ratio})")
        if self.channel_names:
            print(f"   Channel names: {', '.join(self.channel_names[:6])}...")

        return True

    def _connect_mock(self) -> bool:
        """Connect to mock data generator."""
        print("🎮 Mock mode enabled - generating synthetic EEG data")

        # Simulate Neurable specs
        self.original_srate = 500
        self.n_channels = 12
        self.channel_names = [f"Ch{i+1}" for i in range(self.n_channels)]
        self.downsample_ratio = self.original_srate // self.downsample_to

        # Initialize mock generator
        self.mock_generator = MockNeurableGenerator(
            n_channels=self.n_channels,
            srate=self.original_srate
        )

        self.is_connected = True

        print(f"✅ Mock Neurable stream ready")
        print(f"   Channels: {self.n_channels}")
        print(f"   Sampling rate: {self.original_srate} Hz")
        print(f"   Downsampling to: {self.downsample_to} Hz")

        return True

    def get_sample(self) -> Optional[np.ndarray]:
        """
        Get next downsampled EEG sample.

        Returns:
            EEG sample (n_channels,) or None if no data
        """
        if not self.is_connected:
            return None

        # Collect samples until we have enough for downsampling
        while len(self.sample_buffer) < self.downsample_ratio:
            if self.use_mock:
                sample = self.mock_generator.get_sample()
            else:
                sample, timestamp = self.inlet.pull_sample(timeout=0.0)
                if sample is None:
                    return None

            self.sample_buffer.append(sample)

        # Average downsampled samples
        samples = np.array(list(self.sample_buffer)[:self.downsample_ratio])
        self.sample_buffer.clear()

        # Average across downsampling window
        downsampled = np.mean(samples, axis=0)

        # Apply channel selection if specified
        if self.channel_selection:
            downsampled = downsampled[self.channel_selection]

        return downsampled

    def stream(self, callback: Callable[[np.ndarray], None]):
        """
        Stream data continuously and call callback for each sample.

        Args:
            callback: Function to call with each downsampled sample
        """
        if not self.is_connected:
            print("❌ Not connected to Neurable stream")
            return

        print(f"🎧 Streaming started... (Ctrl+C to stop)")

        try:
            while True:
                sample = self.get_sample()
                if sample is not None:
                    callback(sample)
                else:
                    time.sleep(0.001)  # Small delay if no data

        except KeyboardInterrupt:
            print("\n⏹️  Streaming stopped")

    def disconnect(self):
        """Disconnect from stream."""
        if self.inlet:
            self.inlet.close_stream()
        self.is_connected = False
        print("🔌 Disconnected from Neurable")

    def get_info(self) -> Dict:
        """
        Get stream information.

        Returns:
            Dictionary with stream details
        """
        return {
            'connected': self.is_connected,
            'device_name': self.device_name,
            'original_srate': self.original_srate,
            'downsample_srate': self.downsample_to,
            'n_channels': self.n_channels,
            'channel_names': self.channel_names,
            'downsample_ratio': self.downsample_ratio,
            'mock_mode': self.use_mock
        }


class MockNeurableGenerator:
    """
    Generates realistic mock EEG data for testing without hardware.
    Simulates Neurable's 12-channel, 500 Hz stream.
    """

    def __init__(self, n_channels: int = 12, srate: int = 500):
        """
        Initialize mock generator.

        Args:
            n_channels: Number of EEG channels
            srate: Sampling rate in Hz
        """
        self.n_channels = n_channels
        self.srate = srate
        self.t = 0
        self.dt = 1.0 / srate

        # Random phase offsets for each channel
        self.phase_offsets = np.random.rand(n_channels) * 2 * np.pi

        # Base frequencies for each band
        self.delta_freq = np.random.uniform(1, 3)
        self.theta_freq = np.random.uniform(5, 7)
        self.alpha_freq = np.random.uniform(9, 11)
        self.beta_freq = np.random.uniform(18, 25)
        self.gamma_freq = np.random.uniform(35, 45)

    def get_sample(self) -> np.ndarray:
        """
        Generate one multi-channel sample.

        Returns:
            Sample array (n_channels,)
        """
        sample = np.zeros(self.n_channels)

        for ch in range(self.n_channels):
            # Mix of frequency bands with realistic amplitudes
            delta = np.sin(2 * np.pi * self.delta_freq * self.t + self.phase_offsets[ch]) * 15
            theta = np.sin(2 * np.pi * self.theta_freq * self.t + self.phase_offsets[ch]) * 20
            alpha = np.sin(2 * np.pi * self.alpha_freq * self.t + self.phase_offsets[ch]) * 30
            beta = np.sin(2 * np.pi * self.beta_freq * self.t + self.phase_offsets[ch]) * 12
            gamma = np.sin(2 * np.pi * self.gamma_freq * self.t + self.phase_offsets[ch]) * 8

            # Add some noise
            noise = np.random.randn() * 5

            # Combine
            sample[ch] = delta + theta + alpha + beta + gamma + noise

        self.t += self.dt
        return sample


def main():
    """Example usage of Neurable integration."""
    print("=" * 60)
    print("NEURABLE EEG STREAMING TEST")
    print("=" * 60)

    # Create stream (use mock if no device available)
    stream = NeurableStream(
        device_name="Neurable",
        downsample_to=100,
        use_mock=True  # Set to False to use real device
    )

    # Connect
    if not stream.connect(timeout=10):
        print("\n❌ Failed to connect")
        return

    # Display info
    info = stream.get_info()
    print(f"\n📊 Stream Info:")
    for key, value in info.items():
        print(f"   {key}: {value}")

    # Test streaming for 10 seconds
    print(f"\n🧪 Testing stream for 10 seconds...")
    sample_count = 0
    start_time = time.time()

    def count_callback(sample):
        nonlocal sample_count
        sample_count += 1
        if sample_count % 100 == 0:  # Every second at 100 Hz
            elapsed = time.time() - start_time
            print(f"   Received {sample_count} samples in {elapsed:.1f}s")
            print(f"   Current sample: {sample[:3]}... (first 3 channels)")

    try:
        while time.time() - start_time < 10:
            sample = stream.get_sample()
            if sample is not None:
                count_callback(sample)

    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted")

    # Disconnect
    stream.disconnect()

    print(f"\n✅ Test complete!")
    print(f"   Total samples: {sample_count}")
    print(f"   Effective rate: {sample_count / (time.time() - start_time):.1f} Hz")


if __name__ == "__main__":
    main()
