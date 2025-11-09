#!/usr/bin/env python3
"""
Generate Sample EEG Data for Testing
Creates realistic EEG data simulating different meditation states.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple


class EEGDataGenerator:
    """
    Generate synthetic EEG data for testing meditation analysis.
    """

    def __init__(self, sampling_rate: int = 256):
        """
        Initialize the EEG data generator.

        Args:
            sampling_rate: Sampling frequency in Hz
        """
        self.sampling_rate = sampling_rate

        # Typical amplitude ranges for each band (in microvolts)
        self.amplitude_ranges = {
            'Delta': (20, 60),   # 0.5-4 Hz
            'Theta': (5, 30),    # 4-8 Hz
            'Alpha': (10, 50),   # 8-13 Hz
            'Beta': (5, 20),     # 13-30 Hz
            'Gamma': (2, 10)     # 30-50 Hz
        }

        # Center frequencies for each band
        self.center_frequencies = {
            'Delta': 2.0,
            'Theta': 6.0,
            'Alpha': 10.0,
            'Beta': 20.0,
            'Gamma': 40.0
        }

    def generate_band_signal(self, duration: float, band: str,
                            amplitude: float, frequency_variation: float = 0.5) -> np.ndarray:
        """
        Generate signal for a specific frequency band.

        Args:
            duration: Signal duration in seconds
            band: Frequency band name
            amplitude: Signal amplitude in microvolts
            frequency_variation: Frequency variation around center frequency

        Returns:
            Generated signal
        """
        samples = int(duration * self.sampling_rate)
        time = np.arange(samples) / self.sampling_rate

        # Base frequency with slight variation
        base_freq = self.center_frequencies[band]
        freq_modulation = np.sin(2 * np.pi * 0.1 * time) * frequency_variation

        # Generate signal with frequency modulation
        phase = 2 * np.pi * base_freq * time
        signal = amplitude * np.sin(phase + freq_modulation)

        # Add harmonics for more realistic signal
        if band in ['Alpha', 'Beta']:
            signal += amplitude * 0.3 * np.sin(2 * phase)
            signal += amplitude * 0.1 * np.sin(3 * phase)

        return signal

    def generate_meditation_progression(self, duration: float,
                                      initial_state: str = "alert",
                                      target_state: str = "deep_meditation") -> np.ndarray:
        """
        Generate EEG data showing progression from one state to another.

        Args:
            duration: Total duration in seconds
            initial_state: Starting mental state
            target_state: Target mental state

        Returns:
            Generated EEG signal
        """
        samples = int(duration * self.sampling_rate)
        time = np.arange(samples) / self.sampling_rate
        progress = time / duration  # 0 to 1 progression

        # State profiles (relative power for each band)
        states = {
            'alert': {
                'Delta': 0.05,
                'Theta': 0.10,
                'Alpha': 0.25,
                'Beta': 0.45,
                'Gamma': 0.15
            },
            'relaxed': {
                'Delta': 0.10,
                'Theta': 0.15,
                'Alpha': 0.40,
                'Beta': 0.25,
                'Gamma': 0.10
            },
            'light_meditation': {
                'Delta': 0.15,
                'Theta': 0.25,
                'Alpha': 0.35,
                'Beta': 0.20,
                'Gamma': 0.05
            },
            'deep_meditation': {
                'Delta': 0.30,
                'Theta': 0.35,
                'Alpha': 0.20,
                'Beta': 0.10,
                'Gamma': 0.05
            }
        }

        initial_profile = states[initial_state]
        target_profile = states[target_state]

        # Initialize combined signal
        eeg_signal = np.zeros(samples)

        # Generate and combine band signals with progression
        for band in self.amplitude_ranges.keys():
            # Interpolate amplitude between states
            initial_amp = initial_profile[band] * 100
            target_amp = target_profile[band] * 100
            amplitude = initial_amp + (target_amp - initial_amp) * progress

            # Generate band signal
            band_signal = self.generate_band_signal(duration, band, 1.0)

            # Apply time-varying amplitude
            band_signal = band_signal * amplitude

            # Add to combined signal
            eeg_signal += band_signal

        # Add realistic noise
        pink_noise = self.generate_pink_noise(samples) * 5
        eeg_signal += pink_noise

        # Add occasional artifacts
        eeg_signal = self.add_artifacts(eeg_signal, artifact_probability=0.001)

        return eeg_signal

    def generate_pink_noise(self, samples: int) -> np.ndarray:
        """
        Generate pink (1/f) noise for more realistic EEG.

        Args:
            samples: Number of samples

        Returns:
            Pink noise signal
        """
        # Generate white noise
        white = np.random.randn(samples)

        # Apply 1/f filter in frequency domain
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(samples)
        # Avoid division by zero
        freqs[0] = 1
        fft = fft / np.sqrt(freqs)
        pink = np.fft.irfft(fft, samples)

        return pink

    def add_artifacts(self, signal: np.ndarray,
                     artifact_probability: float = 0.001) -> np.ndarray:
        """
        Add realistic artifacts to EEG signal.

        Args:
            signal: Clean EEG signal
            artifact_probability: Probability of artifact at each sample

        Returns:
            Signal with artifacts
        """
        signal = signal.copy()

        # Eye blink artifacts (large amplitude, low frequency)
        blink_mask = np.random.random(len(signal)) < artifact_probability / 10
        blink_indices = np.where(blink_mask)[0]

        for idx in blink_indices:
            duration = int(0.2 * self.sampling_rate)  # 200ms blink
            if idx + duration < len(signal):
                blink = np.sin(np.linspace(0, np.pi, duration)) * 100
                signal[idx:idx+duration] += blink

        # Muscle artifacts (high frequency, moderate amplitude)
        muscle_mask = np.random.random(len(signal)) < artifact_probability
        muscle_noise = np.random.randn(np.sum(muscle_mask)) * 30
        signal[muscle_mask] += muscle_noise

        return signal

    def generate_multiple_channels(self, duration: float,
                                  num_channels: int = 8,
                                  state: str = "deep_meditation") -> np.ndarray:
        """
        Generate multi-channel EEG data.

        Args:
            duration: Signal duration in seconds
            num_channels: Number of EEG channels
            state: Mental state to simulate

        Returns:
            Multi-channel EEG data (channels x samples)
        """
        samples = int(duration * self.sampling_rate)
        channels = []

        # Channel correlations (frontal channels more correlated)
        for i in range(num_channels):
            if i == 0:
                # Base channel
                channel = self.generate_meditation_progression(
                    duration, "relaxed", state
                )
            else:
                # Correlated channels with some independence
                base = channels[0] if i < 4 else channels[4] if i >= 4 else channels[0]
                correlation = 0.7 if i < 4 else 0.5
                independent = self.generate_meditation_progression(
                    duration, "relaxed", state
                )
                channel = correlation * base + (1 - correlation) * independent

            channels.append(channel)

        return np.array(channels)

    def save_eeg_data(self, eeg_data: np.ndarray,
                     metadata: Dict,
                     filename: str = "sample_eeg_data.npz"):
        """
        Save EEG data with metadata.

        Args:
            eeg_data: EEG signal data
            metadata: Recording metadata
            filename: Output filename
        """
        np.savez_compressed(
            filename,
            eeg_data=eeg_data,
            sampling_rate=self.sampling_rate,
            metadata=json.dumps(metadata)
        )
        print(f"EEG data saved to {filename}")

    def load_eeg_data(self, filename: str) -> Tuple[np.ndarray, Dict]:
        """
        Load EEG data and metadata.

        Args:
            filename: Input filename

        Returns:
            EEG data and metadata
        """
        data = np.load(filename)
        eeg_data = data['eeg_data']
        metadata = json.loads(str(data['metadata']))
        return eeg_data, metadata

    def generate_session_dataset(self, session_duration: float = 600,
                                num_subjects: int = 5) -> pd.DataFrame:
        """
        Generate a dataset with multiple meditation sessions.

        Args:
            session_duration: Duration of each session in seconds
            num_subjects: Number of subjects

        Returns:
            DataFrame with session data
        """
        sessions = []

        for subject_id in range(1, num_subjects + 1):
            # Vary meditation skill level
            skill_level = np.random.choice(['beginner', 'intermediate', 'advanced'])

            # Generate progression based on skill
            if skill_level == 'beginner':
                states = [('alert', 'relaxed'), ('relaxed', 'light_meditation')]
            elif skill_level == 'intermediate':
                states = [('relaxed', 'light_meditation'), ('light_meditation', 'deep_meditation')]
            else:
                states = [('relaxed', 'deep_meditation'), ('light_meditation', 'deep_meditation')]

            for session_num, (initial, target) in enumerate(states, 1):
                # Generate EEG data
                eeg_data = self.generate_meditation_progression(
                    session_duration, initial, target
                )

                # Create metadata
                metadata = {
                    'subject_id': f"S{subject_id:03d}",
                    'session_number': session_num,
                    'timestamp': (datetime.now() - timedelta(days=subject_id, hours=session_num)).isoformat(),
                    'duration': session_duration,
                    'skill_level': skill_level,
                    'initial_state': initial,
                    'target_state': target,
                    'sampling_rate': self.sampling_rate,
                    'channels': ['Fz'],  # Single channel for simplicity
                    'notes': f"Synthetic data for {skill_level} practitioner"
                }

                # Save individual session
                filename = f"eeg_session_{metadata['subject_id']}_session{session_num}.npz"
                self.save_eeg_data(eeg_data, metadata, filename)

                # Add to sessions list
                sessions.append(metadata)

        # Create DataFrame
        df = pd.DataFrame(sessions)
        df.to_csv('meditation_sessions_metadata.csv', index=False)
        print(f"Metadata saved to meditation_sessions_metadata.csv")

        return df

    def visualize_generated_data(self, eeg_data: np.ndarray, title: str = "Generated EEG Data"):
        """
        Visualize generated EEG data.

        Args:
            eeg_data: EEG signal to visualize
            title: Plot title
        """
        if len(eeg_data.shape) == 1:
            # Single channel
            channels_data = [eeg_data]
            num_channels = 1
        else:
            # Multi-channel
            channels_data = eeg_data
            num_channels = len(eeg_data)

        fig, axes = plt.subplots(num_channels, 1, figsize=(15, 2*num_channels))
        if num_channels == 1:
            axes = [axes]

        time = np.arange(len(channels_data[0])) / self.sampling_rate

        for i, (ax, channel) in enumerate(zip(axes, channels_data)):
            # Show first 10 seconds
            display_samples = min(10 * self.sampling_rate, len(channel))
            ax.plot(time[:display_samples], channel[:display_samples],
                   linewidth=0.5, color='blue')
            ax.set_ylabel(f'Ch {i+1}\n(μV)')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-100, 100)

            if i == 0:
                ax.set_title(title)
            if i == num_channels - 1:
                ax.set_xlabel('Time (seconds)')

        plt.tight_layout()
        plt.show()


def main():
    """
    Generate sample EEG datasets for testing.
    """
    print("=" * 60)
    print("EEG Data Generator for Meditation Analysis")
    print("=" * 60)
    print()

    generator = EEGDataGenerator(sampling_rate=256)

    # 1. Generate short sample for quick testing
    print("1. Generating 1-minute quick test sample...")
    quick_sample = generator.generate_meditation_progression(
        duration=60,
        initial_state="alert",
        target_state="light_meditation"
    )

    metadata_quick = {
        'subject_id': 'TEST001',
        'session_number': 1,
        'timestamp': datetime.now().isoformat(),
        'duration': 60,
        'description': 'Quick test sample - Alert to Light Meditation',
        'sampling_rate': 256,
        'channels': ['Fz']
    }

    generator.save_eeg_data(quick_sample, metadata_quick, 'sample_eeg_quick.npz')
    print("   Saved to: sample_eeg_quick.npz")
    print()

    # 2. Generate full meditation session
    print("2. Generating 5-minute meditation session...")
    meditation_session = generator.generate_meditation_progression(
        duration=300,
        initial_state="relaxed",
        target_state="deep_meditation"
    )

    metadata_session = {
        'subject_id': 'MED001',
        'session_number': 1,
        'timestamp': datetime.now().isoformat(),
        'duration': 300,
        'description': 'Full meditation session - Relaxed to Deep Meditation',
        'sampling_rate': 256,
        'channels': ['Fz']
    }

    generator.save_eeg_data(meditation_session, metadata_session, 'sample_eeg_meditation.npz')
    print("   Saved to: sample_eeg_meditation.npz")
    print()

    # 3. Generate multi-channel data
    print("3. Generating 2-minute multi-channel recording...")
    multichannel = generator.generate_multiple_channels(
        duration=120,
        num_channels=4,
        state="light_meditation"
    )

    metadata_multi = {
        'subject_id': 'MULTI001',
        'session_number': 1,
        'timestamp': datetime.now().isoformat(),
        'duration': 120,
        'description': 'Multi-channel recording - 4 channels',
        'sampling_rate': 256,
        'channels': ['Fz', 'Cz', 'Pz', 'Oz']
    }

    generator.save_eeg_data(multichannel, metadata_multi, 'sample_eeg_multichannel.npz')
    print("   Saved to: sample_eeg_multichannel.npz")
    print()

    # 4. Generate dataset with multiple subjects
    print("4. Generating full dataset with multiple subjects...")
    print("   This will create multiple session files...")
    df_sessions = generator.generate_session_dataset(
        session_duration=180,  # 3 minutes per session
        num_subjects=3
    )
    print(f"   Generated {len(df_sessions)} sessions")
    print()

    # 5. Visualize a sample
    print("5. Visualizing generated data samples...")
    generator.visualize_generated_data(
        quick_sample[:2560],  # First 10 seconds
        "Quick Test Sample - Alert to Light Meditation (10s)"
    )

    # Print summary
    print()
    print("=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - sample_eeg_quick.npz        : 1-minute test sample")
    print("  - sample_eeg_meditation.npz   : 5-minute meditation session")
    print("  - sample_eeg_multichannel.npz : 2-minute multi-channel data")
    print("  - meditation_sessions_metadata.csv : Session metadata")
    print("  - eeg_session_*.npz : Individual session files")
    print()
    print("To analyze these files, run:")
    print("  python eeg_analysis.py")
    print()


if __name__ == "__main__":
    main()