#!/usr/bin/env python3
"""
EEG Analysis Tool for Meditation Data
Analyzes brainwave patterns with focus on Delta and Theta waves during meditation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import zscore
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class EEGAnalyzer:
    """
    Advanced EEG analysis for meditation practice monitoring.
    Focuses on Delta (0.5-4 Hz) and Theta (4-8 Hz) wave analysis.
    """

    def __init__(self, sampling_rate: int = 256):
        """
        Initialize the EEG analyzer.

        Args:
            sampling_rate: Sampling frequency in Hz (default 256 Hz)
        """
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2

        # Define frequency bands
        self.bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }

        # Meditation-specific thresholds
        self.meditation_thresholds = {
            'delta_theta_ratio': 1.5,  # Higher ratio indicates deeper meditation
            'theta_prominence': 0.3,    # Minimum theta power ratio
            'alpha_suppression': 0.7    # Alpha reduction during deep states
        }

    def preprocess_signal(self, eeg_data: np.ndarray,
                         remove_artifacts: bool = True) -> np.ndarray:
        """
        Preprocess EEG signal with filtering and artifact removal.

        Args:
            eeg_data: Raw EEG signal
            remove_artifacts: Whether to apply artifact removal

        Returns:
            Preprocessed EEG signal
        """
        # Remove DC offset
        eeg_data = eeg_data - np.mean(eeg_data)

        # Apply bandpass filter (0.5-50 Hz or adjusted for sampling rate)
        # Upper cutoff must be < Nyquist frequency
        upper_cutoff = min(50, self.nyquist - 1)
        sos = signal.butter(4, [0.5, upper_cutoff], btype='band',
                           fs=self.sampling_rate, output='sos')
        filtered = signal.sosfilt(sos, eeg_data)

        # Remove artifacts if requested
        if remove_artifacts:
            filtered = self._remove_artifacts(filtered)

        # Apply notch filter for powerline noise (50/60 Hz) only if below Nyquist
        notch_freq = 50  # Change to 60 for US
        if notch_freq < self.nyquist:
            quality_factor = 30
            w0 = notch_freq / self.nyquist
            b, a = signal.iirnotch(w0, quality_factor)
            filtered = signal.filtfilt(b, a, filtered)

        return filtered

    def _remove_artifacts(self, signal_data: np.ndarray,
                         threshold: float = 3.5) -> np.ndarray:
        """
        Remove artifacts using statistical thresholding.

        Args:
            signal_data: EEG signal
            threshold: Z-score threshold for artifact detection

        Returns:
            Cleaned signal
        """
        z_scores = np.abs(zscore(signal_data))
        artifacts = z_scores > threshold

        # Interpolate artifact regions
        if np.any(artifacts):
            clean_indices = ~artifacts
            if np.sum(clean_indices) > 10:  # Ensure enough clean data
                signal_data[artifacts] = np.interp(
                    np.where(artifacts)[0],
                    np.where(clean_indices)[0],
                    signal_data[clean_indices]
                )

        return signal_data

    def compute_power_spectrum(self, eeg_data: np.ndarray,
                              method: str = 'welch') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density of EEG signal.

        Args:
            eeg_data: Preprocessed EEG signal
            method: Method for PSD calculation ('welch' or 'fft')

        Returns:
            Frequencies and power spectral density
        """
        if method == 'welch':
            nperseg = min(len(eeg_data), self.sampling_rate * 4)  # 4-second windows
            freqs, psd = signal.welch(eeg_data, self.sampling_rate,
                                     nperseg=nperseg, noverlap=nperseg//2)
        else:  # FFT method
            fft = np.fft.rfft(eeg_data * np.hanning(len(eeg_data)))
            psd = np.abs(fft) ** 2 / (self.sampling_rate * len(eeg_data))
            freqs = np.fft.rfftfreq(len(eeg_data), 1/self.sampling_rate)

        return freqs, psd

    def calculate_band_power(self, freqs: np.ndarray, psd: np.ndarray,
                            band: Tuple[float, float]) -> float:
        """
        Calculate power in specific frequency band.

        Args:
            freqs: Frequency array
            psd: Power spectral density
            band: Frequency band (low, high) in Hz

        Returns:
            Band power
        """
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        band_power = np.trapz(psd[band_mask], freqs[band_mask])
        return band_power

    def analyze_meditation_state(self, eeg_data: np.ndarray) -> Dict:
        """
        Comprehensive analysis of meditation state from EEG data.

        Args:
            eeg_data: Raw EEG signal

        Returns:
            Dictionary containing meditation metrics
        """
        # Preprocess signal
        clean_eeg = self.preprocess_signal(eeg_data)

        # Compute power spectrum
        freqs, psd = self.compute_power_spectrum(clean_eeg)

        # Calculate band powers
        band_powers = {}
        total_power = 0
        for band_name, band_range in self.bands.items():
            power = self.calculate_band_power(freqs, psd, band_range)
            band_powers[band_name] = power
            total_power += power

        # Calculate relative band powers
        relative_powers = {
            f"{band}_relative": power / total_power
            for band, power in band_powers.items()
        }

        # Meditation-specific metrics
        delta_theta_ratio = band_powers['Delta'] / band_powers['Theta']
        theta_alpha_ratio = band_powers['Theta'] / band_powers['Alpha']

        # Meditation depth score (0-100)
        meditation_score = self._calculate_meditation_score(
            band_powers, relative_powers
        )

        # State classification
        state = self._classify_meditation_state(
            delta_theta_ratio, relative_powers['Theta_relative']
        )

        # Compile results
        results = {
            'band_powers': band_powers,
            'relative_powers': relative_powers,
            'delta_theta_ratio': delta_theta_ratio,
            'theta_alpha_ratio': theta_alpha_ratio,
            'meditation_score': meditation_score,
            'state': state,
            'dominant_frequency': freqs[np.argmax(psd)],
            'spectral_edge_frequency': self._calculate_sef(freqs, psd),
            'spectral_entropy': self._calculate_spectral_entropy(psd)
        }

        return results

    def _calculate_meditation_score(self, band_powers: Dict,
                                   relative_powers: Dict) -> float:
        """
        Calculate meditation depth score (0-100).

        Args:
            band_powers: Absolute band powers
            relative_powers: Relative band powers

        Returns:
            Meditation score
        """
        # Weighted scoring based on meditation markers
        theta_score = min(relative_powers['Theta_relative'] * 200, 40)
        delta_score = min(relative_powers['Delta_relative'] * 150, 30)
        alpha_suppression = max(0, 1 - relative_powers['Alpha_relative'] * 3) * 20
        beta_suppression = max(0, 1 - relative_powers['Beta_relative'] * 5) * 10

        total_score = theta_score + delta_score + alpha_suppression + beta_suppression
        return min(100, max(0, total_score))

    def _classify_meditation_state(self, delta_theta_ratio: float,
                                  theta_relative: float) -> str:
        """
        Classify meditation state based on EEG patterns.

        Args:
            delta_theta_ratio: Ratio of delta to theta power
            theta_relative: Relative theta power

        Returns:
            State classification
        """
        if delta_theta_ratio > 2.0 and theta_relative > 0.35:
            return "Deep Meditation"
        elif theta_relative > 0.3:
            return "Moderate Meditation"
        elif theta_relative > 0.2:
            return "Light Meditation"
        else:
            return "Alert/Active"

    def _calculate_sef(self, freqs: np.ndarray, psd: np.ndarray,
                      percentile: float = 95) -> float:
        """
        Calculate spectral edge frequency.

        Args:
            freqs: Frequency array
            psd: Power spectral density
            percentile: Percentile for SEF calculation

        Returns:
            Spectral edge frequency
        """
        cumsum = np.cumsum(psd)
        total = cumsum[-1]
        threshold = total * percentile / 100
        idx = np.where(cumsum >= threshold)[0][0]
        return freqs[idx]

    def _calculate_spectral_entropy(self, psd: np.ndarray) -> float:
        """
        Calculate spectral entropy as a measure of signal complexity.

        Args:
            psd: Power spectral density

        Returns:
            Spectral entropy
        """
        psd_norm = psd / np.sum(psd)
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-15))
        return entropy

    def plot_analysis(self, eeg_data: np.ndarray,
                     results: Dict, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of EEG analysis.

        Args:
            eeg_data: Raw EEG signal
            results: Analysis results dictionary
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('EEG Meditation Analysis Report', fontsize=16, fontweight='bold')

        # Preprocess for plotting
        clean_eeg = self.preprocess_signal(eeg_data)
        time = np.arange(len(clean_eeg)) / self.sampling_rate
        freqs, psd = self.compute_power_spectrum(clean_eeg)

        # 1. Raw vs Filtered Signal
        ax = axes[0, 0]
        ax.plot(time[:5*self.sampling_rate], eeg_data[:5*self.sampling_rate],
               alpha=0.5, label='Raw', linewidth=0.8)
        ax.plot(time[:5*self.sampling_rate], clean_eeg[:5*self.sampling_rate],
               label='Filtered', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (μV)')
        ax.set_title('EEG Signal (5 second sample)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Power Spectrum
        ax = axes[0, 1]
        ax.semilogy(freqs[freqs <= 50], psd[freqs <= 50])
        # Highlight frequency bands
        for band_name, (low, high) in self.bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            ax.fill_between(freqs[band_mask], psd[band_mask],
                          alpha=0.3, label=band_name)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (μV²/Hz)')
        ax.set_title('Power Spectral Density')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 3. Band Powers Bar Chart
        ax = axes[1, 0]
        bands = list(results['band_powers'].keys())
        powers = list(results['band_powers'].values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        bars = ax.bar(bands, powers, color=colors)
        ax.set_ylabel('Power (μV²)')
        ax.set_title('Absolute Band Powers')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, power in zip(bars, powers):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{power:.2f}', ha='center', va='bottom', fontsize=9)

        # 4. Relative Band Powers Pie Chart
        ax = axes[1, 1]
        relative_powers = {
            band: results['relative_powers'][f"{band}_relative"]
            for band in bands
        }
        wedges, texts, autotexts = ax.pie(
            relative_powers.values(),
            labels=relative_powers.keys(),
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        ax.set_title('Relative Band Powers')

        # 5. Meditation Metrics
        ax = axes[2, 0]
        ax.axis('off')
        metrics_text = f"""
        Meditation State: {results['state']}
        Meditation Score: {results['meditation_score']:.1f}/100

        Delta/Theta Ratio: {results['delta_theta_ratio']:.2f}
        Theta/Alpha Ratio: {results['theta_alpha_ratio']:.2f}

        Dominant Frequency: {results['dominant_frequency']:.2f} Hz
        Spectral Edge (95%): {results['spectral_edge_frequency']:.2f} Hz
        Spectral Entropy: {results['spectral_entropy']:.2f}
        """
        ax.text(0.1, 0.5, metrics_text, fontsize=12,
               verticalalignment='center', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Meditation Metrics', fontsize=12, fontweight='bold')

        # 6. Time-Frequency Spectrogram
        ax = axes[2, 1]
        f, t, Sxx = signal.spectrogram(clean_eeg, self.sampling_rate,
                                       nperseg=self.sampling_rate*2)
        im = ax.pcolormesh(t, f[f <= 30], 10 * np.log10(Sxx[f <= 30]),
                          shading='gouraud', cmap='viridis')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_title('Spectrogram (0-30 Hz)')
        plt.colorbar(im, ax=ax, label='Power (dB)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def analyze_session(self, eeg_data: np.ndarray,
                       window_size: int = 30,
                       overlap: float = 0.5) -> pd.DataFrame:
        """
        Analyze meditation session with sliding window approach.

        Args:
            eeg_data: Full session EEG data
            window_size: Window size in seconds
            overlap: Overlap fraction between windows

        Returns:
            DataFrame with time-series analysis results
        """
        window_samples = window_size * self.sampling_rate
        step_samples = int(window_samples * (1 - overlap))

        results_list = []

        for start in range(0, len(eeg_data) - window_samples, step_samples):
            end = start + window_samples
            window_data = eeg_data[start:end]

            # Analyze window
            window_results = self.analyze_meditation_state(window_data)

            # Add timestamp
            window_results['time'] = start / self.sampling_rate
            window_results['window_start'] = start
            window_results['window_end'] = end

            results_list.append(window_results)

        # Convert to DataFrame
        df = pd.DataFrame(results_list)

        # Flatten nested dictionaries
        band_powers_df = pd.DataFrame(df['band_powers'].tolist())
        band_powers_df.columns = [f'power_{col}' for col in band_powers_df.columns]

        relative_powers_df = pd.DataFrame(df['relative_powers'].tolist())

        # Combine all data
        df = pd.concat([
            df[['time', 'meditation_score', 'state', 'delta_theta_ratio',
                'theta_alpha_ratio', 'dominant_frequency',
                'spectral_edge_frequency', 'spectral_entropy']],
            band_powers_df,
            relative_powers_df
        ], axis=1)

        return df

    def generate_report(self, session_df: pd.DataFrame,
                       save_path: Optional[str] = None) -> str:
        """
        Generate text report of meditation session analysis.

        Args:
            session_df: DataFrame from analyze_session
            save_path: Optional path to save report

        Returns:
            Report text
        """
        report = []
        report.append("=" * 60)
        report.append("EEG MEDITATION SESSION ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")

        # Session Overview
        report.append("SESSION OVERVIEW")
        report.append("-" * 40)
        duration = session_df['time'].max()
        report.append(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        report.append(f"Average Meditation Score: {session_df['meditation_score'].mean():.1f}/100")
        report.append(f"Peak Meditation Score: {session_df['meditation_score'].max():.1f}/100")
        report.append("")

        # State Distribution
        report.append("MEDITATION STATE DISTRIBUTION")
        report.append("-" * 40)
        state_dist = session_df['state'].value_counts(normalize=True) * 100
        for state, percentage in state_dist.items():
            report.append(f"{state}: {percentage:.1f}%")
        report.append("")

        # Brainwave Analysis
        report.append("AVERAGE BRAINWAVE POWERS")
        report.append("-" * 40)
        for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
            rel_power = session_df[f'{band}_relative'].mean() * 100
            report.append(f"{band}: {rel_power:.1f}%")
        report.append("")

        # Key Metrics
        report.append("KEY MEDITATION INDICATORS")
        report.append("-" * 40)
        report.append(f"Avg Delta/Theta Ratio: {session_df['delta_theta_ratio'].mean():.2f}")
        report.append(f"Avg Theta/Alpha Ratio: {session_df['theta_alpha_ratio'].mean():.2f}")
        report.append(f"Avg Dominant Frequency: {session_df['dominant_frequency'].mean():.2f} Hz")
        report.append(f"Avg Spectral Entropy: {session_df['spectral_entropy'].mean():.2f}")
        report.append("")

        # Meditation Quality Assessment
        report.append("MEDITATION QUALITY ASSESSMENT")
        report.append("-" * 40)
        avg_score = session_df['meditation_score'].mean()
        if avg_score >= 70:
            quality = "Excellent - Deep meditative state achieved"
        elif avg_score >= 50:
            quality = "Good - Moderate meditation with room for deepening"
        elif avg_score >= 30:
            quality = "Fair - Light meditation, consider longer practice"
        else:
            quality = "Poor - Difficulty achieving meditative state"
        report.append(f"Overall Quality: {quality}")
        report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)

        theta_avg = session_df['Theta_relative'].mean()
        if theta_avg < 0.25:
            report.append("- Focus on relaxation techniques to increase theta waves")

        beta_avg = session_df['Beta_relative'].mean()
        if beta_avg > 0.3:
            report.append("- Work on reducing mental chatter and active thinking")

        if session_df['meditation_score'].std() > 20:
            report.append("- Practice maintaining consistent meditation depth")

        if avg_score < 50:
            report.append("- Consider guided meditation or breathing exercises")
            report.append("- Ensure comfortable posture and quiet environment")

        report.append("")
        report.append("=" * 60)

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)

        return report_text


def main():
    """
    Example usage of the EEG Analyzer.
    """
    print("EEG Meditation Analysis Tool")
    print("-" * 40)

    # Initialize analyzer
    analyzer = EEGAnalyzer(sampling_rate=256)

    # Generate sample data (replace with real EEG data)
    print("Generating sample EEG data...")
    duration = 300  # 5 minutes
    sampling_rate = 256
    time = np.arange(0, duration, 1/sampling_rate)

    # Simulate meditation EEG with increasing theta/delta over time
    eeg_data = (
        # Delta component (increases over time)
        np.sin(2 * np.pi * 2 * time) * (1 + 0.5 * time/duration) * 20 +
        # Theta component (dominant)
        np.sin(2 * np.pi * 6 * time) * 30 +
        # Alpha component (decreases over time)
        np.sin(2 * np.pi * 10 * time) * (20 - 10 * time/duration) +
        # Beta component (minimal)
        np.sin(2 * np.pi * 20 * time) * 5 +
        # Noise
        np.random.randn(len(time)) * 10
    )

    # Analyze single window
    print("\nAnalyzing meditation state...")
    sample_window = eeg_data[:30*sampling_rate]  # First 30 seconds
    results = analyzer.analyze_meditation_state(sample_window)

    print(f"\nMeditation State: {results['state']}")
    print(f"Meditation Score: {results['meditation_score']:.1f}/100")
    print(f"Delta/Theta Ratio: {results['delta_theta_ratio']:.2f}")

    # Analyze full session
    print("\nAnalyzing full session...")
    session_df = analyzer.analyze_session(eeg_data, window_size=30, overlap=0.5)

    # Generate report
    print("\nGenerating session report...")
    report = analyzer.generate_report(session_df)
    print("\n" + report)

    # Create visualizations
    print("\nCreating visualizations...")
    analyzer.plot_analysis(sample_window, results)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()