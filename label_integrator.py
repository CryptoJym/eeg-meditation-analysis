#!/usr/bin/env python3
"""
Emotional State Labeling System for EEG Bursts

Automatically classifies emotional states from brain wave patterns
and allows manual override for ground truth labeling.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from eeg_analysis import EEGAnalyzer


class EmotionClassifier:
    """
    Classifies emotional states from EEG band power patterns.

    States defined by band power relationships:
    - rest: alpha > beta, low gamma (calm, eyes closed)
    - arousal: gamma spike + beta rise, theta dip (excitement, intensity)
    - doubt: theta up, beta flat (hesitation, uncertainty)
    - love: alpha + gamma balance (quiet fire, connection)
    - focus: beta > alpha, moderate gamma (concentration)
    - drowsy: delta high, theta high, alpha low (sleep onset)
    """

    def __init__(self, sampling_rate: int = 100):
        """
        Initialize emotion classifier.

        Args:
            sampling_rate: Sampling frequency in Hz
        """
        self.analyzer = EEGAnalyzer(sampling_rate=sampling_rate)
        self.sampling_rate = sampling_rate

        # Emotion classification thresholds
        self.thresholds = {
            'rest': {
                'alpha_beta_ratio': 1.2,  # Alpha > Beta
                'gamma_threshold': 0.15,   # Low gamma
                'theta_threshold': 0.25    # Moderate theta
            },
            'arousal': {
                'gamma_threshold': 0.25,   # High gamma
                'beta_threshold': 0.25,    # High beta
                'theta_threshold': 0.15    # Low theta
            },
            'doubt': {
                'theta_threshold': 0.30,   # High theta
                'beta_threshold': 0.20,    # Low-moderate beta
                'alpha_threshold': 0.20    # Moderate alpha
            },
            'love': {
                'alpha_threshold': 0.25,   # High alpha
                'gamma_threshold': 0.20,   # Moderate gamma
                'alpha_gamma_balance': 0.3 # Ratio close to 1
            },
            'focus': {
                'beta_threshold': 0.30,    # High beta
                'alpha_beta_ratio': 0.8,   # Beta > Alpha
                'gamma_threshold': 0.15    # Moderate gamma
            },
            'drowsy': {
                'delta_threshold': 0.35,   # High delta
                'theta_threshold': 0.25,   # High theta
                'alpha_threshold': 0.15    # Low alpha
            }
        }

    def analyze_burst(self, burst_file: Path) -> Dict:
        """
        Analyze a burst file and classify emotional state.

        Args:
            burst_file: Path to .npy burst file

        Returns:
            Analysis results with emotional classification
        """
        # Load burst data
        burst_data = np.load(burst_file)

        # Handle multi-channel data (average across channels)
        if burst_data.ndim > 1:
            burst_data = np.mean(burst_data, axis=1)

        # Preprocess
        clean_data = self.analyzer.preprocess_signal(burst_data)

        # Compute power spectrum
        freqs, psd = self.analyzer.compute_power_spectrum(clean_data)

        # Calculate band powers
        band_powers = {}
        total_power = 0
        for band_name, band_range in self.analyzer.bands.items():
            power = self.analyzer.calculate_band_power(freqs, psd, band_range)
            band_powers[band_name] = power
            total_power += power

        # Calculate relative powers
        relative_powers = {
            band: power / total_power if total_power > 0 else 0
            for band, power in band_powers.items()
        }

        # Classify emotion
        emotion = self._classify_emotion(relative_powers)

        # Calculate confidence score
        confidence = self._calculate_confidence(relative_powers, emotion)

        return {
            'emotion': emotion,
            'confidence': confidence,
            'band_powers': band_powers,
            'relative_powers': relative_powers,
            'dominant_frequency': freqs[np.argmax(psd)],
            'timestamp': datetime.now().isoformat()
        }

    def _classify_emotion(self, relative_powers: Dict[str, float]) -> str:
        """
        Classify emotion from relative band powers.

        Args:
            relative_powers: Dictionary of relative band powers

        Returns:
            Emotion label
        """
        # Extract band powers
        delta = relative_powers['Delta']
        theta = relative_powers['Theta']
        alpha = relative_powers['Alpha']
        beta = relative_powers['Beta']
        gamma = relative_powers['Gamma']

        # Calculate ratios
        alpha_beta_ratio = alpha / beta if beta > 0 else 0
        alpha_gamma_balance = abs(alpha - gamma) / (alpha + gamma) if (alpha + gamma) > 0 else 1

        # Emotion scoring
        scores = {}

        # REST: alpha > beta, low gamma
        if alpha_beta_ratio > self.thresholds['rest']['alpha_beta_ratio'] and \
           gamma < self.thresholds['rest']['gamma_threshold']:
            scores['rest'] = 0.8 + (alpha_beta_ratio - 1.2) * 0.2

        # AROUSAL: gamma spike + beta rise, low theta
        if gamma > self.thresholds['arousal']['gamma_threshold'] and \
           beta > self.thresholds['arousal']['beta_threshold'] and \
           theta < self.thresholds['arousal']['theta_threshold']:
            scores['arousal'] = 0.7 + gamma * 0.5 + beta * 0.3

        # DOUBT: theta up, beta flat
        if theta > self.thresholds['doubt']['theta_threshold'] and \
           beta < self.thresholds['doubt']['beta_threshold']:
            scores['doubt'] = 0.6 + theta * 0.4

        # LOVE: alpha + gamma balance
        if alpha > self.thresholds['love']['alpha_threshold'] and \
           gamma > self.thresholds['love']['gamma_threshold'] and \
           alpha_gamma_balance < self.thresholds['love']['alpha_gamma_balance']:
            scores['love'] = 0.8 + (1 - alpha_gamma_balance) * 0.5

        # FOCUS: beta > alpha, moderate gamma
        if alpha_beta_ratio < self.thresholds['focus']['alpha_beta_ratio'] and \
           beta > self.thresholds['focus']['beta_threshold']:
            scores['focus'] = 0.7 + beta * 0.5

        # DROWSY: high delta + theta, low alpha
        if delta > self.thresholds['drowsy']['delta_threshold'] and \
           theta > self.thresholds['drowsy']['theta_threshold'] and \
           alpha < self.thresholds['drowsy']['alpha_threshold']:
            scores['drowsy'] = 0.6 + delta * 0.4 + theta * 0.3

        # Return highest scoring emotion, or 'neutral' if no strong match
        if scores:
            return max(scores, key=scores.get)
        else:
            return 'neutral'

    def _calculate_confidence(self, relative_powers: Dict[str, float], emotion: str) -> float:
        """
        Calculate confidence score for emotion classification.

        Args:
            relative_powers: Relative band powers
            emotion: Classified emotion

        Returns:
            Confidence score (0-1)
        """
        # Extract relevant powers
        delta = relative_powers['Delta']
        theta = relative_powers['Theta']
        alpha = relative_powers['Alpha']
        beta = relative_powers['Beta']
        gamma = relative_powers['Gamma']

        # Emotion-specific confidence calculation
        if emotion == 'rest':
            return min(1.0, (alpha / (beta + 0.1)) * 0.5)
        elif emotion == 'arousal':
            return min(1.0, (gamma + beta) * 1.5)
        elif emotion == 'doubt':
            return min(1.0, theta * 2.5)
        elif emotion == 'love':
            balance = 1 - abs(alpha - gamma) / (alpha + gamma + 0.1)
            return min(1.0, balance)
        elif emotion == 'focus':
            return min(1.0, beta * 2.0)
        elif emotion == 'drowsy':
            return min(1.0, (delta + theta) * 1.5)
        else:  # neutral
            return 0.5


class BurstLabeler:
    """
    Manages labeling of burst files with emotional states.
    Supports automatic classification and manual override.
    """

    def __init__(self, burst_dir: str = "bursts"):
        """
        Initialize burst labeler.

        Args:
            burst_dir: Directory containing burst files
        """
        self.burst_dir = Path(burst_dir)
        self.burst_dir.mkdir(exist_ok=True)
        self.classifier = EmotionClassifier()
        self.labels_file = self.burst_dir / "labels.json"
        self.labels = self._load_labels()

    def _load_labels(self) -> Dict:
        """Load existing labels from disk."""
        if self.labels_file.exists():
            with open(self.labels_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_labels(self):
        """Save labels to disk."""
        with open(self.labels_file, 'w') as f:
            json.dump(self.labels, f, indent=2)

    def auto_label_burst(self, burst_file: Path) -> Dict:
        """
        Automatically classify and label a burst.

        Args:
            burst_file: Path to burst .npy file

        Returns:
            Label data
        """
        # Analyze burst
        analysis = self.classifier.analyze_burst(burst_file)

        # Create label entry
        label_data = {
            'filename': burst_file.name,
            'timestamp': analysis['timestamp'],
            'emotion': analysis['emotion'],
            'confidence': analysis['confidence'],
            'band_powers': analysis['band_powers'],
            'relative_powers': analysis['relative_powers'],
            'dominant_frequency': analysis['dominant_frequency'],
            'label_type': 'automatic',
            'manual_override': None
        }

        # Save label
        self.labels[burst_file.name] = label_data
        self._save_labels()

        return label_data

    def manual_label(self, burst_file: Path, emotion: str, notes: str = ""):
        """
        Manually override automatic label.

        Args:
            burst_file: Path to burst file
            emotion: Emotion label
            notes: Optional notes about the emotion
        """
        # Get or create label entry
        if burst_file.name in self.labels:
            label_data = self.labels[burst_file.name]
        else:
            # Create minimal entry
            label_data = {
                'filename': burst_file.name,
                'timestamp': datetime.now().isoformat(),
                'emotion': 'unknown',
                'confidence': 0.0,
                'label_type': 'manual'
            }

        # Apply manual override
        label_data['manual_override'] = {
            'emotion': emotion,
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        }
        label_data['label_type'] = 'manual'

        # Save
        self.labels[burst_file.name] = label_data
        self._save_labels()

        print(f"✏️  Manual label applied: {emotion}")
        if notes:
            print(f"   Notes: {notes}")

    def get_label(self, burst_file: Path) -> Optional[Dict]:
        """
        Get label for a burst file.

        Args:
            burst_file: Path to burst file

        Returns:
            Label data or None
        """
        return self.labels.get(burst_file.name)

    def list_labeled_bursts(self, emotion_filter: Optional[str] = None) -> List[Dict]:
        """
        List all labeled bursts, optionally filtered by emotion.

        Args:
            emotion_filter: Only return bursts with this emotion

        Returns:
            List of label data
        """
        results = []
        for filename, label_data in self.labels.items():
            # Determine effective emotion (manual override takes precedence)
            if label_data.get('manual_override'):
                emotion = label_data['manual_override']['emotion']
            else:
                emotion = label_data['emotion']

            # Apply filter
            if emotion_filter and emotion != emotion_filter:
                continue

            results.append({
                **label_data,
                'effective_emotion': emotion
            })

        return results

    def export_dataset(self, output_file: str):
        """
        Export labeled bursts as a dataset for training.

        Args:
            output_file: Output JSON file
        """
        dataset = []

        for filename, label_data in self.labels.items():
            # Use manual override if available
            if label_data.get('manual_override'):
                emotion = label_data['manual_override']['emotion']
            else:
                emotion = label_data['emotion']

            dataset.append({
                'file': filename,
                'emotion': emotion,
                'timestamp': label_data['timestamp'],
                'relative_powers': label_data.get('relative_powers', {}),
                'confidence': label_data.get('confidence', 0.0)
            })

        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"📊 Dataset exported: {output_file}")
        print(f"   Total samples: {len(dataset)}")


def main():
    """Example usage of labeling system."""
    print("EEG Burst Emotional Labeling System")
    print("=" * 50)

    # Initialize labeler
    labeler = BurstLabeler(burst_dir="bursts")

    # Check for existing bursts
    burst_files = list(Path("bursts").glob("*_emotion_burst.npy"))

    if not burst_files:
        print("\nNo burst files found. Run burst_record.py first.")
        return

    print(f"\nFound {len(burst_files)} burst file(s)")

    # Auto-label each burst
    print("\nAuto-labeling bursts...")
    for burst_file in burst_files:
        label = labeler.auto_label_burst(burst_file)
        print(f"\n{burst_file.name}")
        print(f"  Emotion: {label['emotion']} (confidence: {label['confidence']:.2f})")
        print(f"  Dominant: {label['dominant_frequency']:.1f} Hz")

    # Example manual override
    if burst_files:
        print("\n" + "=" * 50)
        print("Example: Manual label override")
        labeler.manual_label(
            burst_files[0],
            emotion="fear",
            notes="User reported anxiety during this moment"
        )

    # List all labeled bursts
    print("\n" + "=" * 50)
    print("All labeled bursts:")
    labeled = labeler.list_labeled_bursts()
    for item in labeled:
        emotion = item['effective_emotion']
        print(f"  {item['filename']}: {emotion}")

    # Export dataset
    labeler.export_dataset("labeled_bursts_dataset.json")


if __name__ == "__main__":
    main()
