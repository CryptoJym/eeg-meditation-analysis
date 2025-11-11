#!/usr/bin/env python3
"""
Test script for event-triggered burst system.
"""

import numpy as np
from burst_record import BurstRecorder
from label_integrator import BurstLabeler
from pathlib import Path


def test_burst_recording():
    """Test burst recording with complete capture."""
    print("=" * 60)
    print("TESTING EVENT-TRIGGERED BURST SYSTEM")
    print("=" * 60)

    # Initialize recorder
    recorder = BurstRecorder(sampling_rate=100, output_dir="bursts")

    # Generate test data: 90 seconds
    # - First 30s: baseline (relaxed)
    # - 30-35s: emotional spike (arousal)
    # - 35-90s: post-trigger continuation
    duration = 90
    sampling_rate = 100
    t = np.linspace(0, duration, duration * sampling_rate)

    print("\n1. Generating test EEG data...")
    print(f"   Duration: {duration}s")
    print(f"   Sampling rate: {sampling_rate} Hz")

    # Baseline: calm state with alpha dominance
    signal = (
        np.sin(2 * np.pi * 2 * t) * 10 +   # Delta
        np.sin(2 * np.pi * 6 * t) * 12 +   # Theta
        np.sin(2 * np.pi * 10 * t) * 25 +  # Alpha (high)
        np.sin(2 * np.pi * 20 * t) * 8 +   # Beta (low)
        np.random.randn(len(t)) * 5        # Noise
    )

    # Add emotional event at t=30s: gamma + beta surge (arousal)
    spike_center = 32
    spike_width = 3
    spike = 70 * np.exp(-((t - spike_center) ** 2) / (2 * spike_width ** 2))

    # Arousal pattern: high gamma + beta, low theta
    gamma_component = spike * np.sin(2 * np.pi * 40 * t)
    beta_component = spike * 0.7 * np.sin(2 * np.pi * 25 * t)

    signal = signal + gamma_component + beta_component

    print("   ✓ Test signal generated")
    print("     - Baseline: 0-30s (calm/rest)")
    print("     - Event: ~32s (arousal spike)")
    print("     - Post-event: 35-90s (continuation)")

    # Process stream
    print("\n2. Processing EEG stream...")
    print("   Building baseline...")

    burst_saved = False
    burst_file = None

    for i, sample in enumerate(signal):
        is_quiet = i < 30 * sampling_rate or i > 35 * sampling_rate  # Active during spike

        result = recorder.process_sample(sample, is_quiet)

        if result:
            burst_file = result
            burst_saved = True
            break

        # Progress indicator
        if i % (sampling_rate * 10) == 0 and i > 0:
            print(f"   t={i/sampling_rate:.0f}s", end="\r")

    if burst_saved:
        print(f"\n   ✓ Burst captured and saved!")
    else:
        print(f"\n   ✗ No burst triggered")
        return None

    return burst_file


def test_labeling(burst_file):
    """Test automatic and manual labeling."""
    if not burst_file:
        print("\n⚠️  No burst file to label!")
        return

    print("\n" + "=" * 60)
    print("3. Testing automatic labeling...")

    labeler = BurstLabeler(burst_dir="bursts")

    # Auto-label the burst
    burst_path = Path(burst_file)
    label = labeler.auto_label_burst(burst_path)

    print(f"   ✓ Automatic label applied")
    print(f"     Emotion: {label['emotion']}")
    print(f"     Confidence: {label['confidence']:.2f}")
    print(f"     Dominant frequency: {label['dominant_frequency']:.1f} Hz")

    print("\n   Band Powers:")
    for band, power in label['relative_powers'].items():
        print(f"     {band:6s}: {power*100:5.1f}%")

    # Test manual override
    print("\n4. Testing manual label override...")
    labeler.manual_label(
        burst_path,
        emotion="intense_focus",
        notes="Test manual override - high concentration moment"
    )
    print("   ✓ Manual label applied")

    # List labeled bursts
    print("\n5. Listing all labeled bursts...")
    labeled = labeler.list_labeled_bursts()
    print(f"   ✓ Found {len(labeled)} labeled burst(s)")

    for item in labeled:
        emotion = item['effective_emotion']
        label_type = "📝 Manual" if item.get('manual_override') else "🤖 Auto"
        print(f"     {label_type}: {emotion}")

    # Export dataset
    print("\n6. Exporting labeled dataset...")
    labeler.export_dataset("test_labeled_dataset.json")


def main():
    """Run all tests."""
    # Test recording
    burst_file = test_burst_recording()

    # Test labeling
    if burst_file:
        test_labeling(burst_file)

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - bursts/*_emotion_burst.npy     : Burst data")
        print("  - bursts/*_emotion_burst_meta.json : Burst metadata")
        print("  - bursts/labels.json              : All labels")
        print("  - test_labeled_dataset.json       : Exported dataset")
        print("\n🎉 Event-triggered burst system is working!")
    else:
        print("\n❌ TEST FAILED: Burst recording did not complete")


if __name__ == "__main__":
    main()
