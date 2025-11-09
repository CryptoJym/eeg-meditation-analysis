#!/usr/bin/env python3
"""
Example Usage of EEG Meditation Analysis Toolkit

This script demonstrates how to use the EEG analysis tools
for analyzing meditation sessions.
"""

import numpy as np
from eeg_analysis import EEGAnalyzer
from generate_sample_eeg import EEGDataGenerator
import matplotlib.pyplot as plt

def demo_basic_analysis():
    """Demonstrate basic EEG analysis workflow."""
    print("=" * 60)
    print("EEG MEDITATION ANALYSIS - EXAMPLE USAGE")
    print("=" * 60)
    print()

    # Initialize the analyzer
    print("1. Initializing EEG Analyzer...")
    analyzer = EEGAnalyzer(sampling_rate=256)

    # Generate sample data (replace with your real data)
    print("2. Generating sample meditation data...")
    generator = EEGDataGenerator(sampling_rate=256)

    # Create a 2-minute progression from relaxed to deep meditation
    duration = 120  # seconds
    eeg_data = generator.generate_meditation_progression(
        duration=duration,
        initial_state="relaxed",
        target_state="deep_meditation"
    )

    print(f"   Generated {duration} seconds of EEG data")
    print(f"   Sample rate: 256 Hz")
    print(f"   Total samples: {len(eeg_data)}")
    print()

    # Analyze the first 30 seconds
    print("3. Analyzing EEG signal (first 30 seconds)...")
    window_data = eeg_data[:30*256]  # 30 seconds
    results = analyzer.analyze_meditation_state(window_data)

    # Display results
    print("\n" + "=" * 40)
    print("ANALYSIS RESULTS")
    print("=" * 40)
    print(f"Meditation State: {results['state']}")
    print(f"Meditation Score: {results['meditation_score']:.1f}/100")
    print()

    print("Band Powers (μV²):")
    for band, power in results['band_powers'].items():
        rel_power = results['relative_powers'][f'{band}_relative'] * 100
        print(f"  {band:6s}: {power:8.2f} ({rel_power:5.1f}%)")
    print()

    print("Key Indicators:")
    print(f"  Delta/Theta Ratio: {results['delta_theta_ratio']:.2f}")
    print(f"  Theta/Alpha Ratio: {results['theta_alpha_ratio']:.2f}")
    print(f"  Dominant Frequency: {results['dominant_frequency']:.2f} Hz")
    print(f"  Spectral Entropy: {results['spectral_entropy']:.2f}")
    print()

    # Create visualization
    print("4. Creating visualization...")
    analyzer.plot_analysis(window_data, results, save_path='example_analysis.png')
    print("   Visualization saved to: example_analysis.png")
    print()

    return analyzer, eeg_data

def demo_session_analysis(analyzer, eeg_data):
    """Demonstrate session-wide analysis with sliding windows."""
    print("5. Analyzing full session with sliding windows...")
    session_df = analyzer.analyze_session(
        eeg_data,
        window_size=30,  # 30-second windows
        overlap=0.5      # 50% overlap
    )

    print(f"   Analyzed {len(session_df)} windows")
    print(f"   Time span: 0 to {session_df['time'].max():.1f} seconds")
    print()

    # Generate session report
    print("6. Generating session report...")
    report = analyzer.generate_report(session_df, save_path='session_report.txt')
    print("\nSESSION REPORT PREVIEW:")
    print("-" * 40)
    # Print first few lines of report
    report_lines = report.split('\n')
    for line in report_lines[:20]:
        print(line)
    print("...")
    print("   Full report saved to: session_report.txt")
    print()

    # Plot session progression
    print("7. Creating session progression plot...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Meditation score over time
    axes[0].plot(session_df['time'], session_df['meditation_score'],
                 linewidth=2, color='purple')
    axes[0].set_ylabel('Meditation Score')
    axes[0].set_title('Meditation Session Progression')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 100)

    # Band power trends
    axes[1].plot(session_df['time'], session_df['Delta_relative'],
                 label='Delta', linewidth=2)
    axes[1].plot(session_df['time'], session_df['Theta_relative'],
                 label='Theta', linewidth=2)
    axes[1].plot(session_df['time'], session_df['Alpha_relative'],
                 label='Alpha', linewidth=2)
    axes[1].set_ylabel('Relative Power')
    axes[1].set_title('Brainwave Band Evolution')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    # Delta/Theta ratio
    axes[2].plot(session_df['time'], session_df['delta_theta_ratio'],
                 linewidth=2, color='green')
    axes[2].set_ylabel('Delta/Theta Ratio')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_title('Meditation Depth Indicator')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=1.5, color='red', linestyle='--', alpha=0.5,
                    label='Deep Meditation Threshold')
    axes[2].legend(loc='best')

    plt.tight_layout()
    plt.savefig('session_progression.png', dpi=150)
    plt.show()

    print("   Session progression plot saved to: session_progression.png")
    print()

    return session_df

def demo_multi_state_comparison():
    """Compare different meditation states."""
    print("8. Comparing different meditation states...")
    print("-" * 40)

    analyzer = EEGAnalyzer(sampling_rate=256)
    generator = EEGDataGenerator(sampling_rate=256)

    states = ['alert', 'relaxed', 'light_meditation', 'deep_meditation']
    state_results = {}

    for state in states:
        # Generate 30 seconds of each state
        eeg_data = generator.generate_meditation_progression(
            duration=30,
            initial_state=state,
            target_state=state  # Stay in same state
        )

        # Analyze
        results = analyzer.analyze_meditation_state(eeg_data)
        state_results[state] = results

        print(f"\n{state.upper().replace('_', ' ')}:")
        print(f"  Classification: {results['state']}")
        print(f"  Score: {results['meditation_score']:.1f}/100")
        print(f"  Theta Power: {results['relative_powers']['Theta_relative']*100:.1f}%")
        print(f"  Delta Power: {results['relative_powers']['Delta_relative']*100:.1f}%")

    print()
    return state_results

def main():
    """Run all demonstrations."""
    try:
        # Basic analysis demo
        analyzer, eeg_data = demo_basic_analysis()

        # Session analysis demo
        session_df = demo_session_analysis(analyzer, eeg_data)

        # State comparison demo
        state_results = demo_multi_state_comparison()

        print("=" * 60)
        print("EXAMPLE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - example_analysis.png    : Single window analysis")
        print("  - session_report.txt      : Full session report")
        print("  - session_progression.png : Session progression plots")
        print()
        print("Next steps:")
        print("  1. Replace sample data with your real EEG recordings")
        print("  2. Adjust parameters based on your EEG device")
        print("  3. Customize analysis for your specific needs")
        print()

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        raise

if __name__ == "__main__":
    main()