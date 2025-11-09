# Quick Start Guide - EEG Meditation Analysis

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install numpy scipy pandas matplotlib seaborn
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Step 2: Generate Sample Data

Run the sample data generator to create test EEG files:

```bash
python generate_sample_eeg.py
```

This creates several sample files:
- `sample_eeg_quick.npz` - 1-minute test sample
- `sample_eeg_meditation.npz` - 5-minute meditation session
- `sample_eeg_multichannel.npz` - Multi-channel recording

### Step 3: Run Your First Analysis

```bash
python eeg_analysis.py
```

This will:
1. Load sample EEG data
2. Analyze meditation state
3. Generate visualizations
4. Create a session report

## 🎯 Common Use Cases

### Analyzing Your Own EEG Data

```python
from eeg_analysis import EEGAnalyzer
import numpy as np

# Load your data (must be a 1D numpy array)
eeg_data = np.loadtxt('your_data.csv')  # Or any other format

# Create analyzer
analyzer = EEGAnalyzer(sampling_rate=256)  # Adjust sampling rate as needed

# Analyze
results = analyzer.analyze_meditation_state(eeg_data)

# View results
print(f"Meditation State: {results['state']}")
print(f"Meditation Score: {results['meditation_score']:.1f}/100")
print(f"Delta/Theta Ratio: {results['delta_theta_ratio']:.2f}")
```

### Analyzing a Full Session

```python
# Load a longer recording (e.g., 10 minutes)
long_recording = np.load('10min_meditation.npy')

# Analyze with sliding windows
session_df = analyzer.analyze_session(
    long_recording,
    window_size=30,  # 30-second windows
    overlap=0.5      # 50% overlap
)

# Generate report
report = analyzer.generate_report(session_df)
print(report)

# Save results
session_df.to_csv('session_analysis.csv')
```

### Creating Custom Visualizations

```python
# Create and save visualization
analyzer.plot_analysis(
    eeg_data,
    results,
    save_path='meditation_analysis.png'
)
```

## 📊 Understanding the Output

### Meditation States
- **Alert/Active**: Beta dominant, low theta
- **Light Meditation**: Increased theta, reduced beta
- **Moderate Meditation**: Strong theta, some delta
- **Deep Meditation**: High delta and theta

### Key Metrics
- **Meditation Score**: 0-100 scale
  - 0-30: Minimal meditation
  - 30-50: Light meditation
  - 50-70: Good meditation
  - 70-100: Deep meditation

- **Delta/Theta Ratio**:
  - < 1.0: Theta dominant (active meditation)
  - 1.0-2.0: Balanced (moderate depth)
  - > 2.0: Delta dominant (deep meditation)

## 🛠️ Customization

### Adjust Sampling Rate
```python
# For different EEG devices
analyzer = EEGAnalyzer(sampling_rate=512)  # For 512 Hz devices
```

### Modify Frequency Bands
```python
# Custom frequency band definitions
analyzer.bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 7),    # Narrower theta band
    'Alpha': (7, 13),   # Adjusted alpha
    'Beta': (13, 30),
    'Gamma': (30, 45)
}
```

### Change Analysis Parameters
```python
# More aggressive artifact removal
clean_data = analyzer.preprocess_signal(
    eeg_data,
    remove_artifacts=True
)

# Different PSD method
freqs, psd = analyzer.compute_power_spectrum(
    clean_data,
    method='fft'  # Use FFT instead of Welch
)
```

## 📝 Data Format Requirements

### Input Data
- **Format**: 1D NumPy array for single channel
- **Sampling Rate**: Typically 256 Hz or 512 Hz
- **Units**: Microvolts (μV)
- **Duration**: Minimum 30 seconds recommended

### Multi-channel Data
```python
# For multi-channel recordings
multi_channel_data = np.load('4_channel_eeg.npy')
# Shape should be (n_channels, n_samples)

# Analyze each channel
for i, channel_data in enumerate(multi_channel_data):
    results = analyzer.analyze_meditation_state(channel_data)
    print(f"Channel {i+1}: {results['state']}")
```

## 🔧 Troubleshooting

### Common Issues

1. **ImportError**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **ValueError: Input data too short**: Ensure at least 30 seconds of data
   ```python
   min_samples = analyzer.sampling_rate * 30
   if len(eeg_data) < min_samples:
       print(f"Need at least {min_samples} samples")
   ```

3. **Noisy Results**: Apply preprocessing
   ```python
   clean_data = analyzer.preprocess_signal(eeg_data, remove_artifacts=True)
   results = analyzer.analyze_meditation_state(clean_data)
   ```

## 📚 Next Steps

1. **Explore Advanced Features**:
   - Read the full [README.md](README.md)
   - Check example notebooks in `/examples`

2. **Integrate with Your Workflow**:
   - Import analyzer into your projects
   - Build real-time analysis pipelines
   - Create custom visualizations

3. **Contribute**:
   - Share your use cases
   - Report issues on GitHub
   - Contribute improvements

## 💡 Tips for Better Results

1. **Data Quality**:
   - Ensure good electrode contact
   - Minimize movement artifacts
   - Use quiet environment

2. **Session Length**:
   - Minimum 5 minutes for reliable analysis
   - 10-20 minutes ideal for tracking progression

3. **Regular Practice**:
   - Track scores over multiple sessions
   - Look for trends rather than absolute values
   - Compare morning vs. evening sessions

## 🆘 Getting Help

- **Documentation**: See [README.md](README.md)
- **Issues**: Open an issue on GitHub
- **Examples**: Check `/examples` folder
- **Community**: Join discussions on GitHub

---

Happy meditating! 🧘‍♀️🧘‍♂️