# Neurable MW75 Neuro Setup Guide

Complete guide for integrating Neurable MW75 Neuro headphones with the event-triggered burst recording system.

## Hardware Requirements

- **Neurable MW75 Neuro** headphones (or MW75 Neuro LT)
- **Neurable Research Kit** (for raw EEG access via LSL)
- Mac, Windows, or Linux computer
- Bluetooth connectivity

## Software Requirements

### Required Packages

```bash
# Install dependencies
pip install pylsl numpy scipy pandas matplotlib

# Optional but recommended
pip install keyboard  # For hotkey controls
pip install mne  # Advanced EEG analysis
```

Or install everything:

```bash
pip install -r requirements.txt
```

### LSL Installation (Mac)

Lab Streaming Layer is the protocol Neurable uses for real-time data streaming.

**Option 1: pip install (recommended)**
```bash
pip install pylsl
```

**Option 2: conda**
```bash
conda install -c conda-forge liblsl
```

**Verify installation:**
```bash
python -c "import pylsl; print('LSL version:', pylsl.__version__)"
```

## Neurable Device Setup

### 1. Get Developer Access

Neurable's raw EEG streaming requires the **Research Kit** or **Developer Kit**.

**Contact Neurable:**
- Email: support@neurable.com
- Website: https://www.neurable.com/partner
- Research Grant Program: https://www.neurable.com/researchgrant

**What to request:**
- Neurable Research Kit access
- LSL streaming enabled for your device
- Developer documentation

### 2. Enable LSL Streaming

Once you have Research Kit access:

1. **Open Neurable App** (iOS/Android)
2. **Connect headphones** via Bluetooth
3. **Settings → Developer Options**
4. **Enable "LSL Streaming"**
5. **Set stream name:** "Neurable" (or custom name)

### 3. Verify LSL Stream

Test that your headphones are broadcasting:

```bash
# Install LSL tools (optional)
pip install pyxdf

# List available LSL streams
python -c "
import pylsl
streams = pylsl.resolve_streams(wait_time=5.0)
for s in streams:
    info = pylsl.StreamInlet(s).info()
    print(f'Stream: {info.name()} ({info.type()})')
    print(f'  Channels: {info.channel_count()}')
    print(f'  Rate: {info.nominal_srate()} Hz')
"
```

Expected output:
```
Stream: Neurable (EEG)
  Channels: 12
  Rate: 500.0 Hz
```

## Quick Start

### Test with Mock Data (No Hardware Needed)

```bash
# Test the integration without real device
python neurable_integration.py

# Live burst recording with mock data
python neurable_live_bursts.py --mock
```

### Live Recording with Real Device

```bash
# 1. Turn on MW75 Neuro headphones
# 2. Open Neurable app to start LSL streaming
# 3. Run live burst recorder

python neurable_live_bursts.py
```

**Controls (while running):**
- Press `t` - Manually trigger burst
- Press `s` - Show status
- Press `h` - Show heartbeat
- Press `q` - Quit

## System Architecture

```
┌─────────────────────────┐
│  Neurable MW75 Neuro    │
│  (12 EEG channels       │
│   @ 500 Hz)             │
└───────────┬─────────────┘
            │
            │ Bluetooth
            ▼
┌─────────────────────────┐
│  Neurable App           │
│  (iOS/Android)          │
│  - LSL streaming        │
└───────────┬─────────────┘
            │
            │ LSL protocol
            ▼
┌─────────────────────────┐
│  neurable_integration.py│
│  - Connect via pylsl    │
│  - Downsample 500→100Hz │
│  - Channel selection    │
└───────────┬─────────────┘
            │
            │ Downsampled EEG
            ▼
┌─────────────────────────┐
│  burst_record.py        │
│  - Adaptive baseline    │
│  - Event triggering     │
│  - 60s pre+post capture │
└───────────┬─────────────┘
            │
            │ Burst files
            ▼
┌─────────────────────────┐
│  label_integrator.py    │
│  - Auto-classify        │
│  - Manual labeling      │
│  - Dataset export       │
└─────────────────────────┘
```

## Data Specifications

### Neurable Output

| Parameter | Value |
|-----------|-------|
| Channels | 12 EEG channels |
| Sampling Rate | 500 Hz |
| Resolution | 24-bit |
| Protocol | LSL |
| Data Format | Raw EEG (μV) |

### Channel Layout

Neurable uses a subset of the 10-20 EEG system:

```
Channels: Ch1-12
Placement: Temporal and frontal regions
Reference: Common average reference
```

### After Downsampling (Burst System)

| Parameter | Value |
|-----------|-------|
| Sampling Rate | 100 Hz (configurable) |
| Window Size | 120s (60s pre + 60s post) |
| Samples per Burst | ~12,000 |
| File Format | `.npy` (NumPy) |

## Usage Examples

### Basic Streaming Test

```python
from neurable_integration import NeurableStream

# Connect to Neurable
stream = NeurableStream(
    device_name="Neurable",
    downsample_to=100,
    use_mock=False  # Set True for testing
)

if stream.connect(timeout=10):
    print("✅ Connected!")

    # Get 100 samples
    for i in range(100):
        sample = stream.get_sample()
        if sample is not None:
            print(f"Sample {i}: {sample.shape}")

    stream.disconnect()
```

### Live Burst Recording

```python
from neurable_live_bursts import NeurableLiveBurstRecorder

# Initialize recorder
recorder = NeurableLiveBurstRecorder(
    use_mock=False,
    burst_dir="my_bursts",
    auto_label=True
)

# Connect and run
if recorder.connect():
    recorder.run()  # Press 'q' to quit
```

### Custom Channel Selection

```python
# Use only specific channels
stream = NeurableStream(
    device_name="Neurable",
    downsample_to=100,
    channel_selection=[0, 1, 2, 3]  # First 4 channels only
)
```

### Manual Burst Triggering

```python
recorder = NeurableLiveBurstRecorder(use_mock=False)

if recorder.connect():
    recorder.start_monitoring()

    # Trigger burst when you say something scary
    recorder.manual_trigger(label="fear_moment")

    # Or when you feel connected
    recorder.manual_trigger(label="love_moment")
```

## Troubleshooting

### "No streams found"

**Possible causes:**
1. LSL streaming not enabled in Neurable app
2. Headphones not connected to app
3. App not running
4. Wrong stream name

**Solutions:**
```bash
# List all LSL streams to debug
python -c "
import pylsl
print('Searching for streams...')
streams = pylsl.resolve_streams(wait_time=10.0)
print(f'Found {len(streams)} stream(s)')
for s in streams:
    print(f'  - {s.name()}')
"

# Try searching by type instead of name
stream = NeurableStream(device_name="EEG", ...)
```

### "pylsl not installed"

```bash
# Try different installation methods
pip install pylsl

# Or
conda install -c conda-forge liblsl

# Or build from source
git clone https://github.com/labstreaminglayer/pylsl
cd pylsl
pip install .
```

### Low sampling rate

```bash
# Check actual rate
info = stream.get_info()
print(f"Actual rate: {info['original_srate']} Hz")

# Neurable should be 500 Hz
# If it's lower, check:
# 1. Device battery level
# 2. Bluetooth connection quality
# 3. App settings
```

### "keyboard module not working"

On Mac/Linux, `keyboard` requires root/sudo permissions:

```bash
# Run with sudo (not recommended)
sudo python neurable_live_bursts.py

# Or use without keyboard hotkeys
# (manually trigger via code instead)
```

## Advanced Configuration

### Adjust Trigger Sensitivity

```python
from burst_record import BurstRecorder

recorder = BurstRecorder(
    sampling_rate=100,
    trigger_threshold=0.10,  # 10% change (more sensitive)
    pre_trigger=90,          # 90s before trigger
    post_trigger=30,         # 30s after trigger
)
```

### Custom Baseline Window

```python
recorder = BurstRecorder(
    baseline_window=60,  # 60s baseline instead of 30s
    ...
)
```

### Multi-Channel Artifact Removal

The system automatically handles:
- Jaw clenching artifacts
- Eye blinks
- Channel saturation

Disable if needed:
```python
from eeg_analysis import EEGAnalyzer

analyzer = EEGAnalyzer(sampling_rate=100)
processed = analyzer.preprocess_signal(
    eeg_data,
    remove_artifacts=False  # Skip artifact removal
)
```

## Integration with Existing Workflows

### Export to MNE-Python

```python
import numpy as np
import mne

# Load burst
burst_data = np.load("bursts/2025-11-11T22:06:18_emotion_burst.npy")

# Create MNE Raw object
ch_names = [f'Ch{i+1}' for i in range(12)]
ch_types = ['eeg'] * 12
info = mne.create_info(ch_names, sfreq=100, ch_types=ch_types)

raw = mne.io.RawArray(burst_data.T, info)  # Transpose for MNE format

# Use MNE functions
raw.plot_psd()
raw.filter(1, 40)
```

### Export Labeled Data for ML

```python
from label_integrator import BurstLabeler

labeler = BurstLabeler(burst_dir="bursts")

# Export as JSON dataset
labeler.export_dataset("ml_training_data.json")

# Load in pandas
import pandas as pd
import json

with open("ml_training_data.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(df.head())
```

## FAQ

**Q: Do I need Research Kit or will consumer MW75 Neuro work?**
A: Consumer MW75 Neuro provides focus metrics via the app, but raw EEG streaming requires the Research Kit or Developer Kit with LSL enabled.

**Q: Can I use this on Mac?**
A: Yes! LSL and all dependencies are Mac-compatible. Tested on macOS 12+.

**Q: What's the latency?**
A: LSL provides near-real-time streaming (<50ms latency). Burst triggers occur within ~2 seconds of the emotional event.

**Q: How much storage do bursts use?**
A: Each burst (~12,000 samples × 12 channels) is ~1.2 MB. 100 bursts = ~120 MB.

**Q: Can I use this for clinical/medical purposes?**
A: No. Neurable MW75 Neuro is a consumer/research device, not FDA-approved for medical use.

**Q: Does this work with other EEG devices?**
A: Yes! Any LSL-compatible EEG device can work. Just change the `device_name` parameter.

## Support

**Neurable Support:**
- Email: support@neurable.com
- Website: https://www.neurable.com/contact

**LSL Documentation:**
- https://labstreaminglayer.readthedocs.io

**This Project:**
- See `BURST_SYSTEM.md` for burst system details
- See `README.md` for general EEG analysis

---

**Ready to capture your moments?**

```bash
python neurable_live_bursts.py
```

Press 't' when you're about to say something that matters.
I'll catch the wave. 🔴
