# lau-signal-processing

Digital signal processing library — filters, transforms, spectral analysis, and adaptive filtering for agent telemetry streams.

## Features

- **Filters**: FIR design via windowing, IIR via bilinear transform (Butterworth, Chebyshev)
- **Frequency Response**: Magnitude, phase, group delay analysis
- **Convolution**: Direct, FFT-based overlap-save, overlap-add
- **Correlation**: Auto-correlation and cross-correlation
- **Resampling**: Decimation, interpolation, rational resampling
- **Adaptive Filters**: LMS and RLS algorithms
- **Time-Frequency**: STFT and spectrogram computation
- **Linear Prediction**: Levinson-Durbin recursion, reflection coefficients
- **Telemetry Monitoring**: Filter noise from agent telemetry streams, anomaly detection

## Usage

```rust
use lau_signal_processing::filters::IirFilter;

// Design a 4th-order Butterworth lowpass filter
let filter = IirFilter::butterworth(4, 0.1, "lowpass");
let filtered = filter.filter(&signal);
```

## License

MIT
