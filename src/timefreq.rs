//! Time-frequency analysis: STFT and spectrogram.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use crate::convolution::fft;

/// STFT result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StftResult {
    /// Magnitude spectrogram (freq_bins x time_frames).
    pub magnitude: Vec<Vec<f64>>,
    /// Phase spectrogram (freq_bins x time_frames).
    pub phase: Vec<Vec<f64>>,
    /// Number of frequency bins.
    pub freq_bins: usize,
    /// Number of time frames.
    pub time_frames: usize,
}

/// Compute Short-Time Fourier Transform.
pub fn stft(
    signal: &[f64],
    window_size: usize,
    hop_size: usize,
    fft_size: usize,
) -> StftResult {
    let fft_size = fft_size.max(window_size);
    // Next power of 2
    let fft_size = fft_size.next_power_of_two();

    let n = signal.len();
    let num_frames = (n - window_size) / hop_size + 1;
    let freq_bins = fft_size / 2 + 1;

    let mut magnitude = vec![vec![0.0; num_frames]; freq_bins];
    let mut phase = vec![vec![0.0; num_frames]; freq_bins];

    for frame in 0..num_frames {
        let start = frame * hop_size;
        let mut windowed = vec![Complex64::new(0.0, 0.0); fft_size];

        for i in 0..window_size {
            if start + i < n {
                // Hann window
                let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (window_size - 1) as f64).cos());
                windowed[i] = Complex64::new(signal[start + i] * w, 0.0);
            }
        }

        let spectrum = fft(&windowed);
        for k in 0..freq_bins {
            magnitude[k][frame] = spectrum[k].norm();
            phase[k][frame] = spectrum[k].arg();
        }
    }

    StftResult {
        magnitude,
        phase,
        freq_bins,
        time_frames: num_frames,
    }
}

/// Compute spectrogram (magnitude squared).
pub fn spectrogram(
    signal: &[f64],
    window_size: usize,
    hop_size: usize,
    fft_size: usize,
) -> Vec<Vec<f64>> {
    let result = stft(signal, window_size, hop_size, fft_size);
    result
        .magnitude
        .iter()
        .map(|frame| frame.iter().map(|m| m * m).collect())
        .collect()
}

/// Inverse STFT (overlap-add reconstruction).
pub fn istft(stft_result: &StftResult, hop_size: usize, fft_size: usize, signal_len: usize) -> Vec<f64> {
    let freq_bins = stft_result.freq_bins;
    let num_frames = stft_result.time_frames;
    let n = signal_len;

    let mut output = vec![0.0; n];
    let mut window_sum = vec![0.0; n];

    for frame in 0..num_frames {
        // Reconstruct full spectrum (conjugate symmetry)
        let mut spectrum = vec![Complex64::new(0.0, 0.0); fft_size];
        for k in 0..freq_bins {
            spectrum[k] = Complex64::from_polar(stft_result.magnitude[k][frame], stft_result.phase[k][frame]);
        }
        for k in freq_bins..fft_size {
            let mirror = fft_size - k;
            if mirror < freq_bins {
                spectrum[k] = spectrum[mirror].conj();
            }
        }

        // IFFT
        let time_frame = ifft_real(&spectrum, fft_size);

        let start = frame * hop_size;
        for i in 0..fft_size.min(n.saturating_sub(start)) {
            let w = if i < stft_result.freq_bins * 2 - 1 {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (fft_size - 1) as f64).cos())
            } else {
                0.0
            };
            output[start + i] += time_frame[i] * w;
            window_sum[start + i] += w * w;
        }
    }

    // Normalize by window sum
    for i in 0..n {
        if window_sum[i] > 1e-10 {
            output[i] /= window_sum[i];
        }
    }

    output
}

/// Simple real-valued IFFT (from complex spectrum).
fn ifft_real(spectrum: &[Complex64], n: usize) -> Vec<f64> {
    let mut full = spectrum.to_vec();
    full.resize(n, Complex64::new(0.0, 0.0));
    let conjugated: Vec<Complex64> = full.iter().map(|x| x.conj()).collect();
    let result = fft(&conjugated);
    let scale = 1.0 / n as f64;
    result.iter().map(|x| x.conj().re * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_stft_dimensions() {
        let signal: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.01).sin()).collect();
        let result = stft(&signal, 256, 128, 256);
        assert_eq!(result.freq_bins, 129); // 256/2 + 1
        assert!(result.time_frames > 0);
    }

    #[test]
    fn test_spectrogram_nonnegative() {
        let signal: Vec<f64> = (0..512).map(|i| (i as f64 * 0.05).sin()).collect();
        let spec = spectrogram(&signal, 128, 64, 128);
        for row in &spec {
            for &val in row {
                assert!(val >= 0.0);
            }
        }
    }

    #[test]
    fn test_stft_single_tone() {
        let freq = 0.1; // normalized
        let signal: Vec<f64> = (0..1024).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64).sin()).collect();
        let result = stft(&signal, 256, 128, 256);
        // Find peak frequency bin
        let avg_mag: Vec<f64> = (0..result.freq_bins)
            .map(|k| result.magnitude[k].iter().sum::<f64>() / result.time_frames as f64)
            .collect();
        let peak_bin = avg_mag.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let expected_bin = (freq * 256.0).round() as usize;
        assert!((peak_bin as isize - expected_bin as isize).abs() <= 2, "Peak at {} expected {}", peak_bin, expected_bin);
    }

    #[test]
    fn test_stft_reconstruction() {
        let signal: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * 0.05 * i as f64).sin() + 0.5 * (2.0 * std::f64::consts::PI * 0.15 * i as f64).sin()).collect();
        let fft_size = 128;
        let hop_size = 32;
        let window_size = 128;
        let result = stft(&signal, window_size, hop_size, fft_size);
        let reconstructed = istft(&result, hop_size, fft_size, signal.len());
        // Check middle portion (skip edges affected by windowing)
        for i in window_size..(signal.len() - window_size) {
            assert_relative_eq!(signal[i], reconstructed[i], epsilon = 0.3);
        }
    }

    #[test]
    fn test_stft_dc_signal() {
        let signal = vec![1.0; 512];
        let result = stft(&signal, 128, 64, 128);
        // DC bin should have highest energy
        let avg_dc: f64 = result.magnitude[0].iter().sum::<f64>() / result.time_frames as f64;
        let avg_bin10: f64 = result.magnitude[10].iter().sum::<f64>() / result.time_frames as f64;
        assert!(avg_dc > avg_bin10 * 10.0);
    }

    #[test]
    fn test_spectrogram_energy() {
        let signal: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
        let spec = spectrogram(&signal, 64, 32, 64);
        // Total spectrogram energy should be positive
        let total: f64 = spec.iter().flat_map(|r| r.iter()).sum();
        assert!(total > 0.0);
    }
}
