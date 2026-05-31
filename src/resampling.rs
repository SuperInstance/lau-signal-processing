//! Resampling: decimation, interpolation, polyphase.

use crate::filters::IirFilter;

/// Decimate signal by factor M (lowpass filter then downsample).
pub fn decimate(signal: &[f64], factor: usize) -> Vec<f64> {
    if factor <= 1 {
        return signal.to_vec();
    }
    let cutoff = 1.0 / (2.0 * factor as f64);
    let iir = IirFilter::butterworth(4, cutoff, "lowpass");
    let filtered = iir.filter(signal);
    filtered.iter().step_by(factor).copied().collect()
}

/// Interpolate signal by factor L (upsample then lowpass filter).
pub fn interpolate(signal: &[f64], factor: usize) -> Vec<f64> {
    if factor <= 1 {
        return signal.to_vec();
    }
    let cutoff = 1.0 / (2.0 * factor as f64);
    let iir = IirFilter::butterworth(4, cutoff, "lowpass");

    // Upsample: insert L-1 zeros between samples
    let mut upsampled = vec![0.0; signal.len() * factor];
    for (i, &s) in signal.iter().enumerate() {
        upsampled[i * factor] = s;
    }

    let filtered = iir.filter(&upsampled);
    // Scale by factor to compensate for zero-insertion
    filtered.into_iter().map(|x| x * factor as f64).collect()
}

/// Resample by rational factor L/M using polyphase decomposition.
pub fn resample_polyphase(signal: &[f64], l: usize, m: usize) -> Vec<f64> {
    if l == m {
        return signal.to_vec();
    }
    // Use cascaded interpolate + decimate approach
    let upsampled = interpolate(signal, l);
    decimate(&upsampled, m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_decimate_length() {
        let signal = vec![1.0; 100];
        let decimated = decimate(&signal, 5);
        assert_eq!(decimated.len(), 20);
    }

    #[test]
    fn test_decimate_constant() {
        let signal = vec![5.0; 200];
        let decimated = decimate(&signal, 2);
        // After Butterworth lowpass, constant should pass through (after transient)
        let mean: f64 = decimated[20..].iter().sum::<f64>() / (decimated.len() - 20) as f64;
        assert_relative_eq!(mean, 5.0, epsilon = 0.5);
    }

    #[test]
    fn test_interpolate_length() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let interpolated = interpolate(&signal, 3);
        assert_eq!(interpolated.len(), 12);
    }

    #[test]
    fn test_interpolate_length_correct() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let interpolated = interpolate(&signal, 2);
        assert_eq!(interpolated.len(), 8);
    }

    #[test]
    fn test_resample_identity() {
        let signal: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let resampled = resample_polyphase(&signal, 1, 1);
        assert_eq!(resampled.len(), signal.len());
        for (a, b) in signal.iter().zip(resampled.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_resample_up_down_roundtrip() {
        // Low-frequency signal well within passband
        let signal: Vec<f64> = (0..200).map(|i| (2.0 * std::f64::consts::PI * i as f64 * 0.005).sin()).collect();
        let up = resample_polyphase(&signal, 2, 1);
        let down = resample_polyphase(&up, 1, 2);
        let compare_len = signal.len().min(down.len());
        // Skip edges affected by filter transients
        for i in 40..compare_len - 10 {
            assert_relative_eq!(signal[i], down[i], epsilon = 0.15);
        }
    }

    #[test]
    fn test_decimate_aliasing_protection() {
        // High frequency signal should be suppressed by decimation filter
        let signal: Vec<f64> = (0..1000)
            .map(|i| (2.0 * std::f64::consts::PI * 0.45 * i as f64).sin())
            .collect();
        let decimated = decimate(&signal, 5);
        let power: f64 = decimated[10..].iter().map(|x| x * x).sum::<f64>() / (decimated.len() - 10) as f64;
        let in_power: f64 = signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64;
        assert!(power < in_power * 0.1, "High frequency should be filtered: power={}, in_power={}", power, in_power);
    }
}
