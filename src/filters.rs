//! FIR and IIR filter design (Butterworth, Chebyshev, windowing, bilinear transform).

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// FIR filter coefficients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirFilter {
    /// Feed-forward taps (b coefficients).
    pub coeffs: Vec<f64>,
}

impl FirFilter {
    pub fn new(coeffs: Vec<f64>) -> Self {
        Self { coeffs }
    }

    /// Design FIR filter via windowing method.
    pub fn design_windowed(order: usize, cutoff: f64, window: &str, filter_type: &str) -> Self {
        let num_taps = order + 1;
        let mut coeffs = Vec::with_capacity(num_taps);
        let m = order as f64;

        for n in 0..num_taps {
            let nm = n as f64 - m / 2.0;
            let ideal = if filter_type == "highpass" {
                if nm.abs() < 1e-10 {
                    1.0 - 2.0 * cutoff
                } else {
                    -(2.0 * cutoff * (std::f64::consts::PI * nm).sin() / (std::f64::consts::PI * nm))
                }
            } else {
                if nm.abs() < 1e-10 {
                    2.0 * cutoff
                } else {
                    (2.0 * cutoff * std::f64::consts::PI * nm).sin() / (std::f64::consts::PI * nm)
                }
            };
            coeffs.push(ideal * apply_window(n, num_taps, window));
        }
        Self { coeffs }
    }

    /// Apply filter to input signal (direct-form convolution).
    pub fn filter(&self, input: &[f64]) -> Vec<f64> {
        let n = input.len();
        let m = self.coeffs.len();
        let mut output = vec![0.0; n];
        for i in 0..n {
            for j in 0..m {
                if i >= j {
                    output[i] += self.coeffs[j] * input[i - j];
                }
            }
        }
        output
    }
}

fn apply_window(n: usize, size: usize, window: &str) -> f64 {
    let n_f = n as f64;
    let size_f = (size - 1) as f64;
    match window {
        "hann" => 0.5 * (1.0 - (2.0 * std::f64::consts::PI * n_f / size_f).cos()),
        "hamming" => 0.54 - 0.46 * (2.0 * std::f64::consts::PI * n_f / size_f).cos(),
        "blackman" => {
            0.42 - 0.5 * (2.0 * std::f64::consts::PI * n_f / size_f).cos()
                + 0.08 * (4.0 * std::f64::consts::PI * n_f / size_f).cos()
        }
        _ => 1.0,
    }
}

/// IIR filter using second-order sections (biquad cascade).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IirFilter {
    /// Numerator (b) coefficients (combined).
    pub b: Vec<f64>,
    /// Denominator (a) coefficients (combined, a[0] = 1.0).
    pub a: Vec<f64>,
    /// Second-order sections: each is [b0, b1, b2, 1.0, a1, a2].
    pub sos: Vec<[f64; 6]>,
}

impl IirFilter {
    pub fn new(b: Vec<f64>, a: Vec<f64>) -> Self {
        Self { b, a, sos: vec![] }
    }

    /// Design Butterworth filter.
    /// Uses analog prototype → bilinear transform → SOS cascade.
    pub fn butterworth(order: usize, cutoff: f64, filter_type: &str) -> Self {
        let mut sos_list: Vec<[f64; 6]> = Vec::new();

        // Pre-warp cutoff
        let wc = (std::f64::consts::PI * cutoff).tan();
        let two = Complex64::new(2.0, 0.0);
        let _one = Complex64::new(1.0, 0.0);

        // Analog Butterworth poles: s_k = exp(j * pi * (2k+1)/(2N) + j*pi/2) for k=0..N-1
        let analog_poles: Vec<Complex64> = (0..order)
            .map(|k| {
                let theta =
                    std::f64::consts::PI * (2 * k + 1) as f64 / (2 * order) as f64 + std::f64::consts::PI / 2.0;
                Complex64::from_polar(1.0, theta) * wc
            })
            .collect();

        // Process pole pairs (and single real pole for odd orders)
        let mut idx = 0;
        while idx < analog_poles.len() {
            let p = analog_poles[idx];
            if p.im.abs() < 1e-10 {
                // Real pole → first-order section via bilinear
                // z = (2 + s) / (2 - s)
                let z = (two + p) / (two - p);
                let z_re = z.re;

                // For lowpass: numerator [1, 1] → b0 = K, b1 = K
                // For highpass: numerator [1, -1] → b0 = K, b1 = -K
                // Denominator: [1, -z_re]
                // Gain: for lowpass at DC (z=1): K*(1+1) / (1 - z_re) = desired DC gain
                // We'll normalize later, so just use K=1
                let a1 = -z_re;
                if filter_type == "lowpass" {
                    sos_list.push([1.0, 1.0, 0.0, 1.0, a1, 0.0]);
                } else {
                    sos_list.push([1.0, -1.0, 0.0, 1.0, a1, 0.0]);
                }
                idx += 1;
            } else {
                // Complex pole pair
                let z1 = (two + p) / (two - p);
                let z2 = (two + p.conj()) / (two - p.conj());

                let a1 = -(z1 + z2).re;
                let a2 = (z1 * z2).re;

                if filter_type == "lowpass" {
                    // Numerator: (1+z^{-1})^2 → [1, 2, 1]
                    sos_list.push([1.0, 2.0, 1.0, 1.0, a1, a2]);
                } else {
                    // Numerator: (1-z^{-1})^2 → [1, -2, 1]
                    sos_list.push([1.0, -2.0, 1.0, 1.0, a1, a2]);
                }
                idx += 2;
            }
        }

        // Normalize: compute gain so that DC (z=1) response = 1 for lowpass,
        // or Nyquist (z=-1) response = 1 for highpass
        let z_eval = if filter_type == "lowpass" { 1.0 } else { -1.0 };

        let mut dc_response = 1.0f64;
        for section in &sos_list {
            let num = section[0] + section[1] * z_eval + section[2] * z_eval * z_eval;
            let den = section[3] + section[4] * z_eval + section[5] * z_eval * z_eval;
            if den.abs() > 1e-15 {
                dc_response *= num / den;
            }
        }

        // Apply gain to first section
        let gain = dc_response.recip();
        if let Some(first) = sos_list.first_mut() {
            first[0] *= gain;
            first[1] *= gain;
            first[2] *= gain;
        }

        let (combined_b, combined_a) = combine_sos(&sos_list);
        Self {
            b: combined_b,
            a: combined_a,
            sos: sos_list,
        }
    }

    /// Design Chebyshev Type I filter.
    pub fn chebyshev(order: usize, cutoff: f64, ripple_db: f64, filter_type: &str) -> Self {
        let eps = (10.0_f64.powf(ripple_db / 10.0) - 1.0).sqrt();
        let wc = (std::f64::consts::PI * cutoff).tan();
        let two = Complex64::new(2.0, 0.0);

        let alpha = (1.0 / eps).asinh() / order as f64;
        let analog_poles: Vec<Complex64> = (0..order)
            .map(|k| {
                let theta = std::f64::consts::PI * (2 * k + 1) as f64 / (2 * order) as f64;
                Complex64::new(-alpha.sinh() * theta.cos(), alpha.cosh() * theta.sin()) * wc
            })
            .collect();

        let mut sos_list: Vec<[f64; 6]> = Vec::new();
        let mut idx = 0;
        while idx < analog_poles.len() {
            let p = analog_poles[idx];
            if p.im.abs() < 1e-10 {
                let z = (two + p) / (two - p);
                let a1 = -z.re;
                if filter_type == "lowpass" {
                    sos_list.push([1.0, 1.0, 0.0, 1.0, a1, 0.0]);
                } else {
                    sos_list.push([1.0, -1.0, 0.0, 1.0, a1, 0.0]);
                }
                idx += 1;
            } else {
                let z1 = (two + p) / (two - p);
                let z2 = (two + p.conj()) / (two - p.conj());
                let a1 = -(z1 + z2).re;
                let a2 = (z1 * z2).re;
                if filter_type == "lowpass" {
                    sos_list.push([1.0, 2.0, 1.0, 1.0, a1, a2]);
                } else {
                    sos_list.push([1.0, -2.0, 1.0, 1.0, a1, a2]);
                }
                idx += 2;
            }
        }

        let z_eval = if filter_type == "lowpass" { 1.0 } else { -1.0 };
        let mut dc_response = 1.0f64;
        for section in &sos_list {
            let num = section[0] + section[1] * z_eval + section[2] * z_eval * z_eval;
            let den = section[3] + section[4] * z_eval + section[5] * z_eval * z_eval;
            if den.abs() > 1e-15 {
                dc_response *= num / den;
            }
        }
        let gain = dc_response.recip();
        if let Some(first) = sos_list.first_mut() {
            first[0] *= gain;
            first[1] *= gain;
            first[2] *= gain;
        }

        let (combined_b, combined_a) = combine_sos(&sos_list);
        Self {
            b: combined_b,
            a: combined_a,
            sos: sos_list,
        }
    }

    /// Apply IIR filter using SOS (numerically stable).
    pub fn filter(&self, input: &[f64]) -> Vec<f64> {
        if self.sos.is_empty() {
            return self.filter_direct(input);
        }
        let mut signal = input.to_vec();
        for section in &self.sos {
            signal = Self::apply_sos_section(section, &signal);
        }
        signal
    }

    fn apply_sos_section(section: &[f64; 6], input: &[f64]) -> Vec<f64> {
        let b0 = section[0];
        let b1 = section[1];
        let b2 = section[2];
        let a1 = section[4];
        let a2 = section[5];
        let n = input.len();
        let mut output = vec![0.0; n];
        let mut w1 = 0.0f64;
        let mut w2 = 0.0f64;
        for i in 0..n {
            let w0 = input[i] - a1 * w1 - a2 * w2;
            output[i] = b0 * w0 + b1 * w1 + b2 * w2;
            w2 = w1;
            w1 = w0;
        }
        output
    }

    fn filter_direct(&self, input: &[f64]) -> Vec<f64> {
        let n = input.len();
        let nb = self.b.len();
        let na = self.a.len();
        let nf = nb.max(na);
        let mut w = vec![0.0; nf];
        let mut output = vec![0.0; n];
        for i in 0..n {
            let y = self.b.first().copied().unwrap_or(0.0) * input[i] + w[0];
            for j in 1..nf {
                let bj = self.b.get(j).copied().unwrap_or(0.0);
                let aj = self.a.get(j).copied().unwrap_or(0.0);
                w[j - 1] = bj * input[i] - aj * y + w.get(j).copied().unwrap_or(0.0);
            }
            output[i] = y;
        }
        output
    }
}

/// Combine second-order sections into a single transfer function.
fn combine_sos(sos: &[[f64; 6]]) -> (Vec<f64>, Vec<f64>) {
    let mut b = vec![1.0_f64];
    let mut a = vec![1.0_f64];
    for section in sos {
        let sb = [section[0], section[1], section[2]];
        let sa = [section[3], section[4], section[5]];
        b = poly_mul(&b, &sb);
        a = poly_mul(&a, &sa);
    }
    // Normalize a[0] = 1
    if a[0].abs() > 1e-15 {
        let a0 = a[0];
        b = b.iter().map(|x| x / a0).collect();
        a = a.iter().map(|x| x / a0).collect();
    }
    (b, a)
}

fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; a.len() + b.len() - 1];
    for i in 0..a.len() {
        for j in 0..b.len() {
            result[i + j] += a[i] * b[j];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fir_rectangular_window() {
        let fir = FirFilter::design_windowed(10, 0.1, "rectangular", "lowpass");
        assert_eq!(fir.coeffs.len(), 11);
    }

    #[test]
    fn test_fir_hann_window() {
        let fir = FirFilter::design_windowed(20, 0.2, "hann", "lowpass");
        assert_eq!(fir.coeffs.len(), 21);
        assert_relative_eq!(fir.coeffs[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fir_highpass() {
        let fir = FirFilter::design_windowed(10, 0.2, "hamming", "highpass");
        assert_eq!(fir.coeffs.len(), 11);
    }

    #[test]
    fn test_fir_filter_identity() {
        let fir = FirFilter::new(vec![1.0]);
        let output = fir.filter(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_fir_filter_moving_average() {
        let fir = FirFilter::new(vec![0.5, 0.5]);
        let output = fir.filter(&[2.0, 4.0, 6.0]);
        assert_relative_eq!(output[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(output[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(output[2], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_iir_butterworth_lowpass() {
        let iir = IirFilter::butterworth(4, 0.1, "lowpass");
        assert_relative_eq!(iir.a[0], 1.0, epsilon = 1e-10);
        // High frequency signal at 0.4 normalized freq
        let input: Vec<f64> = (0..2000)
            .map(|i| (2.0 * std::f64::consts::PI * 0.4 * i as f64).sin())
            .collect();
        let output = iir.filter(&input);
        let out_rms = (output[1000..].iter().map(|x| x * x).sum::<f64>() / 1000.0).sqrt();
        let in_rms = (input[1000..].iter().map(|x| x * x).sum::<f64>() / 1000.0).sqrt();
        assert!(out_rms < in_rms * 0.1, "Should attenuate: in={}, out={}", in_rms, out_rms);
    }

    #[test]
    fn test_iir_butterworth_passes_low() {
        let iir = IirFilter::butterworth(4, 0.1, "lowpass");
        let input: Vec<f64> = (0..2000)
            .map(|i| (2.0 * std::f64::consts::PI * 0.01 * i as f64 / 1000.0).sin())
            .collect();
        let output = iir.filter(&input);
        let out_rms = (output[1000..].iter().map(|x| x * x).sum::<f64>() / 1000.0).sqrt();
        let in_rms = (input[1000..].iter().map(|x| x * x).sum::<f64>() / 1000.0).sqrt();
        assert!(out_rms > in_rms * 0.5, "Should pass low freq: in={}, out={}", in_rms, out_rms);
    }

    #[test]
    fn test_iir_chebyshev() {
        let iir = IirFilter::chebyshev(4, 0.1, 1.0, "lowpass");
        assert_relative_eq!(iir.a[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_iir_filter_stability() {
        let iir = IirFilter::butterworth(2, 0.25, "lowpass");
        let output = iir.filter(&vec![1.0; 100]);
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_blackman_window() {
        let fir = FirFilter::design_windowed(20, 0.2, "blackman", "lowpass");
        assert_eq!(fir.coeffs.len(), 21);
        assert_relative_eq!(fir.coeffs[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_iir_butterworth_dc_gain() {
        let iir = IirFilter::butterworth(4, 0.1, "lowpass");
        let input = vec![1.0; 500];
        let output = iir.filter(&input);
        let steady = output[400..].iter().sum::<f64>() / 100.0;
        assert_relative_eq!(steady, 1.0, epsilon = 0.01);
    }
}
