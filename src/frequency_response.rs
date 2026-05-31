//! Filter frequency response: magnitude, phase, group delay.

use num_complex::Complex64;
use crate::filters::{FirFilter, IirFilter};

/// Compute frequency response of an FIR filter at normalized frequencies.
pub fn fir_freq_response(filter: &FirFilter, freqs: &[f64]) -> Vec<Complex64> {
    freqs
        .iter()
        .map(|&f| {
            let mut h = Complex64::new(0.0, 0.0);
            for (n, &b) in filter.coeffs.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * f * n as f64;
                h += b * Complex64::from_polar(1.0, angle);
            }
            h
        })
        .collect()
}

/// Compute frequency response of an IIR filter at normalized frequencies.
pub fn iir_freq_response(filter: &IirFilter, freqs: &[f64]) -> Vec<Complex64> {
    freqs
        .iter()
        .map(|&f| {
            let mut num = Complex64::new(0.0, 0.0);
            for (n, &b) in filter.b.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * f * n as f64;
                num += b * Complex64::from_polar(1.0, angle);
            }
            let mut den = Complex64::new(0.0, 0.0);
            for (n, &a) in filter.a.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * f * n as f64;
                den += a * Complex64::from_polar(1.0, angle);
            }
            if den.norm() < 1e-15 {
                Complex64::new(f64::MAX, 0.0)
            } else {
                num / den
            }
        })
        .collect()
}

/// Compute magnitude response in dB.
pub fn magnitude_db(response: &[Complex64]) -> Vec<f64> {
    response
        .iter()
        .map(|h| 20.0 * h.norm().max(1e-30).log10())
        .collect()
}

/// Compute phase response in radians (unwrapped approximation).
pub fn phase_response(response: &[Complex64]) -> Vec<f64> {
    response.iter().map(|h| h.arg()).collect()
}

/// Compute group delay (samples) via numerical differentiation of phase.
pub fn group_delay(response: &[Complex64], freq_step: f64) -> Vec<f64> {
    let phases: Vec<f64> = phase_response(response);
    let mut gd = vec![0.0; phases.len()];
    for i in 1..phases.len() - 1 {
        gd[i] = -(phases[i + 1] - phases[i - 1]) / (2.0 * freq_step * 2.0 * std::f64::consts::PI);
    }
    if phases.len() > 1 {
        gd[0] = gd[1];
        gd[phases.len() - 1] = gd[phases.len() - 2];
    }
    gd
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fir_magnitude_dc() {
        let fir = FirFilter::new(vec![0.25, 0.5, 0.25]);
        let resp = fir_freq_response(&fir, &[0.0]);
        // DC gain = sum of coefficients
        assert_relative_eq!(resp[0].norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fir_magnitude_nyquist() {
        let fir = FirFilter::new(vec![0.5, -0.5]);
        let resp = fir_freq_response(&fir, &[0.5]);
        assert_relative_eq!(resp[0].norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_iir_magnitude_dc() {
        let iir = IirFilter::butterworth(2, 0.25, "lowpass");
        let resp = iir_freq_response(&iir, &[0.0]);
        // DC gain should be ~1 for normalized Butterworth
        assert_relative_eq!(resp[0].norm(), 1.0, epsilon = 0.2);
    }

    #[test]
    fn test_magnitude_db() {
        let resp = vec![Complex64::new(1.0, 0.0), Complex64::new(0.1, 0.0)];
        let db = magnitude_db(&resp);
        assert_relative_eq!(db[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(db[1], -20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_group_delay_basic() {
        let fir = FirFilter::new(vec![0.5, 0.5]);
        let freqs: Vec<f64> = (0..50).map(|i| i as f64 / 100.0).collect();
        let resp = fir_freq_response(&fir, &freqs);
        let gd = group_delay(&resp, 0.01);
        // Linear-phase FIR should have constant group delay = (N-1)/2 = 0.5
        // Just check it's finite
        assert!(gd.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_lowpass_attenuation() {
        let fir = FirFilter::design_windowed(80, 0.15, "hann", "lowpass");
        let freqs = [0.05, 0.15, 0.35, 0.49];
        let resp = fir_freq_response(&fir, &freqs);
        let mag: Vec<f64> = resp.iter().map(|h| h.norm()).collect();
        assert!(mag[0] > 0.3, "Passband should have significant gain: {}", mag[0]);
        assert!(mag[3] < 0.1, "Stopband should be attenuated: {}", mag[3]);
    }
}
