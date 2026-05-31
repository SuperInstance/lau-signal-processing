//! Auto-correlation and cross-correlation.

/// Compute auto-correlation of a signal (biased estimator, normalized by r[0]).
/// Returns correlations for lags 0..max_lag (defaults to signal.len()).
pub fn autocorrelation(signal: &[f64], max_lag: Option<usize>) -> Vec<f64> {
    let n = signal.len();
    let max_lag = max_lag.unwrap_or(n).min(n);
    let r0: f64 = signal.iter().map(|x| x * x).sum::<f64>() / n as f64;
    if r0 < 1e-30 {
        return vec![0.0; max_lag];
    }
    (0..max_lag)
        .map(|lag| {
            let sum: f64 = signal[..n - lag]
                .iter()
                .zip(&signal[lag..])
                .map(|(a, b)| a * b)
                .sum();
            sum / (n as f64 * r0)
        })
        .collect()
}

/// Compute cross-correlation between two signals.
/// Returns correlations for lags -(max_lag-1)..max_lag.
pub fn cross_correlation(a: &[f64], b: &[f64], max_lag: Option<usize>) -> Vec<f64> {
    let n = a.len().max(b.len());
    let max_lag = max_lag.unwrap_or(n);
    let mut result = Vec::with_capacity(2 * max_lag - 1);

    let energy_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let energy_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm = energy_a * energy_b;
    if norm < 1e-30 {
        return vec![0.0; 2 * max_lag - 1];
    }

    // Negative lags: b shifted right
    for lag in (1..max_lag).rev() {
        let mut sum = 0.0;
        for i in lag..a.len() {
            if i - lag < b.len() {
                sum += a[i] * b[i - lag];
            }
        }
        result.push(sum / norm);
    }
    // Zero lag
    {
        let mut sum = 0.0;
        for i in 0..a.len().min(b.len()) {
            sum += a[i] * b[i];
        }
        result.push(sum / norm);
    }
    // Positive lags: a shifted right
    for lag in 1..max_lag {
        let mut sum = 0.0;
        for i in 0..a.len() {
            if i + lag < b.len() {
                sum += a[i] * b[i + lag];
            }
        }
        result.push(sum / norm);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_autocorrelation_peak_at_zero() {
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let ac = autocorrelation(&signal, None);
        assert_relative_eq!(ac[0], 1.0, epsilon = 1e-10);
        for i in 1..ac.len() {
            assert!(ac[i] <= 1.0 + 1e-10);
        }
    }

    #[test]
    fn test_autocorrelation_constant() {
        let signal = vec![3.0, 3.0, 3.0, 3.0];
        let ac = autocorrelation(&signal, Some(4));
        assert_relative_eq!(ac[0], 1.0, epsilon = 1e-10);
        // For short constant signal with biased estimator, lag>0 < 1.0
        assert!(ac[1] > 0.5, "Constant signal should have high correlation");
    }

    #[test]
    fn test_autocorrelation_periodic() {
        let period = 20;
        let signal: Vec<f64> = (0..200)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin())
            .collect();
        let ac = autocorrelation(&signal, Some(40));
        assert_relative_eq!(ac[0], 1.0, epsilon = 1e-10);
        assert!(ac[period] > 0.9, "Periodic peak at lag {}: {}", period, ac[period]);
    }

    #[test]
    fn test_cross_correlation_identical() {
        let signal = vec![1.0, 2.0, 3.0];
        let cc = cross_correlation(&signal, &signal, Some(3));
        // Peak should be at the center (zero lag)
        let center = cc.len() / 2;
        let max_val = cc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert_relative_eq!(cc[center], max_val, epsilon = 1e-10);
    }

    #[test]
    fn test_autocorrelation_noise() {
        // Use a simple PRNG that doesn't overflow
        let mut state: u64 = 42;
        let signal: Vec<f64> = (0..1000)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state as f64 / u64::MAX as f64) * 2.0 - 1.0
            })
            .collect();
        let ac = autocorrelation(&signal, Some(50));
        assert_relative_eq!(ac[0], 1.0, epsilon = 1e-10);
        for i in 5..50 {
            assert!(ac[i].abs() < 0.15, "Lag {} auto-correlation {} too high for noise", i, ac[i]);
        }
    }

    #[test]
    fn test_cross_correlation_shifted() {
        let a = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0, 0.0];
        let cc = cross_correlation(&a, &b, Some(3));
        assert_eq!(cc.len(), 5);
    }
}
