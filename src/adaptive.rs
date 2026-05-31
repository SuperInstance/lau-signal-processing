//! Adaptive filters: LMS and RLS.

use serde::{Deserialize, Serialize};

/// LMS (Least Mean Squares) adaptive filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmsFilter {
    /// Filter weights.
    pub weights: Vec<f64>,
    /// Step size (learning rate).
    pub mu: f64,
}

impl LmsFilter {
    pub fn new(taps: usize, mu: f64) -> Self {
        Self {
            weights: vec![0.0; taps],
            mu,
        }
    }

    /// Run LMS on input/reference pair, returning the output and error.
    pub fn adapt(&mut self, input: &[f64], desired: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = input.len();
        let m = self.weights.len();
        let mut output = vec![0.0; n];
        let mut error = vec![0.0; n];

        for i in 0..n {
            // Compute output
            let mut y = 0.0;
            for j in 0..m {
                if i >= j {
                    y += self.weights[j] * input[i - j];
                }
            }
            output[i] = y;
            error[i] = desired[i] - y;

            // Update weights
            for j in 0..m {
                if i >= j {
                    self.weights[j] += 2.0 * self.mu * error[i] * input[i - j];
                }
            }
        }
        (output, error)
    }
}

/// RLS (Recursive Least Squares) adaptive filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlsFilter {
    /// Filter weights.
    pub weights: Vec<f64>,
    /// Forgetting factor (0 < lambda <= 1).
    pub lambda: f64,
    /// Inverse correlation matrix (flattened, MxM).
    pub p: Vec<f64>,
    /// Regularization.
    pub delta: f64,
}

impl RlsFilter {
    pub fn new(taps: usize, lambda: f64, delta: f64) -> Self {
        let m = taps;
        let p = vec![0.0; m * m];
        // Initialize P as delta * I
        let mut p = p;
        for i in 0..m {
            p[i * m + i] = 1.0 / delta;
        }
        Self {
            weights: vec![0.0; taps],
            lambda,
            p,
            delta,
        }
    }

    /// Run RLS on input/desired pair.
    pub fn adapt(&mut self, input: &[f64], desired: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = input.len();
        let m = self.weights.len();
        let mut output = vec![0.0; n];
        let mut error = vec![0.0; n];

        for i in 0..n {
            // Build input vector
            let mut x = vec![0.0; m];
            for j in 0..m {
                if i >= j {
                    x[j] = input[i - j];
                }
            }

            // Compute output: y = w^T * x
            let y: f64 = self.weights.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum();
            output[i] = y;
            error[i] = desired[i] - y;

            // Pi = P * x
            let mut pi = vec![0.0; m];
            for r in 0..m {
                for c in 0..m {
                    pi[r] += self.p[r * m + c] * x[c];
                }
            }

            // k = Pi / (lambda + x^T * Pi)
            let xtpi: f64 = x.iter().zip(pi.iter()).map(|(xi, pi_i)| xi * pi_i).sum();
            let denom = self.lambda + xtpi;
            if denom.abs() < 1e-30 {
                continue;
            }
            let k: Vec<f64> = pi.iter().map(|p| p / denom).collect();

            // Update weights
            for j in 0..m {
                self.weights[j] += k[j] * error[i];
            }

            // Update P: P = (P - k*x^T*P) / lambda
            let mut new_p = vec![0.0; m * m];
            for r in 0..m {
                for c in 0..m {
                    let kx = k[r] * {
                        let mut dot = 0.0;
                        for j in 0..m {
                            dot += x[j] * self.p[j * m + c];
                        }
                        dot
                    };
                    new_p[r * m + c] = (self.p[r * m + c] - kx) / self.lambda;
                }
            }
            self.p = new_p;
        }
        (output, error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lms_convergence() {
        // System identification: adapt filter to match a known system
        let system = vec![0.5, 0.3, 0.2];
        let mut lms = LmsFilter::new(3, 0.01);
        let input: Vec<f64> = (0..500).map(|i| ((i * 7919) % 1000) as f64 / 500.0 - 1.0).collect();
        let desired: Vec<f64> = input
            .windows(3)
            .map(|w| w[0] * system[0] + w[1] * system[1] + w[2] * system[2])
            .collect();
        // Pad desired to match input length
        let mut desired_full = vec![0.0; input.len()];
        desired_full[..desired.len()].copy_from_slice(&desired);

        let (_output, error) = lms.adapt(&input, &desired_full);

        // Error should decrease over time
        let late_error: f64 = error[400..].iter().map(|e| e * e).sum::<f64>() / 100.0;
        let early_error: f64 = error[..100].iter().map(|e| e * e).sum::<f64>() / 100.0;
        assert!(late_error < early_error * 0.5, "LMS should converge: early={}, late={}", early_error, late_error);
    }

    #[test]
    fn test_lms_weights_converge() {
        let system = vec![1.0];
        let mut lms = LmsFilter::new(1, 0.05);
        let input: Vec<f64> = (0..200).map(|i| ((i * 7919) % 1000) as f64 / 500.0 - 1.0).collect();
        let desired: Vec<f64> = input.iter().map(|x| x * system[0]).collect();
        lms.adapt(&input, &desired);
        assert_relative_eq!(lms.weights[0], 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_rls_convergence() {
        let system = vec![0.5, 0.3, 0.2];
        let mut rls = RlsFilter::new(3, 0.99, 1.0);
        let input: Vec<f64> = (0..500).map(|i| ((i * 7919) % 1000) as f64 / 500.0 - 1.0).collect();
        let desired: Vec<f64> = input
            .windows(3)
            .map(|w| w[0] * system[0] + w[1] * system[1] + w[2] * system[2])
            .collect();
        let mut desired_full = vec![0.0; input.len()];
        desired_full[..desired.len()].copy_from_slice(&desired);

        let (_output, error) = rls.adapt(&input, &desired_full);

        let late_error: f64 = error[400..].iter().map(|e| e * e).sum::<f64>() / 100.0;
        assert!(late_error < 0.1, "RLS should converge fast, error={}", late_error);
    }

    #[test]
    fn test_rls_weights_converge() {
        let system = vec![1.0];
        let mut rls = RlsFilter::new(1, 0.99, 1.0);
        let input: Vec<f64> = (0..100).map(|i| ((i * 7919) % 1000) as f64 / 500.0 - 1.0).collect();
        let desired: Vec<f64> = input.iter().map(|x| x * system[0]).collect();
        rls.adapt(&input, &desired);
        assert_relative_eq!(rls.weights[0], 1.0, epsilon = 0.05);
    }

    #[test]
    fn test_lms_noise_cancellation() {
        // Reference: correlated noise
        let signal: Vec<f64> = (0..500).map(|i| (0.05 * i as f64).sin()).collect();
        let noise: Vec<f64> = (0..500).map(|i| (0.3 * i as f64).sin()).collect();
        let noisy: Vec<f64> = signal.iter().zip(noise.iter()).map(|(s, n)| s + n).collect();
        let ref_noise: Vec<f64> = (0..500).map(|i| (0.3 * i as f64 + 0.5).sin()).collect();

        let mut lms = LmsFilter::new(8, 0.01);
        let (_output, error) = lms.adapt(&ref_noise, &noisy);

        // Error should approximate the original signal (noise removed)
        let late_error_power: f64 = error[300..].iter().map(|e| e * e).sum::<f64>() / 200.0;
        let signal_power: f64 = signal[300..].iter().map(|s| s * s).sum::<f64>() / 200.0;
        // Error power should be closer to signal power than noise power
        assert!(late_error_power < signal_power * 5.0);
    }
}
