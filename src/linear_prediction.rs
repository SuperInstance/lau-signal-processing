//! Linear prediction: Levinson-Durbin recursion, reflection coefficients.

use serde::{Deserialize, Serialize};

/// LPC analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpcResult {
    /// Prediction coefficients a[1..p+1] (a[0]=1 not included).
    pub coefficients: Vec<f64>,
    /// Reflection coefficients (PARCOR).
    pub reflection: Vec<f64>,
    /// Prediction error at each order.
    pub errors: Vec<f64>,
    /// Prediction order.
    pub order: usize,
}

/// Levinson-Durbin recursion for LPC analysis.
/// Input: auto-correlation r[0..p+1], output: LPC coefficients.
pub fn levinson_durbin(autocorr: &[f64], order: usize) -> LpcResult {
    assert!(autocorr.len() > order, "Autocorrelation too short for order {}", order);

    let mut a = vec![0.0; order + 1];
    a[0] = 1.0;
    let mut e = autocorr[0];
    let mut reflection = Vec::with_capacity(order);
    let mut errors = vec![autocorr[0]];

    for m in 1..=order {
        // Compute reflection coefficient
        let mut k = -autocorr[m];
        for j in 1..m {
            k -= a[j] * autocorr[m - j];
        }
        if e.abs() < 1e-30 {
            reflection.push(0.0);
            errors.push(0.0);
            continue;
        }
        let k = k / e;
        reflection.push(k);

        // Update coefficients
        let mut new_a = a.clone();
        for j in 1..m {
            new_a[j] = a[j] + k * a[m - j];
        }
        new_a[m] = k;
        a = new_a;

        // Update error
        e *= 1.0 - k * k;
        e = e.max(0.0);
        errors.push(e);
    }

    LpcResult {
        coefficients: a[1..].to_vec(),
        reflection,
        errors,
        order,
    }
}

/// Compute reflection coefficients from LPC coefficients.
/// Uses the step-down (Schur) recursion.
pub fn lpc_to_reflection(coeffs: &[f64]) -> Vec<f64> {
    let p = coeffs.len();
    if p == 0 {
        return vec![];
    }
    // Build internal array with a[0] = 1
    let mut a = vec![0.0; p + 1];
    a[0] = 1.0;
    for (i, &c) in coeffs.iter().enumerate() {
        a[i + 1] = c;
    }

    let mut reflection = Vec::with_capacity(p);
    for m in (1..=p).rev() {
        let km = a[m];
        reflection.push(km);
        let denom = 1.0 - km * km;
        if denom.abs() < 1e-15 {
            break;
        }
        for j in 1..m {
            a[j] = (a[j] - km * a[m - j]) / denom;
        }
        a[m] = 0.0;
    }
    reflection.reverse();
    reflection
}

/// Apply LPC synthesis filter to excitation signal.
pub fn lpc_synthesize(coeffs: &[f64], excitation: &[f64]) -> Vec<f64> {
    let p = coeffs.len();
    let n = excitation.len();
    let mut output = vec![0.0; n];
    let _memory = vec![0.0; p];

    for i in 0..n {
        let mut y = excitation[i];
        for j in 0..p {
            if i > j {
                y -= coeffs[j] * output[i - 1 - j];
            }
        }
        output[i] = y;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_levinson_durbin_order1() {
        // r = [1.0, 0.5]
        let r = vec![1.0, 0.5];
        let result = levinson_durbin(&r, 1);
        assert_eq!(result.coefficients.len(), 1);
        assert_relative_eq!(result.coefficients[0], -0.5, epsilon = 1e-10);
        assert_relative_eq!(result.reflection[0], -0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_levinson_durbin_order2() {
        let r = vec![1.0, 0.5, 0.3];
        let result = levinson_durbin(&r, 2);
        assert_eq!(result.coefficients.len(), 2);
        assert_eq!(result.reflection.len(), 2);
        // Error should decrease
        assert!(result.errors[2] < result.errors[0]);
    }

    #[test]
    fn test_levinson_durbin_white_noise() {
        // White noise: autocorrelation is impulse
        let r = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let result = levinson_durbin(&r, 4);
        // All coefficients should be ~0
        for c in &result.coefficients {
            assert_relative_eq!(*c, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_reflection_coefficients_roundtrip() {
        let r = vec![1.0, 0.6, 0.4, 0.2];
        let result = levinson_durbin(&r, 3);
        let refl = lpc_to_reflection(&result.coefficients);
        assert_eq!(refl.len(), result.reflection.len());
        for (a, b) in refl.iter().zip(result.reflection.iter()) {
            assert_relative_eq!(a, b, epsilon = 0.01);
        }
    }

    #[test]
    fn test_lpc_synthesize_impulse() {
        let coeffs = vec![-0.5, -0.3];
        let excitation = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let output = lpc_synthesize(&coeffs, &excitation);
        assert_relative_eq!(output[0], 1.0, epsilon = 1e-10);
        // y[1] = 0 - (-0.5)*y[0] = 0.5
        assert_relative_eq!(output[1], 0.5, epsilon = 1e-10);
        // y[2] = 0 - (-0.5)*y[1] - (-0.3)*y[0] = 0.25 + 0.3 = 0.55
        assert_relative_eq!(output[2], 0.55, epsilon = 1e-10);
    }

    #[test]
    fn test_levinson_durbin_prediction_error_decreasing() {
        let r = vec![1.0, 0.8, 0.6, 0.4, 0.2];
        let result = levinson_durbin(&r, 4);
        for i in 1..result.errors.len() {
            assert!(result.errors[i] <= result.errors[i - 1] + 1e-10,
                "Error should be non-increasing: errors={:?}", result.errors);
        }
    }

    #[test]
    fn test_levinson_durbin_stability() {
        // Resulting filter should be stable (reflection coeffs in [-1,1])
        let r = vec![1.0, 0.7, 0.5, 0.3, 0.1, 0.05];
        let result = levinson_durbin(&r, 5);
        for k in &result.reflection {
            assert!(k.abs() <= 1.0 + 1e-10, "Reflection coefficient {} outside [-1,1]", k);
        }
    }
}
