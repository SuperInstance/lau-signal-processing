//! Agent telemetry signal monitoring — filter noise from agent telemetry streams.

use serde::{Deserialize, Serialize};
use crate::filters::IirFilter;
use crate::correlation::autocorrelation;
use crate::linear_prediction::levinson_durbin;

/// Telemetry sample: timestamped metric value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySample {
    pub timestamp: f64,
    pub value: f64,
}

/// Telemetry channel with samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryChannel {
    pub name: String,
    pub samples: Vec<TelemetrySample>,
}

impl TelemetryChannel {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            samples: Vec::new(),
        }
    }

    pub fn add(&mut self, timestamp: f64, value: f64) {
        self.samples.push(TelemetrySample { timestamp, value });
    }

    pub fn values(&self) -> Vec<f64> {
        self.samples.iter().map(|s| s.value).collect()
    }

    pub fn timestamps(&self) -> Vec<f64> {
        self.samples.iter().map(|s| s.timestamp).collect()
    }
}

/// Filter noise from a telemetry channel using Butterworth lowpass.
pub fn filter_telemetry_channel(channel: &TelemetryChannel, cutoff: f64, order: usize) -> TelemetryChannel {
    let values = channel.values();
    let iir = IirFilter::butterworth(order, cutoff, "lowpass");
    let filtered = iir.filter(&values);
    let mut result = TelemetryChannel::new(&channel.name);
    for (i, sample) in channel.samples.iter().enumerate() {
        result.add(sample.timestamp, filtered[i]);
    }
    result
}

/// Detect anomalies in telemetry using deviation from predicted signal.
pub fn detect_anomalies(channel: &TelemetryChannel, lpc_order: usize, threshold: f64) -> Vec<usize> {
    let values = channel.values();
    if values.len() < lpc_order + 1 {
        return vec![];
    }

    // Compute LPC
    let max_lag = lpc_order + 1;
    let ac = autocorrelation(&values, Some(max_lag));
    let lpc = levinson_durbin(&ac, lpc_order);

    // Compute prediction error
    let mut prediction_error = vec![0.0; values.len()];
    for i in lpc_order..values.len() {
        let mut predicted = 0.0;
        for j in 0..lpc_order {
            predicted -= lpc.coefficients[j] * values[i - 1 - j];
        }
        prediction_error[i] = (values[i] - predicted).abs();
    }

    // Find anomalies
    let mean_error: f64 = prediction_error[lpc_order..].iter().sum::<f64>() / (values.len() - lpc_order) as f64;
    prediction_error
        .iter()
        .enumerate()
        .filter(|(_, &e)| e > threshold * mean_error)
        .map(|(i, _)| i)
        .collect()
}

/// Compute telemetry statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryStats {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub zero_crossing_rate: f64,
}

impl TelemetryStats {
    pub fn compute(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self { mean: 0.0, std_dev: 0.0, min: 0.0, max: 0.0, zero_crossing_rate: 0.0 };
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let zero_crossings = values
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();
        let zero_crossing_rate = zero_crossings as f64 / (values.len() - 1).max(1) as f64;
        Self { mean, std_dev, min, max, zero_crossing_rate }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_telemetry_channel_basic() {
        let mut ch = TelemetryChannel::new("cpu");
        ch.add(0.0, 1.0);
        ch.add(1.0, 2.0);
        ch.add(2.0, 3.0);
        assert_eq!(ch.values(), vec![1.0, 2.0, 3.0]);
        assert_eq!(ch.name, "cpu");
    }

    #[test]
    fn test_filter_telemetry() {
        let mut ch = TelemetryChannel::new("latency");
        for i in 0..500 {
            let noise = ((i as u64).wrapping_mul(7919) % 100) as f64 / 50.0 - 1.0;
            ch.add(i as f64, (0.02 * i as f64).sin() + noise * 0.1);
        }
        let filtered = filter_telemetry_channel(&ch, 0.05, 4);
        assert_eq!(filtered.samples.len(), ch.samples.len());
        // After sufficient samples, filtered should be smoother
        let orig_var: f64 = {
            let vals = ch.values();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            vals[200..].iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (vals.len() - 200) as f64
        };
        let filt_var: f64 = {
            let vals = filtered.values();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            vals[200..].iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (vals.len() - 200) as f64
        };
        assert!(filt_var <= orig_var * 1.5, "Filtered variance should not explode");
    }

    #[test]
    fn test_detect_anomalies() {
        let mut ch = TelemetryChannel::new("errors");
        for i in 0..100 {
            ch.add(i as f64, (0.1 * i as f64).sin());
        }
        // Inject anomaly
        ch.samples[50].value = 100.0;
        let anomalies = detect_anomalies(&ch, 5, 5.0);
        assert!(anomalies.contains(&50), "Should detect anomaly at index 50, got {:?}", anomalies);
    }

    #[test]
    fn test_telemetry_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = TelemetryStats::compute(&values);
        assert_relative_eq!(stats.mean, 3.0, epsilon = 1e-10);
        assert_relative_eq!(stats.min, 1.0, epsilon = 1e-10);
        assert_relative_eq!(stats.max, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_telemetry_stats_empty() {
        let stats = TelemetryStats::compute(&[]);
        assert_eq!(stats.mean, 0.0);
    }

    #[test]
    fn test_filter_telemetry_preserves_timestamps() {
        let mut ch = TelemetryChannel::new("mem");
        for i in 0..50 {
            ch.add(i as f64 * 0.1, i as f64);
        }
        let filtered = filter_telemetry_channel(&ch, 0.2, 2);
        for (orig, filt) in ch.samples.iter().zip(filtered.samples.iter()) {
            assert_relative_eq!(orig.timestamp, filt.timestamp, epsilon = 1e-10);
        }
    }
}
