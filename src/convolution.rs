//! Convolution: direct, FFT-based overlap-save, overlap-add.

use num_complex::Complex64;

/// Direct convolution (O(n*m)).
pub fn convolve_direct(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let m = kernel.len();
    let out_len = n + m - 1;
    let mut result = vec![0.0; out_len];
    for i in 0..n {
        for j in 0..m {
            result[i + j] += signal[i] * kernel[j];
        }
    }
    result
}

/// FFT-based convolution using overlap-add method.
pub fn convolve_overlap_add(signal: &[f64], kernel: &[f64], block_size: usize) -> Vec<f64> {
    let m = kernel.len();
    let fft_size = block_size.next_power_of_two().max((m + block_size).next_power_of_two());
    let n = fft_size;

    // Zero-pad kernel to fft_size
    let mut kernel_padded = kernel.to_vec();
    kernel_padded.resize(n, 0.0);
    let kernel_fft = fft_real(&kernel_padded);

    let out_len = signal.len() + kernel.len() - 1;
    let mut output = vec![0.0; out_len];

    let mut pos = 0;
    while pos < signal.len() {
        let end = (pos + block_size).min(signal.len());
        let block_len = end - pos;
        let mut block = vec![0.0; n];
        block[..block_len].copy_from_slice(&signal[pos..end]);

        let block_fft = fft_real(&block);
        let product: Vec<Complex64> = block_fft
            .iter()
            .zip(kernel_fft.iter())
            .map(|(a, b)| a * b)
            .collect();

        let conv_block = ifft_real(&product);
        for i in 0..(n.min(out_len.saturating_sub(pos))) {
            if pos + i < out_len {
                output[pos + i] += conv_block[i];
            }
        }
        pos += block_size;
    }
    output
}

/// FFT-based convolution using overlap-save method.
pub fn convolve_overlap_save(signal: &[f64], kernel: &[f64], block_size: usize) -> Vec<f64> {
    let m = kernel.len();
    let fft_size = block_size.next_power_of_two().max((m + block_size).next_power_of_two());
    let n = fft_size;
    let overlap = m - 1;

    // Zero-pad kernel
    let mut kernel_padded = kernel.to_vec();
    kernel_padded.resize(n, 0.0);
    let kernel_fft = fft_real(&kernel_padded);

    let out_len = signal.len() + kernel.len() - 1;
    let mut output = Vec::with_capacity(out_len);

    // Prepend zeros for overlap
    let mut extended_signal = vec![0.0; overlap];
    extended_signal.extend_from_slice(signal);
    extended_signal.resize(extended_signal.len() + n, 0.0);

    let mut pos = 0;
    while pos < extended_signal.len() - overlap {
        let block_len = n.min(extended_signal.len() - pos);
        let mut block = vec![0.0; n];
        block[..block_len].copy_from_slice(&extended_signal[pos..pos + block_len]);

        let block_fft = fft_real(&block);
        let product: Vec<Complex64> = block_fft
            .iter()
            .zip(kernel_fft.iter())
            .map(|(a, b)| a * b)
            .collect();

        let conv_block = ifft_real(&product);
        // Discard first 'overlap' samples (corrupted by circular convolution)
        let valid_start = overlap;
        for i in valid_start..n {
            if output.len() < out_len {
                output.push(conv_block[i]);
            }
        }
        pos += n - overlap;
    }
    output.truncate(out_len);
    output
}

/// Simple radix-2 FFT (Cooley-Tukey). Input must be power-of-2 length.
pub fn fft(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    if n == 1 {
        return input.to_vec();
    }
    if n == 2 {
        return vec![
            input[0] + input[1],
            input[0] - input[1],
        ];
    }
    // Bit-reversal permutation + iterative FFT
    let mut result = bit_reverse_copy(input);
    let mut size = 2;
    while size <= n {
        let half = size / 2;
        let angle = -2.0 * std::f64::consts::PI / size as f64;
        for start in (0..n).step_by(size) {
            for k in 0..half {
                let w = Complex64::from_polar(1.0, angle * k as f64);
                let t = w * result[start + k + half];
                result[start + k + half] = result[start + k] - t;
                result[start + k] = result[start + k] + t;
            }
        }
        size *= 2;
    }
    result
}

/// Inverse FFT.
pub fn ifft(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    let conjugated: Vec<Complex64> = input.iter().map(|x| x.conj()).collect();
    let mut result = fft(&conjugated);
    let scale = 1.0 / n as f64;
    for x in result.iter_mut() {
        *x = x.conj() * scale;
    }
    result
}

fn bit_reverse_copy(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    let bits = (n as f64).log2() as usize;
    let mut result = vec![Complex64::new(0.0, 0.0); n];
    for i in 0..n {
        let j = bit_reverse(i, bits);
        result[j] = input[i];
    }
    result
}

fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// FFT of real-valued input.
fn fft_real(input: &[f64]) -> Vec<Complex64> {
    let complex: Vec<Complex64> = input.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    fft(&complex)
}

/// IFFT to real-valued output.
fn ifft_real(input: &[Complex64]) -> Vec<f64> {
    ifft(input).iter().map(|x| x.re).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_direct_convolution_delta() {
        let signal = vec![1.0, 2.0, 3.0];
        let kernel = vec![1.0];
        let result = convolve_direct(&signal, &kernel);
        assert_eq!(result, signal);
    }

    #[test]
    fn test_direct_convolution_basic() {
        let signal = vec![1.0, 2.0, 3.0];
        let kernel = vec![1.0, 1.0];
        let result = convolve_direct(&signal, &kernel);
        assert_eq!(result, vec![1.0, 3.0, 5.0, 3.0]);
    }

    #[test]
    fn test_fft_basic() {
        let input = vec![Complex64::new(1.0, 0.0); 4];
        let output = fft(&input);
        assert_relative_eq!(output[0].re, 4.0, epsilon = 1e-10);
        assert_relative_eq!(output[1].re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ifft_roundtrip() {
        let input: Vec<Complex64> = vec![1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|x| Complex64::new(*x, 0.0))
            .collect();
        let transformed = fft(&input);
        let recovered = ifft(&transformed);
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_overlap_add_matches_direct() {
        let signal: Vec<f64> = (0..50).map(|i| (i as f64).sin()).collect();
        let kernel = vec![0.25, 0.5, 0.25];
        let direct = convolve_direct(&signal, &kernel);
        let ola = convolve_overlap_add(&signal, &kernel, 16);
        assert_eq!(direct.len(), ola.len());
        for (a, b) in direct.iter().zip(ola.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_overlap_save_matches_direct() {
        let signal: Vec<f64> = (0..50).map(|i| (i as f64).sin()).collect();
        let kernel = vec![0.25, 0.5, 0.25];
        let direct = convolve_direct(&signal, &kernel);
        let ols = convolve_overlap_save(&signal, &kernel, 16);
        assert_eq!(direct.len(), ols.len());
        for (a, b) in direct.iter().zip(ols.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_convolution_commutative() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.5, 0.5];
        let ab = convolve_direct(&a, &b);
        let ba = convolve_direct(&b, &a);
        assert_eq!(ab, ba);
    }
}
