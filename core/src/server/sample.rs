//! CPU token sampling on a logits vector: greedy at temperature 0, else temperature + top-p with a seeded RNG.

use super::engine::GenParams;

/// splitmix64: tiny, seedable, good enough for sampling
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// uniform in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Debug, Clone)]
pub struct Sampler {
    pub temperature: f32,
    pub top_p: f32,
    rng: Rng,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| {
            let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
            t.as_nanos() as u64 ^ (std::process::id() as u64) << 32
        });
        Self { temperature, top_p, rng: Rng::new(seed) }
    }

    pub fn from_params(p: &GenParams) -> Self {
        Self::new(p.temperature, p.top_p, p.seed)
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.temperature <= 0.0 || !self.temperature.is_finite() {
            return argmax(logits);
        }
        let mx = logits.iter().copied().filter(|v| !v.is_nan()).fold(f32::NEG_INFINITY, f32::max);
        if !mx.is_finite() {
            return argmax(logits);
        }
        let inv_t = 1.0 / self.temperature as f64;
        let w = |l: f32| if l.is_nan() { 0.0 } else { ((l - mx) as f64 * inv_t).exp() };
        let top_p = self.top_p as f64;
        if !(top_p < 1.0) {
            // plain temperature sampling: one pass for the sum, one for the pick
            let sum: f64 = logits.iter().map(|&l| w(l)).sum();
            let mut r = self.rng.next_f64() * sum;
            let mut last = 0;
            for (i, &l) in logits.iter().enumerate() {
                let p = w(l);
                if p > 0.0 {
                    last = i;
                    r -= p;
                    if r < 0.0 {
                        return i as u32;
                    }
                }
            }
            return last as u32;
        }
        // nucleus: grow a sorted top-k until it covers top_p of the mass (usually a few dozen tokens)
        let sum: f64 = logits.iter().map(|&l| w(l)).sum();
        let target = top_p.max(0.0) * sum;
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        let key = |i: &u32| { let l = logits[*i as usize]; if l.is_nan() { f32::NEG_INFINITY } else { l } };
        let mut k = 64.min(idx.len());
        let (cut, mass) = loop {
            if k < idx.len() {
                idx.select_nth_unstable_by(k - 1, |a, b| key(b).total_cmp(&key(a)));
            }
            idx[..k].sort_unstable_by(|a, b| key(b).total_cmp(&key(a)));
            let mut acc = 0.0;
            let mut found = None;
            for (j, &i) in idx[..k].iter().enumerate() {
                acc += w(logits[i as usize]);
                if acc >= target {
                    found = Some(j + 1);
                    break;
                }
            }
            match found {
                Some(c) => break (c, acc),
                None if k == idx.len() => break (k, acc),
                None => k = (k * 4).min(idx.len()),
            }
        };
        let mut r = self.rng.next_f64() * mass;
        for &i in &idx[..cut] {
            r -= w(logits[i as usize]);
            if r < 0.0 {
                return i;
            }
        }
        idx[cut - 1]
    }
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best.0 as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits() -> Vec<f32> {
        // probabilities at T=1 ~ softmax([3, 2, 1, 0, -1, ...])
        let mut l = vec![-10.0f32; 1000];
        l[17] = 3.0;
        l[400] = 2.0;
        l[3] = 1.0;
        l[999] = 0.0;
        l
    }

    fn hist(s: &mut Sampler, l: &[f32], n: usize) -> std::collections::HashMap<u32, usize> {
        let mut h = std::collections::HashMap::new();
        for _ in 0..n {
            *h.entry(s.sample(l)).or_insert(0) += 1;
        }
        h
    }

    #[test]
    fn greedy_is_argmax() {
        let l = logits();
        let mut s = Sampler::new(0.0, 0.9, Some(1));
        for _ in 0..10 {
            assert_eq!(s.sample(&l), 17);
        }
        assert_eq!(argmax(&[f32::NAN, 1.0, 2.0, -1.0]), 2);
    }

    #[test]
    fn seeded_is_deterministic() {
        let l = logits();
        let a: Vec<u32> = { let mut s = Sampler::new(1.0, 1.0, Some(42)); (0..200).map(|_| s.sample(&l)).collect() };
        let b: Vec<u32> = { let mut s = Sampler::new(1.0, 1.0, Some(42)); (0..200).map(|_| s.sample(&l)).collect() };
        let c: Vec<u32> = { let mut s = Sampler::new(1.0, 1.0, Some(43)); (0..200).map(|_| s.sample(&l)).collect() };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn temperature_matches_softmax() {
        let l = logits();
        let n = 200_000;
        let h = hist(&mut Sampler::new(1.0, 1.0, Some(7)), &l, n);
        let z: f64 = l.iter().map(|&x| (x as f64).exp()).sum();
        for i in [17u32, 400, 3, 999] {
            let p = (l[i as usize] as f64).exp() / z;
            let got = *h.get(&i).unwrap_or(&0) as f64 / n as f64;
            assert!((got - p).abs() < 0.01, "token {i}: {got} vs {p}");
        }
        // T=0.5 sharpens: the top token gets more mass than at T=1
        let h2 = hist(&mut Sampler::new(0.5, 1.0, Some(7)), &l, 20_000);
        assert!(h2[&17] as f64 / 20_000.0 > 0.85);
    }

    #[test]
    fn top_p_truncates() {
        let l = logits();
        // p(17) ~ 0.64, p(400) ~ 0.24: top_p 0.5 keeps only 17, 0.8 keeps {17, 400}
        let h = hist(&mut Sampler::new(1.0, 0.5, Some(3)), &l, 5_000);
        assert_eq!(h.len(), 1);
        assert!(h.contains_key(&17));
        let h = hist(&mut Sampler::new(1.0, 0.8, Some(3)), &l, 20_000);
        assert_eq!(h.keys().copied().collect::<std::collections::BTreeSet<_>>(), [17, 400].into());
        let r = h[&17] as f64 / h[&400] as f64; // renormalized ratio stays e^1
        assert!((r - std::f64::consts::E).abs() < 0.2, "{r}");
    }

    #[test]
    fn top_p_grows_past_initial_k() {
        // flat distribution over 129k tokens: the nucleus is far larger than the first top-k window
        let l = vec![0.0f32; 129_280];
        let mut s = Sampler::new(1.0, 0.95, Some(9));
        let mut seen = std::collections::HashSet::new();
        for _ in 0..300 {
            let t = s.sample(&l);
            assert!((t as usize) < l.len());
            seen.insert(t);
        }
        assert!(seen.len() > 290);
    }
}
