//! CPU expert kernel benchmark: weight GB/s per format and thread count over 64 experts (far beyond L3),
//! ms per layer for 6 mixed-type experts, and the pool's fixed per-call overhead.
//! cargo run --release --bin cpubench [-- quick] [fmt=Q2_0]

use dsx::cpu::moe::{CpuMoe, ExpertW};
use dsx::cpu::pool::{Ctx, Pool};
use dsx::cpu::qdot::{QAct, dot_rows};
use dsx::gguf::GType;
use half::f16;
use std::time::{Duration, Instant};

const DIM: usize = 5120;
const INTER: usize = 2304;

struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn normal(&mut self) -> f32 {
        let a = ((self.u64() >> 40) as f32 / (1u64 << 24) as f32).max(1e-7);
        let b = (self.u64() >> 40) as f32 / (1u64 << 24) as f32;
        (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    }
}

fn rand_rows(ty: GType, nr: usize, cols: usize, rng: &mut Rng) -> Vec<u8> {
    let n = ty.row_bytes(cols) * nr;
    let mut v = vec![0u8; n.div_ceil(8) * 8];
    for c in v.chunks_exact_mut(8) {
        c.copy_from_slice(&rng.u64().to_le_bytes());
    }
    v.truncate(n);
    let (_, bs) = ty.block();
    let (at, d) = match ty {
        GType::Q2_0 | GType::Q8_0 | GType::Iq2Xxs | GType::Iq2Xs | GType::Iq2S | GType::Iq1S => (0, 0.02),
        GType::Ptq1_0 => (26, 0.02),
        _ => (108, 0.001),
    };
    for blk in v.chunks_exact_mut(bs) {
        blk[at..at + 2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
    }
    v
}

struct Ex {
    gu: GType,
    dn: GType,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
}

fn make(types: &[(GType, GType)], n: usize, rng: &mut Rng) -> Vec<Ex> {
    (0..n)
        .map(|i| {
            let (gu, dn) = types[i % types.len()];
            Ex {
                gu,
                dn,
                gate: rand_rows(gu, INTER, DIM, rng),
                up: rand_rows(gu, INTER, DIM, rng),
                down: rand_rows(dn, DIM, INTER, rng),
            }
        })
        .collect()
}

fn views(exs: &[Ex]) -> Vec<ExpertW<'_>> {
    exs.iter()
        .map(|e| ExpertW {
            gu: e.gu,
            gate: &e.gate,
            up: &e.up,
            dn: e.dn,
            down: &e.down,
        })
        .collect()
}

/// Layer calls of 6 experts each, cycling through all of them `passes` times: (GB/s of weights, ms per call).
fn layer_rate(moe: &mut CpuMoe, ew: &[ExpertW], t: usize, passes: usize, rng: &mut Rng) -> (f64, f64) {
    let x: Vec<f32> = (0..t * DIM).map(|_| rng.normal()).collect();
    let w = vec![1.0f32 / 6.0; 6 * t];
    let mut out = vec![0f32; t * DIM];
    let groups = ew.len() / 6;
    let mut call = |g: usize| moe.forward(&ew[6 * g..6 * g + 6], DIM, &x, t, &w, &mut out);
    for g in 0..groups {
        call(g); // warm-up pass (page faults, pool spin-up)
    }
    let t0 = Instant::now();
    let mut bytes = 0usize;
    for _ in 0..passes {
        for g in 0..groups {
            call(g);
            bytes += ew[6 * g..6 * g + 6].iter().map(|e| e.bytes()).sum::<usize>();
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    (bytes as f64 / dt / 1e9, dt / (passes * groups) as f64 * 1e3)
}

fn cpus(n: usize) -> Vec<usize> {
    match n {
        1 => vec![0],
        8 => (0..8).map(|c| 2 * c).collect(), // one thread per P-core
        _ => (0..n).collect(),
    }
}

fn main() {
    let quick = std::env::args().any(|a| a == "quick");
    let nexp = if quick { 12 } else { 64 };
    let mut rng = Rng(0x2545f4914f6cdd1d);
    use GType::*;

    // in-cache single-thread kernel rate on a P-core: the compute ceiling per thread
    dsx::cpu::pool::pin_to(0);
    println!("in-cache kernel rate, 1 thread on a P-core (64 rows x 5120, L2-resident)");
    for ty in [Q2_0, Ptq1_0, Q3K, Q8_0, Iq2Xxs, Iq2Xs, Iq2S, Iq1S] {
        let rows = rand_rows(ty, 64, DIM, &mut rng);
        let mut line = format!("  {ty:?}:");
        for t in [1usize, 6] {
            let x: Vec<f32> = (0..t * DIM).map(|_| rng.normal()).collect();
            let act = QAct::new(&x, t, DIM, &[ty]);
            let mut out = vec![0f32; t * 64];
            let reps = 2000;
            let t0 = Instant::now();
            for _ in 0..reps {
                dot_rows(ty, &rows, DIM, &act, &mut out);
            }
            let gbs = (rows.len() * reps) as f64 / t0.elapsed().as_secs_f64() / 1e9;
            line += &format!("  T={t} {gbs:5.1} GB/s");
        }
        println!("{line}");
    }

    // the caller spins while workers run: keep it off the pool's CPUs (16 = first E-core)
    dsx::cpu::pool::pin_to(16);
    println!(
        "\nweights streamed from RAM, {nexp} experts per format, calls of 6 experts (all T tokens routed to each)"
    );
    println!(
        "{:8} {:>7} {:>16} {:>16}",
        "format", "threads", "T=1 GB/s (ms)", "T=6 GB/s (ms)"
    );
    let only = std::env::args().find_map(|a| a.strip_prefix("fmt=").map(str::to_string));
    for ty in [Q2_0, Ptq1_0, Q3K, Q8_0, Iq2Xxs, Iq2Xs, Iq2S, Iq1S] {
        if only.as_ref().is_some_and(|o| *o != format!("{ty:?}")) {
            continue;
        }
        let n = if ty == Q8_0 { nexp.min(24) } else { nexp };
        let exs = make(&[(ty, ty)], n, &mut rng);
        let ew = views(&exs);
        for th in [1usize, 8, 16] {
            let mut moe = CpuMoe::new(Pool::new(&cpus(th), Duration::from_millis(5)));
            let passes = if th == 1 { 1 } else { 3 };
            let (a, am) = layer_rate(&mut moe, &ew, 1, passes, &mut rng);
            let (b, bm) = layer_rate(&mut moe, &ew, 6, passes, &mut rng);
            println!(
                "{:8} {:>7} {:>8.1} ({:5.2}) {:>8.1} ({:5.2})",
                format!("{ty:?}"),
                th,
                a,
                am,
                b,
                bm
            );
        }
    }

    println!("\nmixed layer: 6 experts per call over {nexp}, gate+up / down types cycling");
    let mix = [
        (Q2_0, Q3K),
        (Ptq1_0, Q2_0),
        (Q2_0, Q2_0),
        (Q3K, Q3K),
        (Ptq1_0, Ptq1_0),
        (Q2_0, Ptq1_0),
    ];
    // the same mix with IQ2_XXS in place of Q2_0 (the codebook 2-bit tier)
    let mix_iq = mix.map(|(a, b)| (if a == Q2_0 { Iq2Xxs } else { a }, if b == Q2_0 { Iq2Xxs } else { b }));
    for (name, m) in [("Q2_0", mix), ("IQ2_XXS", mix_iq)] {
        let exs = make(&m, nexp, &mut rng);
        let ew = views(&exs);
        let per6: f64 = ew.iter().map(|e| e.bytes()).sum::<usize>() as f64 / ew.len() as f64 * 6.0 / 1e6;
        println!("  2-bit tier {name}: {per6:.1} MB of weights per layer call");
        for th in [8usize, 16] {
            let mut moe = CpuMoe::new(Pool::new(&cpus(th), Duration::from_millis(5)));
            for t in [1usize, 6] {
                let (g, ms) = layer_rate(&mut moe, &ew, t, 3, &mut rng);
                println!("  {th:2} threads T={t}: {ms:.3} ms per layer ({g:.1} GB/s)");
            }
        }
    }
    let exs = make(&mix, nexp, &mut rng);
    let ew = views(&exs);

    // the same 2 experts (~18 MB, under the 25 MB L3) as all 6 of a call: weights come from cache after the first call,
    // the ceiling for a prefetch-into-L3 scheme
    println!("\nL3-resident: 6 experts per call drawn from the same 2 (~{:.0} MB), engine's 14 threads", (ew[0].bytes() + ew[1].bytes()) as f64 / 1e6);
    let hot: Vec<ExpertW> = (0..6).map(|i| ew[i % 2].clone()).collect();
    {
        let mut moe = CpuMoe::new(Pool::new(&(0..14).collect::<Vec<_>>(), Duration::from_millis(5)));
        let (g, ms) = layer_rate(&mut moe, &hot, 1, 200, &mut rng);
        println!("  T=1: {ms:.3} ms per call ({g:.1} GB/s of weights touched)");
        let (g, ms) = layer_rate(&mut moe, &ew, 1, 3, &mut rng);
        println!("  from RAM, same threads: {ms:.3} ms per call ({g:.1} GB/s)");
    }

    println!("\nfixed overhead per call, 16 threads");
    let pool = Pool::new(&cpus(16), Duration::from_millis(2));
    let empty = |c: &Ctx| c.barrier();
    let time = |f: &mut dyn FnMut(), n: usize, gap: Duration| -> f64 {
        let mut tot = 0.0;
        for _ in 0..n {
            if gap > Duration::from_millis(1) {
                std::thread::sleep(gap);
            } else {
                let t = Instant::now();
                while t.elapsed() < gap {} // caller busy elsewhere (e.g. waiting on the GPU)
            }
            let t0 = Instant::now();
            f();
            tot += t0.elapsed().as_secs_f64();
        }
        tot / n as f64 * 1e6
    };
    for _ in 0..1000 {
        pool.run(&empty);
    }
    println!(
        "  pool.run + 1 barrier, hot:          {:6.2} us",
        time(&mut || pool.run(&empty), 20000, Duration::ZERO)
    );
    println!(
        "  pool.run + 1 barrier, 0.5 ms busy gap:{:6.2} us",
        time(&mut || pool.run(&empty), 2000, Duration::from_micros(500))
    );
    println!(
        "  pool.run + 1 barrier, parked (5 ms sleep):{:6.2} us",
        time(&mut || pool.run(&empty), 200, Duration::from_millis(5))
    );
    let mut moe = CpuMoe::new(pool);
    let few = &ew[..6];
    for t in [1usize, 6] {
        let x: Vec<f32> = (0..t * DIM).map(|_| rng.normal()).collect();
        let w0 = vec![0f32; 6 * t]; // nothing routed: quantize x, dispatch, barrier, return
        let mut out = vec![0f32; t * DIM];
        let us = time(
            &mut || moe.forward(few, DIM, &x, t, &w0, &mut out),
            5000,
            Duration::ZERO,
        );
        println!("  CpuMoe::forward with no routed rows, T={t}: {us:6.2} us (incl. quantizing x)");
    }
}
