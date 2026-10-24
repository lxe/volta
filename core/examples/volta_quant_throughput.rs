//! CPU-only screening benchmark; no model files or GPU calls.
use dsx_core::volta_quant::{Matrix, transform};
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let rows = a.get(1).map(|x| x.parse()).transpose()?.unwrap_or(576);
    let cols = a.get(2).map(|x| x.parse()).transpose()?.unwrap_or(5120);
    let repeats = a.get(3).map(|x| x.parse()).transpose()?.unwrap_or(100);
    for bits in [2, 3, 4] {
        let m = Matrix {
            layer: 0,
            expert: 0,
            part: 0,
            role: "gu".into(),
            rows,
            cols,
            bits,
            mul_bits: 1,
        };
        let (nw, ns) = m.sizes()?;
        let codes: Vec<u8> = (0..nw).map(|i| (i.wrapping_mul(37)) as u8).collect();
        let scales: Vec<u8> = (0..ns).map(|i| i as u8).collect();
        let native = transform(&m, &codes, &scales, true)?;
        assert_eq!(
            transform(&m, &native.0, &native.1, false)?,
            (codes.clone(), scales.clone())
        );
        for to_native in [true, false] {
            let (w, s) = if to_native {
                (&codes, &scales)
            } else {
                (&native.0, &native.1)
            };
            let start = std::time::Instant::now();
            for _ in 0..repeats {
                std::hint::black_box(transform(&m, w, s, to_native)?);
            }
            let seconds = start.elapsed().as_secs_f64();
            println!(
                "bits={bits} rows={rows} cols={cols} direction={} repeats={repeats} bytes_per_matrix={} seconds={seconds:.6} payload_MBps={:.3}",
                if to_native { "load" } else { "export" },
                nw + ns,
                (nw + ns) as f64 * repeats as f64 / seconds / 1e6
            );
        }
    }
    Ok(())
}
