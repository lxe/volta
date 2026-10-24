use anyhow::{Result, ensure};
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(
        a.len() >= 2,
        "usage: volta-quant <export|verify|export-full|verify-full|smoke> ..."
    );
    let result: Result<serde_json::Value> = (|| match a[1].as_str() {
        "export" | "verify" => {
            ensure!(
                a.len() == 6,
                "usage: volta-quant <export|verify> PACK_DIR GGUF DIM INTER"
            );
            let (pack, out) = (std::path::Path::new(&a[2]), std::path::Path::new(&a[3]));
            let (dim, inter) = (a[4].parse()?, a[5].parse()?);
            if a[1] == "export" {
                dsx_core::volta_quant::export(pack, out, dim, inter)?;
                Ok(serde_json::json!({"status":"exported"}))
            } else {
                Ok(serde_json::to_value(dsx_core::volta_quant::verify(
                    pack, out, dim, inter,
                )?)?)
            }
        }
        "export-full" | "verify-full" => {
            ensure!(
                a.len() == 7,
                "usage: volta-quant <export-full|verify-full> PACK_DIR BASE_DIR GGUF DIM INTER"
            );
            let (pack, base, out) = (
                std::path::Path::new(&a[2]),
                std::path::Path::new(&a[3]),
                std::path::Path::new(&a[4]),
            );
            let (dim, inter) = (a[5].parse()?, a[6].parse()?);
            if a[1] == "export-full" {
                dsx_core::volta_quant::complete::export(pack, base, out, dim, inter)?;
                Ok(serde_json::json!({"status":"exported-complete"}))
            } else {
                Ok(serde_json::to_value(
                    dsx_core::volta_quant::complete::verify(pack, base, out, dim, inter)?,
                )?)
            }
        }
        "smoke" => {
            ensure!(a.len() == 3, "usage: volta-quant smoke COMPLETE.gguf");
            dsx_core::volta_quant::complete::smoke(std::path::Path::new(&a[2]))
        }
        _ => anyhow::bail!("unknown volta-quant command"),
    })();
    match result {
        Ok(v) => {
            println!("{}", serde_json::to_string_pretty(&v)?);
            Ok(())
        }
        Err(e) => {
            println!(
                "{}",
                serde_json::json!({"status":"failed","error":format!("{e:#}")})
            );
            Err(e)
        }
    }
}
