//! Model outputs as tables of an exported session.

use std::collections::BTreeMap;
use std::path::Path;

use frenz_export::Session;
use frenz_export::table::{Column, Table};
use serde_json::{Value, json};

use crate::model::Model;
use crate::{Error, Pipeline, fingerprint_dir, resolve, run_session, streams};

/// A table of model outputs written by [`export_models`].
#[derive(Debug, Clone, PartialEq)]
pub struct Exported {
    /// Name of the table in `session.json`, which is the model's name.
    pub table: String,
    /// Steps that gave an output.
    pub rows: usize,
    /// Steps skipped because data was missing, counted by the stream that was missing.
    pub missed: BTreeMap<String, u64>,
}

fn bad(message: String) -> Error {
    Error::Files(message)
}

/// Runs each model in `sources` over `session` and writes what it gave beside the session's
/// other tables in `dir`, which must already hold the session ([`Session::write`]).
///
/// Each model becomes `<name>.parquet`, with `t` (the time of each step's last sample) and one
/// column per output, and an entry in `session.json` that says where the numbers came from: the
/// model's name, version and source, a fingerprint of its files, and the version of this crate.
/// Compare those with the model in hand to see whether a session needs to be exported again.
/// A source is a folder, the name of a model that comes with openfrenz, or a Hugging Face
/// repository; see [`resolve`].
pub fn export_models(session: &Session, dir: &Path, sources: &[String]) -> Result<Vec<Exported>, Error> {
    let manifest = dir.join("session.json");
    let mut meta: Value = serde_json::from_str(&std::fs::read_to_string(&manifest)?)?;
    let mut done = Vec::new();
    for source in sources {
        let folder = resolve(source)?;
        let model = Pipeline::from_dir(&folder)?;
        let spec = model.spec().clone();
        let name = spec.name.clone();
        let plain = !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'));
        if !plain {
            return Err(bad(format!(
                "{source}: the model's name {name:?} cannot name a table (letters, digits, - and _ only)"
            )));
        }
        if streams::index(&name).is_some() || session.tables.contains_key(name.as_str()) {
            return Err(bad(format!(
                "{source}: the model's name {name:?} is already a table of the session"
            )));
        }
        if done.iter().any(|d: &Exported| d.table == name) {
            return Err(bad(format!("two models are named {name:?}")));
        }
        let run = run_session(model, session)?;
        let mut table = Table::default().with("t", Column::F64(run.outputs.iter().map(|o| o.t).collect()));
        for (k, output) in spec.outputs.iter().enumerate() {
            let values = run.outputs.iter().map(|o| o.values.get(k).copied().unwrap_or(f64::NAN));
            table = table.with(output, Column::F64(values.collect()));
        }
        let file = format!("{name}.parquet");
        table.write_parquet(&dir.join(&file))?;
        let entry = json!({
            "file": file,
            "rows": table.rows(),
            "model": {
                "name": name,
                "version": spec.version,
                "source": source,
                "fingerprint": fingerprint_dir(&folder)?,
                "runner": format!("openfrenz frenz-model {}", env!("CARGO_PKG_VERSION")),
                "outputs": spec.outputs,
                "missed": run.missed,
            },
        });
        meta.get_mut("tables")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| bad(format!("{}: no tables", manifest.display())))?
            .insert(name.clone(), entry);
        done.push(Exported {
            table: name,
            rows: table.rows(),
            missed: run.missed,
        });
    }
    std::fs::write(&manifest, serde_json::to_string_pretty(&meta)?)?;
    Ok(done)
}
