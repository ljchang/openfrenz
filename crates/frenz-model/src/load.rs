//! Finding model files and the code that loads them.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::Error;
use crate::model::Model;

/// Key in `config.json` that names the kind of model the folder holds.
pub const CONFIG_KEY: &str = "openfrenz_model";

/// The kind of model the folder `dir` holds, from its `config.json`.
pub fn kind(dir: &Path) -> Result<String, Error> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| Error::Files(format!("{}: {e}", path.display())))?;
    let config: serde_json::Value = serde_json::from_str(&text)?;
    config
        .get(CONFIG_KEY)
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| Error::Files(format!("{}: no {CONFIG_KEY} entry", path.display())))
}

type Loader = Box<dyn Fn(&Path) -> Result<Box<dyn Model>, Error> + Send + Sync>;

/// The kinds of model a program can load.
///
/// A program lists the model code it was built with; [`Registry::load`] then picks the loader the
/// model files ask for.
#[derive(Default)]
pub struct Registry {
    loaders: BTreeMap<String, Loader>,
}

impl Registry {
    /// A registry without any model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds the loader for models of `kind`: a function from the model folder to the model.
    #[must_use]
    pub fn with<F>(mut self, kind: &str, loader: F) -> Self
    where
        F: Fn(&Path) -> Result<Box<dyn Model>, Error> + Send + Sync + 'static,
    {
        self.loaders.insert(kind.to_owned(), Box::new(loader));
        self
    }

    /// Adds [`crate::Pipeline`]: models described entirely by their files.
    #[must_use]
    pub fn with_pipelines(self) -> Self {
        self.with(crate::Pipeline::KIND, |dir| {
            Ok(Box::new(crate::Pipeline::from_dir(dir)?) as Box<dyn Model>)
        })
    }

    /// The kinds that can be loaded.
    pub fn kinds(&self) -> impl Iterator<Item = &str> {
        self.loaders.keys().map(String::as_str)
    }

    /// Loads the model in the folder `dir`.
    pub fn load_dir(&self, dir: &Path) -> Result<Box<dyn Model>, Error> {
        let kind = kind(dir)?;
        match self.loaders.get(&kind) {
            Some(loader) => loader(dir),
            None => {
                let known: Vec<_> = self.kinds().collect();
                Err(Error::UnknownKind {
                    kind,
                    known: if known.is_empty() {
                        "none".into()
                    } else {
                        known.join(", ")
                    },
                })
            }
        }
    }

    /// Loads a model from a folder, by the name of one that comes with openfrenz, or (with the
    /// `hub` feature) from a Hugging Face repository; see [`resolve`].
    pub fn load(&self, source: &str) -> Result<Box<dyn Model>, Error> {
        self.load_dir(&resolve(source)?)
    }
}

/// Models that come with openfrenz: their names and `config.json` files. None needs stored numbers.
pub const BUILT_IN: [(&str, &str); 4] = [
    ("eog", include_str!("../models/eog/config.json")),
    ("brow-emg", include_str!("../models/brow-emg/config.json")),
    ("heart", include_str!("../models/heart/config.json")),
    ("head", include_str!("../models/head/config.json")),
];

/// The folder for `source`: itself if it is a folder; a model that comes with openfrenz if it
/// is one of their names ([`BUILT_IN`]); otherwise, with the `hub` feature, the Hugging Face
/// repository of that name (`owner/name` or `owner/name@revision`), fetched with [`fetch`].
pub fn resolve(source: &str) -> Result<PathBuf, Error> {
    let path = Path::new(source);
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }
    if let Some((name, config)) = BUILT_IN.iter().find(|(name, _)| *name == source) {
        // Written where models are kept, under what the file holds, so that a new version of
        // openfrenz never reads an older one's copy.
        let dir = cache()?
            .join("built-in")
            .join(name)
            .join(fingerprint(&[config.as_bytes(), &[]]));
        let file = dir.join("config.json");
        if std::fs::read_to_string(&file).ok().as_deref() != Some(*config) {
            std::fs::create_dir_all(&dir)?;
            std::fs::write(&file, config)?;
        }
        return Ok(dir);
    }
    remote(source)
}

#[cfg(feature = "hub")]
fn remote(source: &str) -> Result<PathBuf, Error> {
    let (repo, revision) = source.split_once('@').unwrap_or((source, "main"));
    fetch(repo, revision)
}

#[cfg(not(feature = "hub"))]
fn remote(source: &str) -> Result<PathBuf, Error> {
    let known: Vec<&str> = BUILT_IN.iter().map(|(name, _)| *name).collect();
    Err(Error::Files(format!(
        "{source}: not a folder, and not a model that comes with openfrenz ({})",
        known.join(", ")
    )))
}

/// A short name for the contents of some files: the same bytes always give the same name, and
/// any change gives another. It tells whether two results came from the same model; it is not
/// protection against tampering.
pub fn fingerprint(parts: &[&[u8]]) -> String {
    // FNV-1a, 64 bits, with each part's length mixed in so that moving bytes between parts shows.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |byte: u8| {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for part in parts {
        (part.len() as u64).to_le_bytes().into_iter().for_each(&mut mix);
        part.iter().copied().for_each(&mut mix);
    }
    format!("{hash:016x}")
}

/// The fingerprint of the model files in `dir`: `config.json` and, if there, `model.safetensors`.
pub fn fingerprint_dir(dir: &Path) -> Result<String, Error> {
    let config = std::fs::read(dir.join("config.json"))
        .map_err(|e| Error::Files(format!("{}: {e}", dir.join("config.json").display())))?;
    let stored = std::fs::read(dir.join("model.safetensors")).unwrap_or_default();
    Ok(fingerprint(&[&config, &stored]))
}

#[cfg(feature = "hub")]
const FILES: [(&str, bool); 2] = [("config.json", true), ("model.safetensors", false)];

/// Largest model file accepted.
#[cfg(feature = "hub")]
const MAX_BYTES: u64 = 512 << 20;

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// The access token for private repositories: `HF_TOKEN`, or the file the Hugging Face
/// command-line login writes.
#[cfg(feature = "hub")]
fn token() -> Option<String> {
    let clean = |s: String| Some(s.trim().to_owned()).filter(|s| !s.is_empty());
    if let Some(token) = std::env::var("HF_TOKEN").ok().and_then(clean) {
        return Some(token);
    }
    let file = match std::env::var_os("HF_TOKEN_PATH") {
        Some(path) => PathBuf::from(path),
        None => match std::env::var_os("HF_HOME") {
            Some(dir) => PathBuf::from(dir).join("token"),
            None => home()?.join(".cache/huggingface/token"),
        },
    };
    std::fs::read_to_string(file).ok().and_then(clean)
}

fn cache() -> Result<PathBuf, Error> {
    if let Some(dir) = std::env::var_os("OPENFRENZ_CACHE") {
        return Ok(PathBuf::from(dir));
    }
    home()
        .map(|h| h.join(".cache/openfrenz"))
        .ok_or_else(|| Error::Fetch("no home folder for the model cache; set OPENFRENZ_CACHE".into()))
}

#[cfg(feature = "hub")]
fn safe(part: &str) -> bool {
    !part.is_empty()
        && part != "."
        && part != ".."
        && part
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// A downloaded file and the commit its revision stands for, when the server says.
#[cfg(feature = "hub")]
type Download = (Vec<u8>, Option<String>);

/// Downloads `file` of `repo` at `revision`. `Ok(None)` if the repository has no such file.
#[cfg(feature = "hub")]
fn download(repo: &str, revision: &str, file: &str) -> Result<Option<Download>, Error> {
    use std::io::Read;

    let url = format!("https://huggingface.co/{repo}/resolve/{revision}/{file}");
    let mut request = ureq::get(&url);
    if let Some(token) = token() {
        request = request.header("Authorization", format!("Bearer {token}"));
    }
    let mut response = match request.call() {
        Ok(response) => response,
        Err(ureq::Error::StatusCode(404)) => return Ok(None),
        Err(ureq::Error::StatusCode(401 | 403)) => {
            return Err(Error::Fetch(format!(
                "{repo}: access denied; for a private repository set HF_TOKEN or log in with the Hugging Face command-line tool"
            )));
        }
        Err(e) => return Err(Error::Fetch(format!("{repo}/{file}: {e}"))),
    };
    let commit = response
        .headers()
        .get("x-repo-commit")
        .and_then(|v| v.to_str().ok())
        .filter(|c| safe(c))
        .map(str::to_owned);
    let mut bytes = Vec::new();
    response
        .body_mut()
        .as_reader()
        .take(MAX_BYTES)
        .read_to_end(&mut bytes)
        .map_err(|e| Error::Fetch(format!("{repo}/{file}: {e}")))?;
    Ok(Some((bytes, commit)))
}

/// Fetches the model files of the Hugging Face repository `repo` (`owner/name`) at `revision`
/// (a branch, tag or commit) and returns the folder holding them.
///
/// Files are kept per commit under `~/.cache/openfrenz` (or `OPENFRENZ_CACHE`), so a commit is
/// downloaded once; a branch name is looked up each time to see which commit it points to.
/// Private repositories need a token in `HF_TOKEN` or a Hugging Face command-line login.
#[cfg(feature = "hub")]
pub fn fetch(repo: &str, revision: &str) -> Result<PathBuf, Error> {
    let parts: Vec<&str> = repo.split('/').collect();
    if parts.len() != 2 || !parts.iter().all(|p| safe(p)) || !safe(revision) {
        return Err(Error::Fetch(format!(
            "{repo}@{revision}: not a folder, and not a repository written owner/name"
        )));
    }
    let root = cache()?.join("models").join(parts.join("--"));
    let complete = |dir: &Path| dir.join("config.json").is_file() && dir.join(".complete").is_file();
    if complete(&root.join(revision)) {
        return Ok(root.join(revision)); // a commit already fetched
    }
    let (config, commit) = download(repo, revision, "config.json")?
        .ok_or_else(|| Error::Fetch(format!("{repo}@{revision}: no config.json")))?;
    let commit = commit.unwrap_or_else(|| revision.to_owned());
    let dir = root.join(&commit);
    if complete(&dir) {
        return Ok(dir);
    }
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("config.json"), config)?;
    for (file, required) in FILES.into_iter().skip(1) {
        match download(repo, &commit, file)? {
            Some((bytes, _)) => std::fs::write(dir.join(file), bytes)?,
            None if required => return Err(Error::Fetch(format!("{repo}@{revision}: no {file}"))),
            None => {}
        }
    }
    std::fs::write(dir.join(".complete"), "")?;
    Ok(dir)
}
