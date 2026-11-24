//! Finding model files and the code that loads them.

use std::collections::BTreeMap;
use std::path::Path;
#[cfg(feature = "hub")]
use std::path::PathBuf;

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

    /// Loads a model from a folder, or from a Hugging Face repository (`owner/name`, optionally
    /// `owner/name@revision`) when `source` is not a folder.
    #[cfg(feature = "hub")]
    pub fn load(&self, source: &str) -> Result<Box<dyn Model>, Error> {
        self.load_dir(&resolve(source)?)
    }
}

/// The folder for `source`: itself if it is a folder, otherwise the Hugging Face repository of
/// that name (`owner/name` or `owner/name@revision`), fetched with [`fetch`].
#[cfg(feature = "hub")]
pub fn resolve(source: &str) -> Result<PathBuf, Error> {
    let path = Path::new(source);
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }
    let (repo, revision) = source.split_once('@').unwrap_or((source, "main"));
    fetch(repo, revision)
}

#[cfg(feature = "hub")]
const FILES: [(&str, bool); 2] = [("config.json", true), ("model.safetensors", false)];

/// Largest model file accepted.
#[cfg(feature = "hub")]
const MAX_BYTES: u64 = 2 << 30;

#[cfg(feature = "hub")]
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

#[cfg(feature = "hub")]
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
