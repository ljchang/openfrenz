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
pub const BUILT_IN: [(&str, &str); 5] = [
    ("eog", include_str!("../models/eog/config.json")),
    ("brow-emg", include_str!("../models/brow-emg/config.json")),
    ("heart", include_str!("../models/heart/config.json")),
    ("head", include_str!("../models/head/config.json")),
    ("eeg-power", include_str!("../models/eeg-power/config.json")),
];

/// The folder for `source`: itself if it is a folder; a model that comes with openfrenz if it
/// is one of their names ([`BUILT_IN`]); otherwise, with the `hub` feature, the Hugging Face
/// repository of that name (`owner/name` or `owner/name@revision`), fetched with [`fetch`].
pub fn resolve(source: &str) -> Result<PathBuf, Error> {
    resolve_source(source, None)
}

/// [`resolve`], reading Hugging Face repositories with `token` instead of the one in the
/// environment or the command-line login. For programs that keep their own credentials.
#[cfg(feature = "hub")]
pub fn resolve_with(source: &str, token: Option<&str>) -> Result<PathBuf, Error> {
    resolve_source(source, token)
}

fn resolve_source(source: &str, token: Option<&str>) -> Result<PathBuf, Error> {
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
    remote(source, token)
}

#[cfg(feature = "hub")]
fn remote(source: &str, token: Option<&str>) -> Result<PathBuf, Error> {
    let (repo, revision) = source.split_once('@').unwrap_or((source, "main"));
    fetch_with(repo, revision, token)
}

#[cfg(not(feature = "hub"))]
fn remote(source: &str, _token: Option<&str>) -> Result<PathBuf, Error> {
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
fn download(
    endpoint: &str,
    repo: &str,
    revision: &str,
    file: &str,
    token: Option<&str>,
) -> Result<Option<Download>, Fault> {
    use std::io::Read;

    let url = format!("{endpoint}/{repo}/resolve/{revision}/{file}");
    let mut request = ureq::get(&url);
    if let Some(token) = token {
        request = request.header("Authorization", format!("Bearer {token}"));
    }
    let mut response = match request.call() {
        Ok(response) => response,
        Err(ureq::Error::StatusCode(404)) => return Ok(None),
        Err(ureq::Error::StatusCode(401 | 403)) => return Err(Fault::Denied),
        Err(ureq::Error::StatusCode(code)) => {
            return Err(Fault::Other(Error::Fetch(format!(
                "{repo}/{file}: the server answered {code}"
            ))));
        }
        // No answer at all: no network, or the server could not be reached.
        Err(e) => return Err(Fault::Unreachable(format!("{repo}/{file}: {e}"))),
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
        .map_err(|e| Fault::Unreachable(format!("{repo}/{file}: {e}")))?;
    Ok(Some((bytes, commit)))
}

/// Why a download failed.
#[cfg(feature = "hub")]
enum Fault {
    /// The server refused: private, and no token or one without access.
    Denied,
    /// The server could not be reached.
    Unreachable(String),
    Other(Error),
}

/// The error for a repository that cannot be read, saying what to do about it.
#[cfg(feature = "hub")]
fn no_access(repo: &str, token: Option<&str>) -> Error {
    Error::Access(match token {
        None => format!(
            "{repo} was not found, or it is private: give a Hugging Face token from an account that has been invited to it"
        ),
        Some(_) => format!(
            "{repo} was not found, or the Hugging Face token given has no access to it: ask the repository's owners for an invitation"
        ),
    })
}

/// Fetches the model files of the Hugging Face repository `repo` (`owner/name`) at `revision`
/// (a branch, tag or commit) and returns the folder holding them.
///
/// Files are kept per commit under `~/.cache/openfrenz` (or `OPENFRENZ_CACHE`), so a commit is
/// downloaded once; a branch name is looked up each time to see which commit it points to, and
/// when the server cannot be reached the commit it last pointed to is used. Private repositories
/// need a token in `HF_TOKEN` or a Hugging Face command-line login; see [`fetch_with`] to give one.
#[cfg(feature = "hub")]
pub fn fetch(repo: &str, revision: &str) -> Result<PathBuf, Error> {
    fetch_with(repo, revision, token().as_deref())
}

/// [`fetch`] with `token` instead of the one in the environment or the command-line login.
#[cfg(feature = "hub")]
pub fn fetch_with(repo: &str, revision: &str, token: Option<&str>) -> Result<PathBuf, Error> {
    // `HF_ENDPOINT` points at a mirror, as for the Hugging Face tools.
    let endpoint = std::env::var("HF_ENDPOINT")
        .ok()
        .map(|e| e.trim_end_matches('/').to_owned())
        .filter(|e| !e.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_owned());
    fetch_from(&endpoint, &cache()?, repo, revision, token)
}

#[cfg(feature = "hub")]
fn fetch_from(endpoint: &str, cache: &Path, repo: &str, revision: &str, token: Option<&str>) -> Result<PathBuf, Error> {
    let parts: Vec<&str> = repo.split('/').collect();
    if parts.len() != 2 || !parts.iter().all(|p| safe(p)) || !safe(revision) {
        return Err(Error::Fetch(format!(
            "{repo}@{revision}: not a folder, and not a repository written owner/name"
        )));
    }
    let token = token.map(str::trim).filter(|t| !t.is_empty());
    let root = cache.join("models").join(parts.join("--"));
    let complete = |dir: &Path| dir.join("config.json").is_file() && dir.join(".complete").is_file();
    if complete(&root.join(revision)) {
        return Ok(root.join(revision)); // a commit already fetched
    }
    // The commit a branch or tag pointed to when it was last fetched.
    let remembered = root.join("refs").join(revision);
    let (config, commit) = match download(endpoint, repo, revision, "config.json", token) {
        Ok(Some(found)) => found,
        Ok(None) | Err(Fault::Denied) => return Err(no_access(repo, token)),
        Err(Fault::Unreachable(why)) => {
            let last = std::fs::read_to_string(&remembered).unwrap_or_default();
            let last = last.trim();
            if safe(last) && complete(&root.join(last)) {
                return Ok(root.join(last));
            }
            return Err(Error::Fetch(format!(
                "{why}; and {repo}@{revision} has not been fetched on this computer before"
            )));
        }
        Err(Fault::Other(error)) => return Err(error),
    };
    let commit = commit.unwrap_or_else(|| revision.to_owned());
    let dir = root.join(&commit);
    let remember = || -> Result<(), Error> {
        if commit != revision {
            std::fs::create_dir_all(root.join("refs"))?;
            std::fs::write(&remembered, &commit)?;
        }
        Ok(())
    };
    if complete(&dir) {
        remember()?;
        return Ok(dir);
    }
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("config.json"), config)?;
    for (file, required) in FILES.into_iter().skip(1) {
        match download(endpoint, repo, &commit, file, token) {
            Ok(Some((bytes, _))) => std::fs::write(dir.join(file), bytes)?,
            Ok(None) if required => return Err(Error::Fetch(format!("{repo}@{revision}: no {file}"))),
            Ok(None) => {}
            Err(Fault::Denied) => return Err(no_access(repo, token)),
            Err(Fault::Unreachable(why)) => return Err(Error::Fetch(why)),
            Err(Fault::Other(error)) => return Err(error),
        }
    }
    std::fs::write(dir.join(".complete"), "")?;
    remember()?;
    Ok(dir)
}

#[cfg(all(test, feature = "hub"))]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]
mod tests {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    use super::*;

    /// A stand-in for the Hugging Face server: answers every request with what `answer` gives for
    /// its path and authorization header, and records the requests.
    fn server(answer: fn(&str, Option<&str>) -> (u16, String)) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let path = line.split_whitespace().nth(1).unwrap_or_default().to_owned();
                let mut auth = None;
                loop {
                    let mut header = String::new();
                    if reader.read_line(&mut header).unwrap() <= 2 {
                        break;
                    }
                    if let Some(value) = header
                        .strip_prefix("authorization: ")
                        .or(header.strip_prefix("Authorization: "))
                    {
                        auth = Some(value.trim().to_owned());
                    }
                }
                log.lock()
                    .unwrap()
                    .push(format!("{path} {}", auth.as_deref().unwrap_or("-")));
                let (status, body) = answer(&path, auth.as_deref());
                let reply = format!(
                    "HTTP/1.1 {status} X\r\nx-repo-commit: abc123\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(reply.as_bytes()).unwrap();
            }
        });
        (address, seen)
    }

    const CONFIG: &str = r#"{"openfrenz_model": "pipeline"}"#;

    #[test]
    fn private_repositories_need_a_token_with_access() {
        let cache = tempfile::tempdir().unwrap();
        let (hub, seen) = server(|path, auth| match (path.ends_with("config.json"), auth) {
            (true, Some("Bearer good")) => (200, CONFIG.into()),
            (true, Some(_)) => (404, String::new()),
            (true, None) => (401, String::new()),
            _ => (404, String::new()),
        });
        let missing = fetch_from(&hub, cache.path(), "lab/model", "main", None).unwrap_err();
        assert!(
            matches!(&missing, Error::Access(m) if m.contains("give a Hugging Face token")),
            "{missing}"
        );
        let wrong = fetch_from(&hub, cache.path(), "lab/model", "main", Some("bad")).unwrap_err();
        assert!(
            matches!(&wrong, Error::Access(m) if m.contains("ask the repository's owners")),
            "{wrong}"
        );
        let dir = fetch_from(&hub, cache.path(), "lab/model", "main", Some(" good ")).unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("config.json")).unwrap(), CONFIG);
        assert!(dir.ends_with("abc123"));
        assert!(
            seen.lock().unwrap().iter().any(|r| r.ends_with("Bearer good")),
            "the token was trimmed and sent"
        );
    }

    #[test]
    fn a_branch_fetched_before_is_used_when_the_server_cannot_be_reached() {
        let cache = tempfile::tempdir().unwrap();
        let (hub, _) = server(|path, _| {
            if path.ends_with("config.json") {
                (200, CONFIG.into())
            } else {
                (404, String::new())
            }
        });
        let online = fetch_from(&hub, cache.path(), "lab/model", "main", None).unwrap();
        // Nothing listens on this port any more.
        let closed = {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            format!("http://{}", listener.local_addr().unwrap())
        };
        let offline = fetch_from(&closed, cache.path(), "lab/model", "main", None).unwrap();
        assert_eq!(online, offline);
        let never = fetch_from(&closed, cache.path(), "lab/other", "main", None).unwrap_err();
        assert!(
            matches!(&never, Error::Fetch(m) if m.contains("has not been fetched on this computer")),
            "{never}"
        );
    }
}
