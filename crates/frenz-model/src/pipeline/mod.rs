//! Models written as data: a `config.json` that wires standard blocks together.
//!
//! ```json
//! {
//!   "openfrenz_model": "pipeline",
//!   "name": "example", "version": "1",
//!   "step": { "stream": "eeg", "samples": 250 },
//!   "inputs": { "accel": { "samples": 100 } },
//!   "nodes": [
//!     { "name": "clean", "op": "filter", "in": "eeg",
//!       "steps": [{ "type": "bandpass", "low": 0.5, "high": 45 }] },
//!     { "name": "power", "op": "band_power", "in": "clean",
//!       "bands": [[4, 8], [8, 12]], "relative_to": [1, 45] },
//!     { "name": "move", "op": "summary", "in": "accel", "stats": ["sd"] },
//!     { "name": "all", "op": "concat", "in": ["power", "move"] }
//!   ],
//!   "outputs": ["all"]
//! }
//! ```
//!
//! Each node reads streams or earlier nodes (`in`) and has the settings of its block beside it.
//! A stream listed in `inputs` with `"required": false` is used when it is there: while it is
//! missing, the features made from it are not-a-number and the rest of the model carries on.
//! A stream without a steady rate is listed with `"arrived": true` and delivers whatever came
//! during the step.
//! Blocks that need stored numbers (`asr`, `normalize`, `network`) name them in `model.safetensors`
//! with `weights`. The blocks: `select`, `mix`, `filter`, `asr`, `band_power`, `wavelet`, `envelope`,
//! `summary`, `quality`, `coherence`, `coupling`, `phase_amplitude`, `heart`, `peaks`, `jumps`, `tilt`,
//! `ratios`, `average`, `concat`,
//! `normalize`, `history`, `network`. A pipeline
//! is checked when it is loaded by running it on silence, so a mismatch between blocks is
//! reported then, with the node's name.

mod nn;
mod ops;

use std::path::Path;

use serde_json::Value as Json;

use self::nn::Weights;
use self::ops::{Op, Signal, Value};
use crate::model::{Block, Input, Model, ModelError, Spec, Step, Take};
use crate::{Error, streams};

enum Source {
    Stream(String),
    Node(usize),
}

struct Node {
    name: String,
    sources: Vec<Source>,
    op: Box<dyn Op>,
    /// Reads, directly or through other nodes, a stream the model can do without.
    optional: bool,
    /// Names of the features it gives, once it has given some.
    names: Option<Vec<String>>,
    /// Whether it has given anything yet.
    ran: bool,
}

/// What a node made of one step.
enum Made {
    Value(Value),
    /// Not enough steps yet; nothing that depends on it can run.
    Warming,
    /// A stream it could do without was not there.
    Absent,
}

/// A model made of standard blocks, as its `config.json` describes.
pub struct Pipeline {
    spec: Spec,
    nodes: Vec<Node>,
    outputs: Vec<usize>,
}

/// Most samples of one stream in a step, and most nodes in a pipeline. Configs can come from
/// anywhere, so their sizes are bounded before anything is allocated.
const MAX_SAMPLES: u64 = 100_000;
const MAX_NODES: usize = 1024;

/// Names and values of a step's outputs.
type Named = (Vec<String>, Vec<f64>);

/// Steps of silence a pipeline is given to produce its first output when it is checked.
const CHECK_STEPS: usize = 512;

fn bad(message: impl Into<String>) -> Error {
    Error::Files(format!("config.json: {}", message.into()))
}

impl Pipeline {
    /// The name model files use to ask for a pipeline (`openfrenz_model` in `config.json`).
    pub const KIND: &'static str = "pipeline";

    /// Loads the pipeline in the folder `dir`: `config.json`, and `model.safetensors` if any
    /// block needs stored numbers.
    pub fn from_dir(dir: &Path) -> Result<Self, Error> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| Error::Files(format!("{}: {e}", dir.join("config.json").display())))?;
        let config: Json = serde_json::from_str(&text)?;
        let stored = dir.join("model.safetensors");
        let weights = if stored.is_file() {
            Weights::read(&stored).map_err(Error::Files)?
        } else {
            Weights::default()
        };
        Self::build(&config, &weights)
    }

    fn build(config: &Json, weights: &Weights) -> Result<Self, Error> {
        let text = |key: &str| config.get(key).and_then(Json::as_str).unwrap_or_default().to_owned();
        let samples = |entry: &Json, what: &str| {
            entry
                .get("samples")
                .and_then(Json::as_u64)
                .filter(|n| (1..=MAX_SAMPLES).contains(n))
                .map(|n| n as usize)
                .ok_or_else(|| {
                    bad(format!(
                        "{what} needs samples: a whole number between 1 and {MAX_SAMPLES}"
                    ))
                })
        };
        let step = config.get("step").ok_or_else(|| bad("no step"))?;
        let clock = step
            .get("stream")
            .and_then(Json::as_str)
            .ok_or_else(|| bad("step needs a stream"))?;
        let mut inputs = vec![Input::samples(clock, samples(step, "step")?)];
        if let Some(others) = config.get("inputs").and_then(Json::as_object) {
            for (stream, entry) in others {
                if stream == clock {
                    continue;
                }
                let flag = |key: &str, default: bool| entry.get(key).and_then(Json::as_bool).unwrap_or(default);
                inputs.push(if flag("arrived", false) {
                    Input::arrived(stream)
                } else if flag("required", true) {
                    Input::samples(stream, samples(entry, stream)?)
                } else {
                    Input::samples(stream, samples(entry, stream)?).preferred()
                });
            }
        }
        let mut length = 0.0;
        for input in &inputs {
            if streams::index(&input.stream).is_none() {
                return Err(bad(format!("{} is not a stream the band sends", input.stream)));
            }
            if let Take::Samples(n) = input.take {
                let rate = streams::rate_hz(&input.stream).ok_or_else(|| {
                    bad(format!(
                        "{} has no steady rate: give it \"arrived\": true, not samples",
                        input.stream
                    ))
                })?;
                if length == 0.0 {
                    length = n as f64 / rate;
                }
            }
        }

        let mut nodes: Vec<Node> = Vec::new();
        let find = |nodes: &[Node], name: &str| -> Result<Source, Error> {
            if let Some(index) = nodes.iter().position(|n| n.name == name) {
                Ok(Source::Node(index))
            } else if inputs.iter().any(|i| i.stream == name) {
                Ok(Source::Stream(name.to_owned()))
            } else {
                Err(bad(format!("{name:?} is neither an input stream nor an earlier node")))
            }
        };
        let listed = config
            .get("nodes")
            .and_then(Json::as_array)
            .ok_or_else(|| bad("no nodes"))?;
        if listed.len() > MAX_NODES {
            return Err(bad(format!("more than {MAX_NODES} nodes")));
        }
        for entry in listed {
            let name = entry
                .get("name")
                .and_then(Json::as_str)
                .ok_or_else(|| bad("a node has no name"))?;
            let op = entry
                .get("op")
                .and_then(Json::as_str)
                .ok_or_else(|| bad(format!("node {name}: no op")))?;
            if nodes.iter().any(|n| n.name == name) || inputs.iter().any(|i| i.stream == name) {
                return Err(bad(format!("the name {name:?} is used twice")));
            }
            let reads: Vec<&str> = match entry.get("in") {
                Some(Json::String(one)) => vec![one.as_str()],
                Some(Json::Array(many)) => many.iter().filter_map(Json::as_str).collect(),
                _ => return Err(bad(format!("node {name}: in must be a name or a list of names"))),
            };
            let sources: Vec<Source> = reads.iter().map(|r| find(&nodes, r)).collect::<Result<_, _>>()?;
            let optional = sources.iter().any(|source| match source {
                Source::Stream(stream) => inputs.iter().any(|i| &i.stream == stream && !i.required),
                Source::Node(index) => nodes.get(*index).is_some_and(|n| n.optional),
            });
            let op = ops::build(op, entry, weights).map_err(|e| bad(format!("node {name}: {e}")))?;
            nodes.push(Node {
                name: name.to_owned(),
                sources,
                op,
                optional,
                names: None,
                ran: false,
            });
        }
        let mut outputs = Vec::new();
        for name in config
            .get("outputs")
            .and_then(Json::as_array)
            .ok_or_else(|| bad("no outputs"))?
        {
            match name.as_str().map(|n| find(&nodes, n)) {
                Some(Ok(Source::Node(index))) => outputs.push(index),
                _ => return Err(bad(format!("output {name} is not a node"))),
            }
        }

        let mut pipeline = Self {
            spec: Spec {
                name: text("name"),
                version: text("version"),
                inputs,
                outputs: Vec::new(),
                max_step_seconds: config
                    .get("max_step_seconds")
                    .and_then(Json::as_f64)
                    .filter(|s| s.is_finite() && *s > 0.0)
                    .unwrap_or(1.1 * length),
            },
            nodes,
            outputs,
        };
        pipeline.spec.outputs = pipeline.check()?;
        Ok(pipeline)
    }

    /// Runs the pipeline on silence until every node has given something, and returns the
    /// names of the outputs.
    fn check(&mut self) -> Result<Vec<String>, Error> {
        let blocks = self.spec.inputs.iter().filter_map(|input| {
            let channels = streams::channels(&input.stream)?.len();
            // A stream taken as it arrives is given two readings.
            let n = if let Take::Samples(n) = input.take { n } else { 2 };
            Some((
                input.stream.clone(),
                Block {
                    t: vec![0.0; n],
                    values: vec![vec![0.0; n]; channels],
                },
            ))
        });
        let silence = Step::new(0.0, 0.0, blocks);
        let mut names = None;
        for _ in 0..CHECK_STEPS {
            let found = self.run(&silence).map_err(bad)?;
            // Every node must show what it gives, so that later a missing stream can be stood in for.
            if let (Some((found, _)), true) = (found, self.nodes.iter().all(|n| n.ran)) {
                names = Some(found);
                break;
            }
        }
        self.reset();
        names.ok_or_else(|| bad(format!("no output after {CHECK_STEPS} steps")))
    }

    /// One step: the names and values of the outputs, or `None` while a block is still warming up.
    fn run(&mut self, step: &Step) -> Result<Option<Named>, String> {
        let mut results: Vec<Made> = Vec::with_capacity(self.nodes.len());
        for node in &mut self.nodes {
            // What the node reads: its streams as signals, and what earlier nodes made.
            let mut streamed = Vec::new();
            let (mut absent, mut warming) = (false, false);
            for source in &node.sources {
                match source {
                    Source::Stream(stream) => match step.get(stream) {
                        Some(block) => streamed.push(Value::Signal(Signal {
                            rate_hz: streams::rate_hz(stream).unwrap_or(0.0),
                            channels: streams::channels(stream)
                                .unwrap_or_default()
                                .iter()
                                .map(|c| (*c).to_owned())
                                .collect(),
                            data: block.values.clone(),
                        })),
                        None if node.optional => absent = true,
                        None => return Err(format!("the step has no {stream}")),
                    },
                    Source::Node(index) => match results.get(*index) {
                        Some(Made::Value(_)) => {}
                        Some(Made::Absent) => absent = true,
                        _ => warming = true,
                    },
                }
            }
            let made = if absent {
                // Whatever it had gathered no longer follows on from what comes next.
                node.op.reset();
                None
            } else if warming {
                results.push(Made::Warming);
                continue;
            } else {
                let mut streamed = streamed.iter();
                let inputs: Vec<&Value> = node
                    .sources
                    .iter()
                    .filter_map(|source| match source {
                        Source::Stream(_) => streamed.next(),
                        Source::Node(index) => match results.get(*index) {
                            Some(Made::Value(value)) => Some(value),
                            _ => None,
                        },
                    })
                    .collect();
                node.op.run(&inputs).map_err(|e| format!("node {}: {e}", node.name))?
            };
            results.push(match made {
                Some(value) => {
                    node.ran = true;
                    if let (Value::Features { names, .. }, None) = (&value, &node.names) {
                        node.names = Some(names.clone());
                    }
                    Made::Value(value)
                }
                // A node that can be done without never holds the model up: while its stream is
                // missing, and while it gathers steps again afterwards, its features are
                // not-a-number.
                None if node.optional => match &node.names {
                    Some(names) => Made::Value(Value::Features {
                        names: names.clone(),
                        values: vec![f64::NAN; names.len()],
                    }),
                    None if absent => Made::Absent,
                    None => Made::Warming,
                },
                None => Made::Warming,
            });
        }
        let (mut names, mut values) = (Vec::new(), Vec::new());
        for index in &self.outputs {
            match results.get(*index) {
                Some(Made::Value(Value::Features { names: n, values: v })) => {
                    names.extend(n.iter().cloned());
                    values.extend(v);
                }
                Some(Made::Value(_)) => {
                    let name = self.nodes.get(*index).map(|n| n.name.as_str()).unwrap_or_default();
                    return Err(format!("output {name} is not features"));
                }
                _ => return Ok(None),
            }
        }
        Ok(Some((names, values)))
    }
}

impl Model for Pipeline {
    fn spec(&self) -> &Spec {
        &self.spec
    }

    fn reset(&mut self) {
        self.nodes.iter_mut().for_each(|node| node.op.reset());
    }

    fn step(&mut self, step: &Step) -> Result<Option<Vec<f64>>, ModelError> {
        Ok(self.run(step)?.map(|(_, values)| values))
    }
}
