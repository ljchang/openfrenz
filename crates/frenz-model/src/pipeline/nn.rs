//! Stored numbers and the small networks made from them.

// Weight matrices are indexed inside sizes checked when the layer is built.
#![allow(clippy::indexing_slicing)]

use std::collections::BTreeMap;
use std::path::Path;

use safetensors::SafeTensors;
use safetensors::tensor::Dtype;
use serde_json::Value as Json;

/// The arrays of a `model.safetensors` file, by name.
#[derive(Debug, Default)]
pub(crate) struct Weights {
    tensors: BTreeMap<String, (Vec<usize>, Vec<f64>)>,
}

impl Weights {
    pub(crate) fn read(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let file = SafeTensors::deserialize(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut tensors = BTreeMap::new();
        for name in file.names() {
            let tensor = file.tensor(name).map_err(|e| format!("{name}: {e}"))?;
            let data = tensor.data();
            let values: Vec<f64> = match tensor.dtype() {
                Dtype::F32 => data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f64::from(f32::from_le_bytes(*b)))
                    .collect(),
                Dtype::F64 => data.as_chunks::<8>().0.iter().map(|b| f64::from_le_bytes(*b)).collect(),
                _ => continue, // not a kind of number a pipeline uses
            };
            tensors.insert(name.to_owned(), (tensor.shape().to_vec(), values));
        }
        Ok(Self { tensors })
    }

    /// The array called `name` and its shape.
    pub(crate) fn get(&self, name: &str) -> Result<(&[usize], &[f64]), String> {
        self.tensors
            .get(name)
            .map(|(shape, values)| (shape.as_slice(), values.as_slice()))
            .ok_or_else(|| format!("model.safetensors has no array {name:?}"))
    }

    /// The array called `name`, which must be a matrix, as rows.
    pub(crate) fn matrix(&self, name: &str) -> Result<Vec<Vec<f64>>, String> {
        let (shape, values) = self.get(name)?;
        match shape {
            [_, columns] if *columns > 0 => Ok(values.chunks(*columns).map(<[f64]>::to_vec).collect()),
            _ => Err(format!("{name}: expected a matrix, found shape {shape:?}")),
        }
    }

    /// The array called `name`, which must be a list of numbers.
    pub(crate) fn vector(&self, name: &str) -> Result<Vec<f64>, String> {
        let (shape, values) = self.get(name)?;
        if shape.len() != 1 {
            return Err(format!("{name}: expected a list, found shape {shape:?}"));
        }
        Ok(values.to_vec())
    }
}

#[derive(Debug, Clone, Copy)]
enum Activation {
    Relu,
    Elu,
    Tanh,
    Sigmoid,
    Gelu,
}

impl Activation {
    fn named(name: &str) -> Option<Self> {
        Some(match name {
            "relu" => Self::Relu,
            "elu" => Self::Elu,
            "tanh" => Self::Tanh,
            "sigmoid" => Self::Sigmoid,
            "gelu" => Self::Gelu,
            _ => return None,
        })
    }

    fn apply(self, x: f64) -> f64 {
        match self {
            Self::Relu => x.max(0.0),
            Self::Elu => {
                if x > 0.0 {
                    x
                } else {
                    x.exp_m1()
                }
            }
            Self::Tanh => x.tanh(),
            Self::Sigmoid => sigmoid(x),
            // The tanh form; within 0.001 of the exact one.
            Self::Gelu => 0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh()),
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// A matrix with `rows` rows, stored row after row.
#[derive(Debug)]
struct Matrix {
    rows: usize,
    columns: usize,
    values: Vec<f64>,
}

impl Matrix {
    fn read(weights: &Weights, name: &str) -> Result<Self, String> {
        let (shape, values) = weights.get(name)?;
        match shape {
            [rows, columns] => Ok(Self {
                rows: *rows,
                columns: *columns,
                values: values.to_vec(),
            }),
            _ => Err(format!("{name}: expected a matrix, found shape {shape:?}")),
        }
    }

    /// `out[r] += row r of the matrix · x`.
    fn add_product(&self, x: &[f64], out: &mut [f64]) {
        for (r, sum) in out.iter_mut().enumerate().take(self.rows) {
            let row = &self.values[r * self.columns..(r + 1) * self.columns];
            *sum += row.iter().zip(x).map(|(w, v)| w * v).sum::<f64>();
        }
    }
}

#[derive(Debug)]
enum Layer {
    /// `weight · x + bias` on every row, with PyTorch's `Linear` layout.
    Dense {
        weight: Matrix,
        bias: Vec<f64>,
    },
    /// One LSTM layer read forwards over the rows, with PyTorch's layout (gates i, f, g, o).
    Lstm {
        input: Matrix,
        hidden: Matrix,
        bias: Vec<f64>,
    },
    Activation(Activation),
    /// Keeps the last row only.
    Last,
    /// Joins all rows into one.
    Flatten,
    /// The mean of the rows.
    Mean,
}

/// A feed-forward stack of layers over a sequence of rows.
#[derive(Debug)]
pub(crate) struct Network {
    layers: Vec<Layer>,
}

impl Network {
    /// Builds the layers a config lists. Each is `{"type": ...}`; `dense` and `lstm` name their
    /// arrays with `"weights": prefix`, and `dense` may carry an `"activation"`.
    pub(crate) fn build(layers: &[Json], weights: &Weights) -> Result<Self, String> {
        let mut built = Vec::new();
        for layer in layers {
            let kind = layer.get("type").and_then(Json::as_str).ok_or("a layer has no type")?;
            let prefix = || {
                layer
                    .get("weights")
                    .and_then(Json::as_str)
                    .ok_or_else(|| format!("the {kind} layer names no weights"))
            };
            match kind {
                "dense" => {
                    let prefix = prefix()?;
                    let weight = Matrix::read(weights, &format!("{prefix}.weight"))?;
                    let bias = weights.vector(&format!("{prefix}.bias"))?;
                    if bias.len() != weight.rows {
                        return Err(format!("{prefix}: the bias does not fit the weight"));
                    }
                    built.push(Layer::Dense { weight, bias });
                    if let Some(name) = layer.get("activation").and_then(Json::as_str) {
                        built.push(Layer::Activation(
                            Activation::named(name).ok_or_else(|| format!("unknown activation {name:?}"))?,
                        ));
                    }
                }
                "lstm" => {
                    let prefix = prefix()?;
                    let input = Matrix::read(weights, &format!("{prefix}.weight_ih_l0"))?;
                    let hidden = Matrix::read(weights, &format!("{prefix}.weight_hh_l0"))?;
                    let bias: Vec<f64> = weights
                        .vector(&format!("{prefix}.bias_ih_l0"))?
                        .iter()
                        .zip(weights.vector(&format!("{prefix}.bias_hh_l0"))?)
                        .map(|(a, b)| a + b)
                        .collect();
                    if input.rows != 4 * hidden.columns || hidden.rows != input.rows || bias.len() != input.rows {
                        return Err(format!("{prefix}: the arrays do not form an LSTM layer"));
                    }
                    built.push(Layer::Lstm { input, hidden, bias });
                }
                "last" => built.push(Layer::Last),
                "flatten" => built.push(Layer::Flatten),
                "mean" => built.push(Layer::Mean),
                other => built.push(Layer::Activation(
                    Activation::named(other).ok_or_else(|| format!("unknown layer type {other:?}"))?,
                )),
            }
        }
        Ok(Self { layers: built })
    }

    /// Runs the rows through every layer.
    pub(crate) fn forward(&self, mut rows: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, String> {
        for layer in &self.layers {
            let width = rows.first().map_or(0, Vec::len);
            rows = match layer {
                Layer::Dense { weight, bias } => {
                    if width != weight.columns {
                        return Err(format!(
                            "a dense layer takes {} values and was given {width}",
                            weight.columns
                        ));
                    }
                    rows.iter()
                        .map(|x| {
                            let mut out = bias.clone();
                            weight.add_product(x, &mut out);
                            out
                        })
                        .collect()
                }
                Layer::Lstm { input, hidden, bias } => {
                    if width != input.columns {
                        return Err(format!(
                            "an LSTM layer takes {} values and was given {width}",
                            input.columns
                        ));
                    }
                    let size = hidden.columns;
                    let (mut h, mut c) = (vec![0.0; size], vec![0.0; size]);
                    rows.iter()
                        .map(|x| {
                            let mut gates = bias.clone();
                            input.add_product(x, &mut gates);
                            hidden.add_product(&h, &mut gates);
                            for k in 0..size {
                                let (i, f) = (sigmoid(gates[k]), sigmoid(gates[size + k]));
                                let (g, o) = (gates[2 * size + k].tanh(), sigmoid(gates[3 * size + k]));
                                c[k] = f * c[k] + i * g;
                                h[k] = o * c[k].tanh();
                            }
                            h.clone()
                        })
                        .collect()
                }
                Layer::Activation(activation) => rows
                    .iter()
                    .map(|x| x.iter().map(|v| activation.apply(*v)).collect())
                    .collect(),
                Layer::Last => rows.pop().into_iter().collect(),
                Layer::Flatten => vec![rows.concat()],
                Layer::Mean => {
                    let count = rows.len().max(1) as f64;
                    let mut mean = vec![0.0; width];
                    for row in &rows {
                        for (m, v) in mean.iter_mut().zip(row) {
                            *m += v / count;
                        }
                    }
                    vec![mean]
                }
            };
        }
        Ok(rows)
    }
}
