// A small network that scores a position from its signals (see features.rs), as an alternative to
// the hand-written evaluation. One hidden layer, and several outputs so that the margin of a win
// teaches it as well as the win itself, after the way backgammon programs are built.
// Each output may also take the hand-written table's score directly, through a weight of its own,
// so that the network learns a correction to the table rather than the whole game afresh.
use super::features::{features, NUM_FEATURES};
use crate::state::{SideReference, State};
use std::sync::OnceLock;

/// What the network predicts, for the side whose view is taken: that it wins, that it wins by at
/// least two pokemon and by at least four, and that it loses by at least two and by at least four.
pub const NUM_OUTPUTS: usize = 5;

#[derive(Debug, Clone)]
pub struct EvalModel {
    pub inputs: usize,
    pub hidden: usize,
    /// hidden rows of `inputs` weights each
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    /// NUM_OUTPUTS rows of `hidden` weights each
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    /// one weight per output on the table's score times TABLE_SCALE; zeros for none
    pub skip: Vec<f32>,
}

/// The table's score is multiplied by this before it reaches the network: the engine's own scale
/// for turning a score into a chance, fixed here so that a model does not depend on settings.
pub const TABLE_SCALE: f32 = 0.0125;

impl EvalModel {
    /// Reads "inputs hidden outputs" followed by w1, b1, w2, b2 and optionally the skip weights,
    /// all separated by white space.
    pub fn from_text(text: &str) -> Result<EvalModel, String> {
        let mut numbers = text.split_whitespace();
        let mut size = || -> Result<usize, String> {
            numbers
                .next()
                .ok_or("model file is too short".to_string())?
                .parse::<usize>()
                .map_err(|_| "model file does not start with its sizes".to_string())
        };
        let (inputs, hidden, outputs) = (size()?, size()?, size()?);
        if inputs != NUM_FEATURES || outputs != NUM_OUTPUTS {
            return Err(format!(
                "model is for {} signals and {} outputs, but this engine has {} and {}",
                inputs, outputs, NUM_FEATURES, NUM_OUTPUTS
            ));
        }
        let values: Result<Vec<f32>, _> = numbers.map(|number| number.parse::<f32>()).collect();
        let values =
            values.map_err(|_| "model file has something that is not a number".to_string())?;
        let expected = hidden * inputs + hidden + outputs * hidden + outputs;
        if values.len() != expected && values.len() != expected + outputs {
            return Err(format!(
                "model file has {} numbers, expected {}",
                values.len(),
                expected
            ));
        }
        let (w1, rest) = values.split_at(hidden * inputs);
        let (b1, rest) = rest.split_at(hidden);
        let (w2, rest) = rest.split_at(outputs * hidden);
        let (b2, skip) = rest.split_at(outputs);
        let skip = if skip.is_empty() {
            vec![0.0; outputs]
        } else {
            skip.to_vec()
        };
        Ok(EvalModel {
            inputs,
            hidden,
            w1: w1.to_vec(),
            b1: b1.to_vec(),
            w2: w2.to_vec(),
            b2: b2.to_vec(),
            skip,
        })
    }

    pub fn to_text(&self) -> String {
        let mut text = format!("{} {} {}\n", self.inputs, self.hidden, NUM_OUTPUTS);
        for values in [&self.w1, &self.b1, &self.w2, &self.b2, &self.skip] {
            let line: Vec<String> = values.iter().map(|value| format!("{}", value)).collect();
            text.push_str(&line.join(" "));
            text.push('\n');
        }
        text
    }

    /// The hidden layer's activations, which training also needs.
    pub fn hidden_layer(&self, signals: &[f32], hidden: &mut [f32]) {
        for (unit, activation) in hidden.iter_mut().enumerate() {
            let weights = &self.w1[unit * self.inputs..(unit + 1) * self.inputs];
            let mut sum = self.b1[unit];
            for (weight, signal) in weights.iter().zip(signals.iter()) {
                sum += weight * signal;
            }
            *activation = sum.max(0.0);
        }
    }

    /// `table` is the table's score from the same point of view as the signals, times TABLE_SCALE.
    pub fn outputs(&self, signals: &[f32], table: f32) -> [f32; NUM_OUTPUTS] {
        let mut hidden = vec![0.0; self.hidden];
        self.hidden_layer(signals, &mut hidden);
        let mut out = [0.0; NUM_OUTPUTS];
        for (index, value) in out.iter_mut().enumerate() {
            let weights = &self.w2[index * self.hidden..(index + 1) * self.hidden];
            let mut sum = self.b2[index] + self.skip[index] * table;
            for (weight, activation) in weights.iter().zip(hidden.iter()) {
                sum += weight * activation;
            }
            *value = 1.0 / (1.0 + (-sum).exp());
        }
        out
    }

    /// The chance that side one wins from this position, given the table's score for it.
    pub fn win_chance(&self, state: &State, table: f32) -> f32 {
        let mut signals = [0.0; NUM_FEATURES];
        features(state, &SideReference::SideOne, &mut signals);
        self.outputs(&signals, table * TABLE_SCALE)[0]
    }
}

/// How far a game has gone, by the fewest pokemon either side has left: 0 while both have five or
/// six, 1 when one has three or four, 2 when one has one or two. A model may have a network for each.
pub const NUM_STAGES: usize = 3;

pub fn stage_of_left(left: usize) -> usize {
    if left >= 5 {
        0
    } else if left >= 3 {
        1
    } else {
        2
    }
}

pub fn stage_of(state: &State) -> usize {
    let alive = |side: &crate::state::Side| side.pokemon.pkmn.iter().filter(|p| p.hp > 0).count();
    stage_of_left(alive(&state.side_one).min(alive(&state.side_two)))
}

/// Reads a model file: one network, or `stages 3` followed by one network for each stage.
pub fn models_from_text(text: &str) -> Result<Vec<EvalModel>, String> {
    let text = text.trim_start();
    let Some(rest) = text.strip_prefix("stages") else {
        return Ok(vec![EvalModel::from_text(text)?]);
    };
    let blocks: Vec<&str> = rest.split("\nnetwork\n").skip(1).collect();
    if blocks.len() != NUM_STAGES {
        return Err(format!(
            "model file has {} networks, expected one for each of {} stages",
            blocks.len(),
            NUM_STAGES
        ));
    }
    blocks.into_iter().map(EvalModel::from_text).collect()
}

/// Writes networks in the form models_from_text reads.
pub fn models_to_text(models: &[EvalModel]) -> String {
    if models.len() == 1 {
        return models[0].to_text();
    }
    let mut text = format!("stages {}\n", models.len());
    for model in models {
        text.push_str("network\n");
        text.push_str(&model.to_text());
    }
    text
}

/// A network the search uses, with the first layer's weights also laid out by signal, so that a
/// position's score is built up one signal at a time across all hidden units at once. That is
/// several times faster than one long sum per unit, and signals that are zero cost nothing.
struct Loaded {
    model: EvalModel,
    /// `inputs` rows of `hidden` weights each
    by_input: Vec<f32>,
}

/// The largest hidden layer the fast path handles; a larger model takes the plain path.
const FAST_HIDDEN: usize = 256;

static EVAL_MODELS: OnceLock<Vec<Loaded>> = OnceLock::new();

/// Makes the search score positions with these networks instead of the hand-written evaluation:
/// one for every stage, or a single one for all. Only the first call has any effect, and it must
/// come before the first evaluation.
pub fn set_eval_models(models: Vec<EvalModel>) -> bool {
    let loaded = models
        .into_iter()
        .map(|model| {
            let mut by_input = vec![0.0; model.inputs * model.hidden];
            for unit in 0..model.hidden {
                for input in 0..model.inputs {
                    by_input[input * model.hidden + unit] = model.w1[unit * model.inputs + input];
                }
            }
            Loaded { model, by_input }
        })
        .collect();
    EVAL_MODELS.set(loaded).is_ok()
}

pub fn set_eval_model(model: EvalModel) -> bool {
    set_eval_models(vec![model])
}

pub fn eval_model() -> Option<&'static EvalModel> {
    EVAL_MODELS
        .get()
        .and_then(|models| models.first())
        .map(|loaded| &loaded.model)
}

/// The chance that side one wins from this position by the model given at startup, if there is one,
/// given the table's score for it.
pub fn learned_win_chance(state: &State, table: f32) -> Option<f32> {
    let models = EVAL_MODELS.get()?;
    let loaded = if models.len() == NUM_STAGES {
        &models[stage_of(state)]
    } else {
        models.first()?
    };
    let model = &loaded.model;
    let hidden = model.hidden;
    if hidden > FAST_HIDDEN {
        return Some(model.win_chance(state, table));
    }
    let mut signals = [0.0; NUM_FEATURES];
    features(state, &SideReference::SideOne, &mut signals);
    let mut sums = [0.0f32; FAST_HIDDEN];
    let sums = &mut sums[..hidden];
    sums.copy_from_slice(&model.b1);
    for (input, signal) in signals.iter().enumerate() {
        if *signal == 0.0 {
            continue;
        }
        let weights = &loaded.by_input[input * hidden..(input + 1) * hidden];
        for (sum, weight) in sums.iter_mut().zip(weights.iter()) {
            *sum += weight * signal;
        }
    }
    let mut out = model.b2[0] + model.skip[0] * table * TABLE_SCALE;
    for (sum, weight) in sums.iter().zip(model.w2[..hidden].iter()) {
        out += weight * sum.max(0.0);
    }
    Some(1.0 / (1.0 + (-out).exp()))
}
