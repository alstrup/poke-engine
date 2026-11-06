#![cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]

use poke_engine::engine::evaluate::evaluate;
use poke_engine::engine::features::{features, NUM_FEATURES, SIDE_FEATURES};
use poke_engine::engine::learned::{learned_win_chance, set_eval_model, EvalModel, NUM_OUTPUTS};
use poke_engine::state::{PokemonIndex, SideReference, State};

fn signals(state: &State, side: &SideReference) -> [f32; NUM_FEATURES] {
    let mut out = [0.0; NUM_FEATURES];
    features(state, side, &mut out);
    out
}

/// A model with weights that follow a fixed pattern, so that every signal matters.
fn patterned_model(hidden: usize) -> EvalModel {
    let wave = |index: usize, scale: f32| ((index * 37 % 101) as f32 / 101.0 - 0.5) * scale;
    EvalModel {
        inputs: NUM_FEATURES,
        hidden,
        w1: (0..hidden * NUM_FEATURES).map(|i| wave(i, 0.4)).collect(),
        b1: (0..hidden).map(|i| wave(i + 7, 0.2)).collect(),
        w2: (0..NUM_OUTPUTS * hidden)
            .map(|i| wave(i + 3, 0.6))
            .collect(),
        b2: (0..NUM_OUTPUTS).map(|i| wave(i + 11, 0.2)).collect(),
    }
}

#[test]
fn test_signals_of_the_two_sides_swap_with_the_point_of_view() {
    let mut state = State::default();
    state.side_one.get_active().hp -= 40;
    state.side_two.pokemon[PokemonIndex::P2].hp = 0;
    state.side_two.side_conditions.stealth_rock = 1;
    let one = signals(&state, &SideReference::SideOne);
    let two = signals(&state, &SideReference::SideTwo);
    assert_eq!(one[..SIDE_FEATURES], two[SIDE_FEATURES..2 * SIDE_FEATURES]);
    assert_eq!(one[SIDE_FEATURES..2 * SIDE_FEATURES], two[..SIDE_FEATURES]);
    // the field is the same from both sides
    assert_eq!(one[2 * SIDE_FEATURES..], two[2 * SIDE_FEATURES..]);
}

#[test]
fn test_signals_count_what_is_left() {
    let mut state = State::default();
    state.side_two.pokemon[PokemonIndex::P2].hp = 0;
    state.side_two.pokemon[PokemonIndex::P3].hp = 0;
    let one = signals(&state, &SideReference::SideOne);
    // the first signal of a side is its share of pokemon standing, the second its share of health
    assert_eq!(one[0], 1.0);
    assert_eq!(one[1], 1.0);
    assert_eq!(one[SIDE_FEATURES], 4.0 / 6.0);
    assert_eq!(one[SIDE_FEATURES + 1], 4.0 / 6.0);
}

#[test]
fn test_signals_are_finite_with_a_fainted_active_pokemon() {
    let mut state = State::default();
    state.side_one.get_active().hp = 0;
    for value in signals(&state, &SideReference::SideOne) {
        assert!(value.is_finite());
    }
}

#[test]
fn test_model_survives_being_written_and_read() {
    let model = patterned_model(8);
    let read = EvalModel::from_text(&model.to_text()).unwrap();
    let state = State::default();
    assert_eq!(model.win_chance(&state), read.win_chance(&state));
    for output in read.outputs(&signals(&state, &SideReference::SideOne)) {
        assert!(output > 0.0 && output < 1.0);
    }
}

#[test]
fn test_model_for_other_signals_is_refused() {
    assert!(EvalModel::from_text("3 2 5\n1 2 3 4 5 6\n").is_err());
}

#[test]
fn test_search_scores_with_the_model_once_it_is_set() {
    let mut state = State::default();
    state.side_two.get_active().hp /= 3;
    state.side_two.pokemon[PokemonIndex::P4].hp = 0;
    let model = patterned_model(16);
    let plain = model.win_chance(&state);
    assert!(set_eval_model(model));
    // the quick way of working out the first layer gives the same chance as the plain way
    let quick = learned_win_chance(&state).unwrap();
    assert!((quick - plain).abs() < 1e-5);
    // and the evaluation is that chance as log odds, on the scale the search expects
    let chance = quick.clamp(0.001, 0.999);
    let score = evaluate(&state);
    assert!(score.is_finite());
    assert_eq!(score > 0.0, (chance / (1.0 - chance)).ln() > 0.0);
}
