#![cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]

use poke_engine::choices::Choices;
use poke_engine::engine::evaluate::{evaluate, table_evaluate};
use poke_engine::engine::features::{features, NUM_FEATURES, SIDE_FEATURES};
use poke_engine::engine::learned::{
    learned_win_chance, models_from_text, models_to_text, set_eval_model, stage_of, EvalModel,
    NUM_OUTPUTS, TABLE_SCALE,
};
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, SideReference, State};

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
        skip: (0..NUM_OUTPUTS).map(|i| 0.5 + wave(i + 5, 0.2)).collect(),
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
    assert_eq!(
        model.win_chance(&state, 30.0),
        read.win_chance(&state, 30.0)
    );
    for output in read.outputs(&signals(&state, &SideReference::SideOne), 0.3) {
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
    let table = table_evaluate(&state);
    let plain = model.win_chance(&state, table);
    assert!(set_eval_model(model));
    // the quick way of working out the first layer gives the same chance as the plain way
    let quick = learned_win_chance(&state, table).unwrap();
    assert!((quick - plain).abs() < 1e-5);
    // and the evaluation is that chance as log odds, on the scale the search expects
    let chance = quick.clamp(0.001, 0.999);
    let score = evaluate(&state);
    assert!(score.is_finite());
    assert_eq!(score > 0.0, (chance / (1.0 - chance)).ln() > 0.0);
}

#[test]
fn test_model_without_skip_weights_is_still_read() {
    let mut model = patterned_model(4);
    model.skip = vec![0.0; NUM_OUTPUTS];
    let text = model.to_text();
    // a file written before the skip weights existed ends after b2
    let old: String = text.lines().take(5).collect::<Vec<_>>().join("\n");
    let read = EvalModel::from_text(&old).unwrap();
    assert_eq!(read.skip, vec![0.0; NUM_OUTPUTS]);
    let state = State::default();
    assert_eq!(
        model.win_chance(&state, 50.0),
        read.win_chance(&state, 50.0)
    );
}

#[test]
fn test_skip_weight_passes_the_table_through() {
    // a network that adds nothing, with a skip weight of one, gives the table's own chance
    let hidden = 2;
    let model = EvalModel {
        inputs: NUM_FEATURES,
        hidden,
        w1: vec![0.0; hidden * NUM_FEATURES],
        b1: vec![0.0; hidden],
        w2: vec![0.0; NUM_OUTPUTS * hidden],
        b2: vec![0.0; NUM_OUTPUTS],
        skip: vec![1.0; NUM_OUTPUTS],
    };
    let state = State::default();
    let chance = model.win_chance(&state, 80.0);
    assert!((chance - 1.0 / (1.0 + (-80.0 * TABLE_SCALE).exp())).abs() < 1e-6);
}

#[test]
fn test_a_network_for_each_stage_is_written_and_read() {
    let models = vec![patterned_model(3), patterned_model(5), patterned_model(7)];
    let read = models_from_text(&models_to_text(&models)).unwrap();
    assert_eq!(read.len(), 3);
    assert_eq!(read[1].hidden, 5);
    assert_eq!(read[2].w2, models[2].w2);
    // a single network is still written and read as before
    let single = models_from_text(&models_to_text(&models[..1])).unwrap();
    assert_eq!(single.len(), 1);
    assert!(models_from_text("stages 3\nnetwork\n1 2 5\n").is_err());
}

#[test]
fn test_stage_follows_the_side_with_fewest_left() {
    let mut state = State::default();
    assert_eq!(stage_of(&state), 0);
    state.side_two.pokemon[PokemonIndex::P1].hp = 0;
    state.side_two.pokemon[PokemonIndex::P2].hp = 0;
    assert_eq!(stage_of(&state), 1);
    state.side_one.pokemon[PokemonIndex::P1].hp = 0;
    state.side_one.pokemon[PokemonIndex::P2].hp = 0;
    state.side_one.pokemon[PokemonIndex::P3].hp = 0;
    state.side_one.pokemon[PokemonIndex::P4].hp = 0;
    assert_eq!(stage_of(&state), 2);
}

#[test]
fn test_resource_signals_see_rocks_that_could_be_set() {
    // the eleven resource signals end each side's block
    let rocks_possible = SIDE_FEATURES - 11;
    let mut state = State::default();
    let before = signals(&state, &SideReference::SideOne)[rocks_possible];
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::STEALTHROCK);
    let with_setter = signals(&state, &SideReference::SideOne)[rocks_possible];
    assert_eq!(before, 0.0);
    assert!(with_setter > 0.0);
    // once rocks are up, there is nothing more to gain by setting them
    state.side_two.side_conditions.stealth_rock = 1;
    let set = signals(&state, &SideReference::SideOne);
    assert_eq!(set[rocks_possible], 0.0);
    // and the other side now feels them
    assert!(set[SIDE_FEATURES + rocks_possible + 1] > 0.0);
}

#[test]
fn test_opponent_boosts_can_count_for_more() {
    use poke_engine::engine::evaluate::{table_evaluate_with, EvalWeights};
    let mut state = State::default();
    state.side_two.attack_boost = 2;
    let plain = table_evaluate_with(&state, &EvalWeights::default());
    let feared = table_evaluate_with(
        &state,
        &EvalWeights::from_spec("opponent_boost_scale=2").unwrap(),
    );
    assert!(feared < plain);
    // side one's boosts are not touched
    let mut mine = State::default();
    mine.side_one.attack_boost = 2;
    assert_eq!(
        table_evaluate_with(&mine, &EvalWeights::default()),
        table_evaluate_with(
            &mine,
            &EvalWeights::from_spec("opponent_boost_scale=2").unwrap()
        )
    );
}
