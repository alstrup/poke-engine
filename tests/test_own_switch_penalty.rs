use poke_engine::engine::evaluate::{set_eval_weights, EvalWeights};
use poke_engine::engine::state::MoveChoice;
use poke_engine::mcts::{perform_mcts_with_movesets, Movesets};
use poke_engine::state::State;

// The search settings are set once for a process, so this test has a file of its own.

#[test]
fn test_a_large_own_switch_penalty_keeps_side_one_from_searching_switches() {
    // a search score lies between 0 and 1, so a penalty of 2 puts every switch below every move
    let _ = set_eval_weights(EvalWeights::from_spec("own_switch_penalty=2").unwrap());
    let mut state = State::default();
    let (side_one_options, side_two_options) = state.get_all_options();
    let switches = side_one_options
        .iter()
        .filter(|option| matches!(option, MoveChoice::Switch(_)))
        .count() as u32;
    assert!(switches > 0);
    let result = perform_mcts_with_movesets(
        &mut state,
        side_one_options,
        side_two_options,
        std::time::Duration::from_millis(0),
        2000,
        &Movesets::default(),
    );
    // each switch is tried while it is unvisited, as every option is, and not after that
    for option in &result.s1 {
        if matches!(option.move_choice, MoveChoice::Switch(_)) {
            assert!(option.visits < 10, "{} visits", option.visits);
        }
    }
    // side two's switches are not penalised by this setting
    assert!(result
        .s2
        .iter()
        .any(|option| matches!(option.move_choice, MoveChoice::Switch(_)) && option.visits > 1));
}
