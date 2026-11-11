use poke_engine::choices::{Choices, MOVES};
use poke_engine::engine::evaluate::{set_eval_weights, EvalWeights};
use poke_engine::engine::policy::{set_policy, Policy, INPUTS};
use poke_engine::engine::state::MoveChoice;
use poke_engine::mcts::{perform_mcts_with_movesets, Movesets};
use poke_engine::state::{Move, PokemonMoveIndex, State};

// The search settings are set once for a process, so this test has a file of its own.

#[test]
fn test_the_opponent_model_never_draws_a_move_hidden_in_every_set() {
    let _ = set_eval_weights(EvalWeights::from_spec("opponent_model=0.5").unwrap());
    // a policy with no weights, which gives every option the same chance
    assert!(set_policy(Policy {
        hidden: 1,
        w1: vec![0.0; INPUTS],
        b1: vec![0.0],
        w2: vec![0.0],
        b2: 0.0,
    }));
    let mut state = State::default();
    state.side_two.get_active().moves.m4 = Move {
        id: Choices::EARTHQUAKE,
        disabled: false,
        pp: 16,
        choice: MOVES.get(&Choices::EARTHQUAKE).unwrap().to_owned(),
    };
    let (side_one_options, side_two_options) = state.get_all_options();
    let mut movesets = Movesets::default();
    movesets[0] = vec![0b0000_1111];
    let result = perform_mcts_with_movesets(
        &mut state,
        side_one_options,
        side_two_options,
        std::time::Duration::from_millis(0),
        2000,
        &movesets,
    );
    let hidden = result
        .s2
        .iter()
        .find(|option| option.move_choice == MoveChoice::Move(PokemonMoveIndex::M4))
        .unwrap();
    assert_eq!(hidden.visits, 0);
    assert!(result.s2.iter().map(|option| option.visits).sum::<u32>() >= 2000);
}
