#![cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]

use poke_engine::engine::policy::{set_policy, side_two_chances, Policy, INPUTS};
use poke_engine::state::State;

fn patterned(hidden: usize) -> Policy {
    let wave = |index: usize, scale: f32| ((index * 31 % 97) as f32 / 97.0 - 0.5) * scale;
    Policy {
        hidden,
        w1: (0..hidden * INPUTS).map(|i| wave(i, 0.6)).collect(),
        b1: (0..hidden).map(|i| wave(i + 3, 0.2)).collect(),
        w2: (0..hidden).map(|i| wave(i + 7, 0.8)).collect(),
        b2: 0.1,
    }
}

#[test]
fn test_policy_survives_being_written_and_read() {
    let policy = patterned(6);
    let read = Policy::from_text(&policy.to_text()).unwrap();
    let state = State::default();
    let (options, _) = state.root_get_all_options();
    assert_eq!(
        policy.chances(&state, &options),
        read.chances(&state, &options)
    );
    assert!(Policy::from_text("policy 3 2 1\n1 2 3\n").is_err());
}

#[test]
fn test_chances_add_up_and_side_two_is_left_as_it_was() {
    let mut state = State::default();
    state.side_two.get_active().hp /= 2;
    let before = state.serialize();
    let (_, options) = state.root_get_all_options();
    assert!(set_policy(patterned(8)));
    let chances = side_two_chances(&mut state, &options).unwrap();
    assert_eq!(chances.len(), options.len());
    assert!((chances.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    assert!(chances.iter().all(|&chance| chance >= 0.0));
    // the sides are swapped only for the while
    assert_eq!(state.serialize(), before);
}
