use poke_engine::choices::{Choices, MOVES};
use poke_engine::engine::state::MoveChoice;
use poke_engine::state::{Move, PokemonMoveIndex, State};

fn a_move(id: Choices) -> Move {
    Move {
        id,
        disabled: false,
        pp: 16,
        choice: MOVES.get(&id).unwrap().to_owned(),
    }
}

#[test]
fn test_a_pokemon_with_four_moves_is_written_as_before() {
    let mut state = State::default();
    let text = state.side_one.get_active().serialize();
    assert_eq!(text.split(',').count(), 29);
}

#[test]
fn test_extra_moves_are_written_and_read_back() {
    let mut state = State::default();
    state.side_two.get_active().moves.m4 = a_move(Choices::EARTHQUAKE);
    state.side_two.get_active().moves.m5 = a_move(Choices::ICEBEAM);
    let text = state.side_two.get_active().serialize();
    let read = poke_engine::state::Pokemon::deserialize(&text);
    assert_eq!(read.moves.m4.id, Choices::EARTHQUAKE);
    assert_eq!(read.moves.m5.id, Choices::ICEBEAM);
    assert_eq!(read.moves.m6.id, Choices::NONE);
    assert_eq!(read.serialize(), text);
}

#[test]
fn test_extra_moves_are_options_and_empty_extra_slots_are_not() {
    let mut state = State::default();
    state.side_two.get_active().moves.m4 = a_move(Choices::EARTHQUAKE);
    let (_, side_two_options) = state.get_all_options();
    assert!(side_two_options.contains(&MoveChoice::Move(PokemonMoveIndex::M4)));
    assert!(!side_two_options.contains(&MoveChoice::Move(PokemonMoveIndex::M5)));
}

#[test]
fn test_a_search_never_chooses_a_move_hidden_in_every_set() {
    use poke_engine::mcts::{perform_mcts_with_movesets, Movesets};
    let mut state = State::default();
    state.side_two.get_active().moves.m4 = a_move(Choices::EARTHQUAKE);
    let (side_one_options, side_two_options) = state.get_all_options();
    assert!(side_two_options.contains(&MoveChoice::Move(PokemonMoveIndex::M4)));
    // every set has the first four slots and never the fifth
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
    assert_eq!(state.side_two.get_active().moves.hidden, 0);
}
