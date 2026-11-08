use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::StateInstructions;
use poke_engine::mcts::{MoveNode, Node};

fn node_with(options: Vec<MoveNode>) -> Node {
    Node {
        root: true,
        parent: std::ptr::null_mut(),
        times_visited: 100,
        instructions: StateInstructions::default(),
        s1_choice: 0,
        s2_choice: 0,
        s1_options: None,
        s2_options: Some(options),
        s2_prior: None,
    }
}

#[test]
fn test_switch_penalty_makes_a_slightly_better_switch_lose_to_a_move() {
    // equal visits, so the exploration terms are equal and only the averages decide
    let options = || {
        vec![
            MoveNode {
                move_choice: MoveChoice::Move(poke_engine::state::PokemonMoveIndex::M0),
                total_score: 25.0,
                visits: 50,
            },
            MoveNode {
                move_choice: MoveChoice::Switch(poke_engine::state::PokemonIndex::P1),
                total_score: 26.0,
                visits: 50,
            },
        ]
    };
    let node = node_with(options());
    assert_eq!(
        node.maximize_ucb_for_side(node.s2_options.as_ref().unwrap()),
        1
    );
    assert_eq!(
        node.maximize_ucb_with_switch_penalty(node.s2_options.as_ref().unwrap(), 0.0),
        1
    );
    assert_eq!(
        node.maximize_ucb_with_switch_penalty(node.s2_options.as_ref().unwrap(), 0.05),
        0
    );
}
