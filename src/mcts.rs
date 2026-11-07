use crate::engine::evaluate::{eval_weights, evaluate};
use crate::engine::generate_instructions::generate_instructions_from_move_pair;
use crate::engine::policy::side_two_chances;
use crate::engine::state::MoveChoice;
use crate::instruction::{Instruction, StateInstructions};
use crate::state::State;
use rand::prelude::*;
use rand::rng;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::time::Duration;

#[cfg(target_arch = "wasm32")]
use web_time::Instant;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

/// How often a node must have been visited before the policy is asked about side two's options.
const OPPONENT_MODEL_AFTER: u32 = 4;

/// Hashes the children map's keys, which are a node's address and two option indices. The keys
/// come from the search itself, so a fast multiply-and-rotate hash (as in rustc's FxHash) is
/// enough, and much cheaper than the standard library's SipHash.
#[derive(Default, Clone, Copy)]
pub struct ChildKeyHasher(u64);

impl Hasher for ChildKeyHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.write_u64(*byte as u64);
        }
    }
    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

/// The children of each node, by the node's address and the two sides' option indices.
pub type Children = HashMap<(usize, usize, usize), Box<[Node]>, BuildHasherDefault<ChildKeyHasher>>;

fn sigmoid(x: f32) -> f32 {
    // Tuned so that ~200 points is very close to 1.0
    1.0 / (1.0 + (-eval_weights().sigmoid_scale * x).exp())
}

#[derive(Debug)]
pub struct Node {
    pub root: bool,
    pub parent: *mut Node,
    pub times_visited: u32,

    // represents the instructions & s1/s2 moves that led to this node from the parent
    pub instructions: StateInstructions,
    pub s1_choice: u8,
    pub s2_choice: u8,

    // represents the total score and number of visits for this node
    // de-coupled for s1 and s2
    pub s1_options: Option<Vec<MoveNode>>,
    pub s2_options: Option<Vec<MoveNode>>,
    /// the policy's chance of each of side two's options, worked out once the node has been visited
    /// a few times
    pub s2_prior: Option<Vec<f32>>,
}

impl Node {
    fn new() -> Node {
        Node {
            root: false,
            parent: std::ptr::null_mut(),
            instructions: StateInstructions::default(),
            times_visited: 0,
            s1_choice: 0,
            s2_choice: 0,
            s1_options: None,
            s2_options: None,
            s2_prior: None,
        }
    }
    unsafe fn populate(&mut self, s1_options: Vec<MoveChoice>, s2_options: Vec<MoveChoice>) {
        let s1_options_vec: Vec<MoveNode> = s1_options
            .iter()
            .map(|x| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
            })
            .collect();
        let s2_options_vec: Vec<MoveNode> = s2_options
            .iter()
            .map(|x| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
            })
            .collect();

        self.s1_options = Some(s1_options_vec);
        self.s2_options = Some(s2_options_vec);
    }

    pub fn maximize_ucb_for_side(&self, side_map: &[MoveNode]) -> usize {
        let mut choice = 0;
        let mut best_ucb1 = f32::MIN;
        // the exploration term's numerator is the same for every option, so it is worked out once
        let numerator = eval_weights().exploration * (self.times_visited as f32).ln();
        for (index, node) in side_map.iter().enumerate() {
            let this_ucb1 = node.ucb1_with(numerator);
            if this_ucb1 > best_ucb1 {
                best_ucb1 = this_ucb1;
                choice = index;
            }
        }
        choice
    }

    pub unsafe fn selection(
        &mut self,
        state: &mut State,
        children: &mut Children,
        rng: &mut impl Rng,
    ) -> (*mut Node, usize, usize) {
        if self.s1_options.is_none() {
            let (s1_options, s2_options) = state.get_all_options();
            self.populate(s1_options, s2_options);
        }

        let s1_mc_index = self.maximize_ucb_for_side(self.s1_options.as_ref().unwrap());
        let s2_mc_index = self.side_two_choice(state, rng);
        let key = (self as *mut Node as usize, s1_mc_index, s2_mc_index);
        match children.get_mut(&key) {
            Some(child_vector) => {
                let child_vec_ptr = child_vector as *mut Box<[Node]>;
                let chosen_child = self.sample_node(child_vec_ptr, rng);
                state.apply_instructions(&(*chosen_child).instructions.instruction_list);
                (*chosen_child).selection(state, children, rng)
            }
            None => (self as *mut Node, s1_mc_index, s2_mc_index),
        }
    }

    /// Side two's choice: by the search's own rule, or some of the time, once the node has been
    /// visited a few times and a policy is set, drawn from what people are likely to choose.
    fn side_two_choice(&mut self, state: &mut State, rng: &mut impl Rng) -> usize {
        let share = eval_weights().opponent_model;
        if share > 0.0 && self.times_visited >= OPPONENT_MODEL_AFTER && rng.random::<f32>() < share
        {
            if self.s2_prior.is_none() {
                let options: Vec<MoveChoice> = self
                    .s2_options
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|option| option.move_choice.clone())
                    .collect();
                self.s2_prior = Some(side_two_chances(state, &options).unwrap_or_default());
            }
            let prior = self.s2_prior.as_ref().unwrap();
            if prior.len() == self.s2_options.as_ref().unwrap().len() {
                let mut threshold = rng.random::<f32>();
                for (index, chance) in prior.iter().enumerate() {
                    threshold -= chance;
                    if threshold <= 0.0 {
                        return index;
                    }
                }
                return prior.len() - 1;
            }
        }
        self.maximize_ucb_for_side(self.s2_options.as_ref().unwrap())
    }

    unsafe fn sample_node(&self, move_vector: *mut Box<[Node]>, rng: &mut impl Rng) -> *mut Node {
        let nodes = &mut **move_vector;

        let total_weight: f32 = nodes
            .iter()
            .map(|n| n.instructions.percentage.max(0.0))
            .sum();

        let mut threshold = rng.random_range(0.0..total_weight);

        for node in nodes.iter_mut() {
            threshold -= node.instructions.percentage.max(0.0);
            if threshold <= 0.0 {
                return node as *mut Node;
            }
        }

        // fallback: return last node (handles float rounding issues that can come up)
        &mut nodes[nodes.len() - 1] as *mut Node
    }

    pub unsafe fn expand(
        &mut self,
        state: &mut State,
        s1_move_index: usize,
        s2_move_index: usize,
        children: &mut Children,
        rng: &mut impl Rng,
    ) -> *mut Node {
        let s1_move = &self.s1_options.as_ref().unwrap()[s1_move_index].move_choice;
        let s2_move = &self.s2_options.as_ref().unwrap()[s2_move_index].move_choice;
        // if the battle is over or both moves are none there is no need to expand
        if (state.battle_is_over() != 0.0 && !self.root)
            || (s1_move == &MoveChoice::None && s2_move == &MoveChoice::None)
        {
            return self as *mut Node;
        }
        let should_branch_on_damage = self.root || (*self.parent).root;
        let mut new_instructions =
            generate_instructions_from_move_pair(state, s1_move, s2_move, should_branch_on_damage);
        let mut this_pair_vec = Vec::with_capacity(new_instructions.len());
        for state_instructions in new_instructions.drain(..) {
            let mut new_node = Node::new();
            new_node.parent = self;
            new_node.instructions = state_instructions;
            new_node.s1_choice = s1_move_index as u8;
            new_node.s2_choice = s2_move_index as u8;
            this_pair_vec.push(new_node);
        }

        // sample a node from the new instruction list.
        // this is the node that the rollout will be done on.
        // into_boxed_slice drops the Vec's spare capacity and, more importantly,
        // makes it a type that cannot be resized, which ensures the node
        // addresses are stable for the children map keys
        let mut boxed = this_pair_vec.into_boxed_slice();
        let new_node_ptr = self.sample_node(&mut boxed, rng);
        state.apply_instructions(&(*new_node_ptr).instructions.instruction_list);

        let key = (self as *mut Node as usize, s1_move_index, s2_move_index);
        children.insert(key, boxed);
        new_node_ptr
    }

    pub unsafe fn backpropagate(&mut self, score: f32, state: &mut State) {
        self.times_visited += 1;
        if self.root {
            return;
        }

        let parent_s1_movenode =
            &mut (*self.parent).s1_options.as_mut().unwrap()[self.s1_choice as usize];
        parent_s1_movenode.total_score += score;
        parent_s1_movenode.visits += 1;

        let parent_s2_movenode =
            &mut (*self.parent).s2_options.as_mut().unwrap()[self.s2_choice as usize];
        parent_s2_movenode.total_score += 1.0 - score;
        parent_s2_movenode.visits += 1;

        state.reverse_instructions(&self.instructions.instruction_list);
        (*self.parent).backpropagate(score, state);
    }

    pub fn rollout(&mut self, state: &mut State, root_eval: &f32, rng: &mut impl Rng) -> f32 {
        // optionally play a few turns of random moves first, so that a position is not scored in
        // the middle of an exchange
        let weights = eval_weights();
        let mut playout_turns = weights.playout_turns as usize;
        if playout_turns > 0
            && weights.playout_share < 1.0
            && rng.random::<f32>() >= weights.playout_share
        {
            playout_turns = 0;
        }
        let by_damage = weights.playout_policy >= 1.0;
        let mut played: Vec<Vec<Instruction>> = Vec::with_capacity(playout_turns.min(64));
        for _ in 0..playout_turns {
            if state.battle_is_over() != 0.0 {
                break;
            }
            let (s1_options, s2_options) = state.get_all_options();
            let s1_move = if by_damage {
                playout_choice_by_damage(state, &s1_options, true, rng)
            } else {
                playout_choice(&s1_options, rng)
            };
            let s2_move = if by_damage {
                playout_choice_by_damage(state, &s2_options, false, rng)
            } else {
                playout_choice(&s2_options, rng)
            };
            if s1_move == MoveChoice::None && s2_move == MoveChoice::None {
                break;
            }
            let mut outcomes =
                generate_instructions_from_move_pair(state, &s1_move, &s2_move, false);
            if outcomes.is_empty() {
                break;
            }
            let total: f32 = outcomes.iter().map(|o| o.percentage.max(0.0)).sum();
            let mut chosen = 0;
            if total > 0.0 {
                let mut threshold = rng.random_range(0.0..total);
                for (index, outcome) in outcomes.iter().enumerate() {
                    threshold -= outcome.percentage.max(0.0);
                    chosen = index;
                    if threshold <= 0.0 {
                        break;
                    }
                }
            }
            let instructions = outcomes.swap_remove(chosen).instruction_list;
            state.apply_instructions(&instructions);
            played.push(instructions);
        }

        let battle_is_over = state.battle_is_over();
        let score = if battle_is_over == 0.0 {
            let eval = evaluate(state);
            sigmoid(eval - root_eval)
        } else {
            if battle_is_over == -1.0 {
                0.0
            } else {
                battle_is_over
            }
        };
        for instructions in played.iter().rev() {
            state.reverse_instructions(instructions);
        }
        score
    }
}

// the damage one side's move does to the other side's active pokemon in the most likely outcome
fn playout_damage(state: &mut State, choice: &MoveChoice, side_one: bool) -> i32 {
    let before = if side_one {
        state.side_two.get_active_immutable().hp
    } else {
        state.side_one.get_active_immutable().hp
    } as i32;
    let outcomes = if side_one {
        generate_instructions_from_move_pair(state, choice, &MoveChoice::None, false)
    } else {
        generate_instructions_from_move_pair(state, &MoveChoice::None, choice, false)
    };
    let likeliest = match outcomes
        .iter()
        .max_by(|a, b| a.percentage.total_cmp(&b.percentage))
    {
        Some(outcome) => outcome,
        None => return 0,
    };
    state.apply_instructions(&likeliest.instruction_list);
    let after = if side_one {
        state.side_two.get_active_immutable().hp
    } else {
        state.side_one.get_active_immutable().hp
    } as i32;
    state.reverse_instructions(&likeliest.instruction_list);
    before - after
}

// a playout that uses the move doing most damage, with one choice in ten at random for variety
fn playout_choice_by_damage(
    state: &mut State,
    options: &Vec<MoveChoice>,
    side_one: bool,
    rng: &mut impl Rng,
) -> MoveChoice {
    if options.is_empty() {
        return MoveChoice::None;
    }
    if options.len() == 1 || rng.random::<f32>() < 0.1 {
        return options[rng.random_range(0..options.len())].clone();
    }
    let mut best: Option<MoveChoice> = None;
    let mut best_damage = -1;
    for candidate in options.iter() {
        if !matches!(candidate, MoveChoice::Move(_)) {
            continue;
        }
        let damage = playout_damage(state, candidate, side_one);
        if damage > best_damage {
            best_damage = damage;
            best = Some(candidate.clone());
        }
    }
    match best {
        Some(choice) => choice,
        None => options[rng.random_range(0..options.len())].clone(),
    }
}

// a playout attacks or sets up, and only switches when it has to
fn playout_choice(options: &Vec<MoveChoice>, rng: &mut impl Rng) -> MoveChoice {
    let moves: Vec<&MoveChoice> = options
        .iter()
        .filter(|choice| matches!(choice, MoveChoice::Move(_)))
        .collect();
    if !moves.is_empty() {
        return moves[rng.random_range(0..moves.len())].clone();
    }
    if options.is_empty() {
        return MoveChoice::None;
    }
    options[rng.random_range(0..options.len())].clone()
}

#[derive(Debug)]
pub struct MoveNode {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: u32,
}

impl MoveNode {
    /// The same as `ucb1`, given the exploration constant times the log of the parent's visits.
    #[inline]
    pub fn ucb1_with(&self, numerator: f32) -> f32 {
        if self.visits == 0 {
            return f32::INFINITY;
        }
        (self.total_score / self.visits as f32) + (numerator / self.visits as f32).sqrt()
    }
    pub fn ucb1(&self, parent_visits: u32) -> f32 {
        if self.visits == 0 {
            return f32::INFINITY;
        }
        let score = (self.total_score / self.visits as f32)
            + (eval_weights().exploration * (parent_visits as f32).ln() / self.visits as f32)
                .sqrt();
        score
    }
    pub fn average_score(&self) -> f32 {
        let score = self.total_score / self.visits as f32;
        score
    }
}

#[derive(Clone)]
pub struct MctsSideResult {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: u32,
}

impl MctsSideResult {
    pub fn average_score(&self) -> f32 {
        if self.visits == 0 {
            return 0.0;
        }
        let score = self.total_score / self.visits as f32;
        score
    }
}

pub struct MctsResult {
    pub s1: Vec<MctsSideResult>,
    pub s2: Vec<MctsSideResult>,
    pub iteration_count: u32,
}

fn mcts_iteration(
    root_node: &mut Node,
    state: &mut State,
    root_eval: &f32,
    children: &mut Children,
    rng: &mut impl Rng,
) {
    let (mut new_node, s1_move, s2_move) = unsafe { root_node.selection(state, children, rng) };
    new_node = unsafe { (*new_node).expand(state, s1_move, s2_move, children, rng) };
    let rollout_result = unsafe { (*new_node).rollout(state, root_eval, rng) };
    unsafe { (*new_node).backpropagate(rollout_result, state) }
}

enum SearchLimit {
    Time(Duration),
    Iterations(u32),
}

fn run_mcts_loop(
    root_node: &mut Node,
    state: &mut State,
    root_eval: &f32,
    children: &mut Children,
    limit: SearchLimit,
) {
    let mut rng = rng();
    let start_time = Instant::now();
    loop {
        for _ in 0..1000 {
            mcts_iteration(root_node, state, root_eval, children, &mut rng);
        }
        if root_node.times_visited >= 10_000_000 {
            break;
        }
        match limit {
            SearchLimit::Time(max_time) => {
                if start_time.elapsed() >= max_time {
                    break;
                }
            }
            SearchLimit::Iterations(n) => {
                if root_node.times_visited >= n {
                    break;
                }
            }
        }
    }
}

pub fn perform_mcts(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    max_iterations: u32,
) -> MctsResult {
    let mut root_node = Node::new();
    unsafe {
        root_node.populate(side_one_options, side_two_options);
    }
    root_node.root = true;
    let mut children: Children = Children::default();

    let root_eval = evaluate(state);
    let search_limit = if max_iterations > 0 {
        SearchLimit::Iterations(max_iterations)
    } else {
        SearchLimit::Time(max_time)
    };
    run_mcts_loop(
        &mut root_node,
        state,
        &root_eval,
        &mut children,
        search_limit,
    );

    let result = MctsResult {
        s1: root_node
            .s1_options
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        s2: root_node
            .s2_options
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        iteration_count: root_node.times_visited,
    };

    result
}
