// A model of what a player chooses, learned from games between people: a small network that scores
// each option from what kind of choice it is and what it would do, next to the state of the game,
// so that a softmax over the scores gives the chance of each choice. The search can use it to
// expect likely replies from the other side instead of only the best ones.
// Options are described from side one's point of view; for side two the sides are swapped first.
use super::damage_calc::{calculate_damage, type_effectiveness_modifier, DamageRolls};
use super::generate_instructions::get_effective_speed;
use super::items::Items;
use super::state::MoveChoice;
use crate::choices::{Choices, MoveCategory, MoveTarget};
use crate::state::{Pokemon, PokemonType, SideReference, State};
use std::sync::OnceLock;

pub const OPTION_FEATURES: usize = 25;
pub const CONTEXT_FEATURES: usize = 15;
pub const INPUTS: usize = OPTION_FEATURES + CONTEXT_FEATURES;

fn flag(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}

fn health(pokemon: &Pokemon) -> f32 {
    if pokemon.hp <= 0 {
        0.0
    } else {
        pokemon.hp as f32 / pokemon.maxhp.max(1) as f32
    }
}

/// Share of the defender's full health that the attacker's best move takes, from stats and types.
fn rough_hit(attacker: &Pokemon, defender: &Pokemon) -> f32 {
    let mut best: f32 = 0.0;
    for mv in attacker.moves.into_iter() {
        let choice = &mv.choice;
        if mv.id == Choices::NONE
            || choice.base_power <= 0.0
            || choice.category == MoveCategory::Status
        {
            continue;
        }
        let (attack, defense) = if choice.category == MoveCategory::Physical {
            (attacker.attack as f32, defender.defense as f32)
        } else {
            (
                attacker.special_attack as f32,
                defender.special_defense as f32,
            )
        };
        let mut damage = (42.0 * choice.base_power * attack / defense.max(1.0)) / 50.0 + 2.0;
        if attacker.has_type(&choice.move_type) {
            damage *= 1.5;
        }
        damage *= type_effectiveness_modifier(&choice.move_type, defender)
            * 0.925
            * (choice.accuracy / 100.0).min(1.0);
        best = best.max(damage / defender.maxhp.max(1) as f32);
    }
    best
}

/// The best expected damage the active pokemon of `side` does to the other active one, as a share of
/// its current health.
fn best_hit_now(state: &State, side: &SideReference) -> f32 {
    let (me, them) = state.get_both_sides_immutable(side);
    let target = them.get_active_immutable();
    if target.hp <= 0 || me.get_active_immutable().hp <= 0 {
        return 0.0;
    }
    let mut best: f32 = 0.0;
    for mv in me.get_active_immutable().moves.into_iter() {
        if mv.id == Choices::NONE || mv.disabled || mv.pp <= 0 {
            continue;
        }
        if let Some((damage, _)) = calculate_damage(state, side, &mv.choice, DamageRolls::Average) {
            best = best.max(damage as f32 * (mv.choice.accuracy / 100.0).min(1.0));
        }
    }
    (best / target.hp.max(1) as f32).min(1.5)
}

pub fn context(state: &State) -> [f32; CONTEXT_FEATURES] {
    let me = &state.side_one;
    let them = &state.side_two;
    let (mine, theirs) = (me.get_active_immutable(), them.get_active_immutable());
    let alive = |side: &crate::state::Side| {
        side.pokemon.pkmn.iter().filter(|p| p.hp > 0).count() as f32 / 6.0
    };
    let boosts = |side: &crate::state::Side| {
        (side.attack_boost.max(0)
            + side.special_attack_boost.max(0)
            + side.speed_boost.max(0)
            + side.defense_boost.max(0)
            + side.special_defense_boost.max(0)) as f32
            / 6.0
    };
    let hit = best_hit_now(state, &SideReference::SideOne);
    let taken = best_hit_now(state, &SideReference::SideTwo);
    let faster = get_effective_speed(state, &SideReference::SideOne)
        > get_effective_speed(state, &SideReference::SideTwo);
    [
        health(mine),
        health(theirs),
        flag(faster),
        hit.min(1.0),
        taken.min(1.0),
        flag(hit >= 1.0),
        flag(taken >= 1.0),
        boosts(me).min(1.0),
        boosts(them).min(1.0),
        flag(them.side_conditions.stealth_rock > 0),
        flag(me.side_conditions.stealth_rock > 0),
        alive(me),
        alive(them),
        flag(mine.status != crate::state::PokemonStatus::NONE),
        flag(theirs.status != crate::state::PokemonStatus::NONE),
    ]
}

pub fn option_features(state: &State, choice: &MoveChoice) -> [f32; OPTION_FEATURES] {
    let mut out = [0.0f32; OPTION_FEATURES];
    let me = &state.side_one;
    let them = &state.side_two;
    let theirs = them.get_active_immutable();
    match choice {
        MoveChoice::Move(index) | MoveChoice::MoveTera(index) | MoveChoice::MoveMega(index) => {
            let mv = &me.get_active_immutable().moves[index];
            let c = &mv.choice;
            out[9] = flag(matches!(choice, MoveChoice::MoveTera(_)));
            let protect = matches!(
                mv.id,
                Choices::PROTECT
                    | Choices::DETECT
                    | Choices::KINGSSHIELD
                    | Choices::SPIKYSHIELD
                    | Choices::BANEFULBUNKER
                    | Choices::SILKTRAP
                    | Choices::BURNINGBULWARK
            );
            let hazard = matches!(
                mv.id,
                Choices::STEALTHROCK | Choices::SPIKES | Choices::TOXICSPIKES | Choices::STICKYWEB
            );
            let recovery = c.heal.as_ref().map_or(false, |heal| {
                heal.target == MoveTarget::User && heal.amount > 0.0
            });
            let setup = c.category == MoveCategory::Status
                && c.boost.as_ref().map_or(false, |boost| {
                    boost.target == MoveTarget::User
                        && (boost.boosts.attack > 0
                            || boost.boosts.special_attack > 0
                            || boost.boosts.speed > 0
                            || boost.boosts.defense > 0
                            || boost.boosts.special_defense > 0)
                });
            let inflict = c.category == MoveCategory::Status
                && c.status
                    .as_ref()
                    .map_or(false, |status| status.target == MoveTarget::Opponent);
            if c.category != MoveCategory::Status
                && (c.base_power > 0.0
                    || calculate_damage(state, &SideReference::SideOne, c, DamageRolls::Average)
                        .is_some())
            {
                out[0] = 1.0;
                out[6] = flag(c.flags.pivot);
                if theirs.hp > 0 {
                    if let Some((damage, _)) =
                        calculate_damage(state, &SideReference::SideOne, c, DamageRolls::Average)
                    {
                        let share =
                            damage as f32 * (c.accuracy / 100.0).min(1.0) / theirs.hp.max(1) as f32;
                        out[10] = share.min(1.0);
                        out[11] = flag(share >= 1.0);
                    }
                    let effect = type_effectiveness_modifier(&c.move_type, theirs);
                    out[13] = flag(effect > 1.0);
                    out[14] = flag(effect < 1.0);
                }
                out[12] = flag(c.priority > 0);
                out[15] = flag(me.get_active_immutable().has_type(&c.move_type));
            } else if protect {
                out[5] = 1.0;
            } else if hazard {
                out[3] = 1.0;
                out[22] =
                    flag(them.side_conditions.stealth_rock == 0 && mv.id == Choices::STEALTHROCK);
            } else if recovery {
                out[4] = 1.0;
                out[23] = flag(health(me.get_active_immutable()) < 0.5);
            } else if setup {
                out[2] = 1.0;
                out[24] = flag(me.attack_boost > 0 || me.special_attack_boost > 0);
            } else if inflict {
                out[7] = 1.0;
            } else {
                out[8] = 1.0;
            }
        }
        MoveChoice::Switch(index) => {
            out[1] = 1.0;
            let incoming = &me.pokemon[*index];
            out[16] = health(incoming);
            if theirs.hp > 0 {
                out[17] = (rough_hit(theirs, incoming) / health(incoming).max(0.05)).min(1.5) / 1.5;
                out[18] = (rough_hit(incoming, theirs) / health(theirs).max(0.05)).min(1.5) / 1.5;
                let speed = incoming.speed as f32
                    * if incoming.item == Items::CHOICESCARF {
                        1.5
                    } else {
                        1.0
                    };
                out[19] = flag(speed > get_effective_speed(state, &SideReference::SideTwo) as f32);
            }
            if me.side_conditions.stealth_rock > 0 && incoming.item != Items::HEAVYDUTYBOOTS {
                out[20] = type_effectiveness_modifier(&PokemonType::ROCK, incoming) / 8.0;
            }
            out[21] = flag(incoming.item == Items::HEAVYDUTYBOOTS);
        }
        _ => {}
    }
    out
}

#[derive(Clone, Debug)]
pub struct Policy {
    pub hidden: usize,
    /// hidden rows of INPUTS weights each
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: f32,
}

impl Policy {
    /// The score of one option, from its row of inputs; `hidden` receives the hidden layer.
    pub fn score(&self, input: &[f32], hidden: &mut [f32]) -> f32 {
        let mut out = self.b2;
        for unit in 0..self.hidden {
            let weights = &self.w1[unit * INPUTS..(unit + 1) * INPUTS];
            let sum = self.b1[unit] + weights.iter().zip(input).map(|(w, x)| w * x).sum::<f32>();
            hidden[unit] = sum.max(0.0);
            out += self.w2[unit] * hidden[unit];
        }
        out
    }

    pub fn to_text(&self) -> String {
        let mut text = format!(
            "policy {} {} {}\n",
            OPTION_FEATURES, CONTEXT_FEATURES, self.hidden
        );
        for values in [&self.w1, &self.b1, &self.w2] {
            text.push_str(
                &values
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            );
            text.push('\n');
        }
        text.push_str(&format!("{}\n", self.b2));
        text
    }

    pub fn from_text(text: &str) -> Result<Policy, String> {
        let mut words = text.split_whitespace();
        if words.next() != Some("policy") {
            return Err("not a policy file".to_string());
        }
        let mut size = || -> Result<usize, String> {
            words
                .next()
                .and_then(|word| word.parse().ok())
                .ok_or("policy file has no sizes".to_string())
        };
        let (options, context, hidden) = (size()?, size()?, size()?);
        if options != OPTION_FEATURES || context != CONTEXT_FEATURES {
            return Err(format!(
                "policy is for {} and {} features, but this engine has {} and {}",
                options, context, OPTION_FEATURES, CONTEXT_FEATURES
            ));
        }
        let values: Result<Vec<f32>, _> = words.map(|word| word.parse::<f32>()).collect();
        let values =
            values.map_err(|_| "policy file has something that is not a number".to_string())?;
        if values.len() != hidden * INPUTS + 2 * hidden + 1 {
            return Err(format!("policy file has {} numbers", values.len()));
        }
        let (w1, rest) = values.split_at(hidden * INPUTS);
        let (b1, rest) = rest.split_at(hidden);
        let (w2, b2) = rest.split_at(hidden);
        Ok(Policy {
            hidden,
            w1: w1.to_vec(),
            b1: b1.to_vec(),
            w2: w2.to_vec(),
            b2: b2[0],
        })
    }

    /// The chance of each option, for side one choosing in this position.
    pub fn chances(&self, state: &State, options: &[MoveChoice]) -> Vec<f32> {
        let context = context(state);
        let mut hidden = vec![0.0f32; self.hidden];
        let mut row = [0.0f32; INPUTS];
        row[OPTION_FEATURES..].copy_from_slice(&context);
        let scores: Vec<f32> = options
            .iter()
            .map(|option| {
                row[..OPTION_FEATURES].copy_from_slice(&option_features(state, option));
                self.score(&row, &mut hidden)
            })
            .collect();
        let top = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
        let total: f32 = exps.iter().sum();
        exps.iter().map(|e| e / total.max(1e-9)).collect()
    }
}

static POLICY: OnceLock<Policy> = OnceLock::new();

/// Gives the search a model of what the other side is likely to choose. Only the first call counts.
pub fn set_policy(policy: Policy) -> bool {
    POLICY.set(policy).is_ok()
}

pub fn policy() -> Option<&'static Policy> {
    POLICY.get()
}

/// The chance of each of side two's options by the policy, if one is set: the sides are swapped
/// for the while, so that the options are seen from the chooser's point of view.
pub fn side_two_chances(state: &mut State, options: &[MoveChoice]) -> Option<Vec<f32>> {
    let policy = POLICY.get()?;
    std::mem::swap(&mut state.side_one, &mut state.side_two);
    let chances = policy.chances(state, options);
    std::mem::swap(&mut state.side_one, &mut state.side_two);
    Some(chances)
}
