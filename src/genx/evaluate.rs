use super::abilities::Abilities;
use super::items::Items;
use super::learned::learned_win_chance;
use super::state::PokemonVolatileStatus;
use crate::choices::MoveCategory;
use crate::state::{Pokemon, PokemonStatus, Side, State};
use std::sync::OnceLock;

/// The weights of the hand-written evaluation. They can be replaced once, before any search, so
/// that different weights can be played against each other.
#[derive(Debug, Clone)]
pub struct EvalWeights {
    pub pokemon_alive: f32,
    pub pokemon_hp: f32,
    pub used_tera: f32,
    pub pokemon_attack_boost: f32,
    pub pokemon_defense_boost: f32,
    pub pokemon_special_attack_boost: f32,
    pub pokemon_special_defense_boost: f32,
    pub pokemon_speed_boost: f32,
    pub pokemon_frozen: f32,
    pub pokemon_asleep: f32,
    pub pokemon_paralyzed: f32,
    pub pokemon_toxic: f32,
    pub pokemon_poisoned: f32,
    pub pokemon_burned: f32,
    pub leech_seed: f32,
    pub substitute: f32,
    pub confusion: f32,
    pub reflect: f32,
    pub light_screen: f32,
    pub aurora_veil: f32,
    pub safe_guard: f32,
    pub tailwind: f32,
    pub healing_wish: f32,
    pub stealth_rock: f32,
    pub spikes: f32,
    pub toxic_spikes: f32,
    pub sticky_web: f32,
    pub has_item: f32,
    pub sigmoid_scale: f32,
    /// the constant inside the square root of the search's exploration term
    pub exploration: f32,
    /// how many turns the search plays past a new position before scoring it
    pub playout_turns: f32,
    /// how a playout chooses: 0 picks a move at random, 1 the move that does most damage
    pub playout_policy: f32,
    /// the share of new positions that get a playout; the rest are scored at once
    pub playout_share: f32,
    /// the share of side two's choices in the search drawn from the policy of what people choose,
    /// where one is set, instead of the search's own rule; 0 leaves the search as it was
    pub opponent_model: f32,
    /// side two's boosts count this many times as much as side one's in the table
    pub opponent_boost_scale: f32,
    /// taken off the average score of side two's switches when the search chooses for side two,
    /// so that it expects the opponent to switch less often; 0 leaves the search as it was
    pub opponent_switch_penalty: f32,
    /// taken off the average score of side one's switches when the search chooses for side one,
    /// so that it stays in to attack or set up more often; 0 leaves the search as it was
    pub own_switch_penalty: f32,
}

impl Default for EvalWeights {
    fn default() -> EvalWeights {
        EvalWeights {
            pokemon_alive: 30.0,
            pokemon_hp: 100.0,
            used_tera: -75.0,
            pokemon_attack_boost: 30.0,
            pokemon_defense_boost: 15.0,
            pokemon_special_attack_boost: 30.0,
            pokemon_special_defense_boost: 15.0,
            pokemon_speed_boost: 30.0,
            pokemon_frozen: -40.0,
            pokemon_asleep: -25.0,
            pokemon_paralyzed: -25.0,
            pokemon_toxic: -30.0,
            pokemon_poisoned: -10.0,
            pokemon_burned: -25.0,
            leech_seed: -30.0,
            substitute: 75.0,
            confusion: -20.0,
            reflect: 20.0,
            light_screen: 20.0,
            aurora_veil: 40.0,
            safe_guard: 5.0,
            tailwind: 7.0,
            healing_wish: 30.0,
            stealth_rock: -10.0,
            spikes: -7.0,
            toxic_spikes: -7.0,
            sticky_web: -25.0,
            has_item: 10.0,
            sigmoid_scale: 0.0125,
            exploration: 2.0,
            playout_turns: 0.0,
            playout_policy: 0.0,
            playout_share: 1.0,
            opponent_model: 0.0,
            opponent_boost_scale: 1.0,
            opponent_switch_penalty: 0.0,
            own_switch_penalty: 0.0,
        }
    }
}

impl EvalWeights {
    /// Reads weights written as "name=value,name=value" over the defaults.
    pub fn from_spec(spec: &str) -> Result<EvalWeights, String> {
        let mut weights = EvalWeights::default();
        for part in spec.split(',').filter(|part| !part.trim().is_empty()) {
            let (name, value) = part
                .split_once('=')
                .ok_or(format!("expected name=value, got {}", part))?;
            let value: f32 = value
                .trim()
                .parse()
                .map_err(|_| format!("not a number: {}", value))?;
            match name.trim() {
                "pokemon_alive" => weights.pokemon_alive = value,
                "pokemon_hp" => weights.pokemon_hp = value,
                "used_tera" => weights.used_tera = value,
                "pokemon_attack_boost" => weights.pokemon_attack_boost = value,
                "pokemon_defense_boost" => weights.pokemon_defense_boost = value,
                "pokemon_special_attack_boost" => weights.pokemon_special_attack_boost = value,
                "pokemon_special_defense_boost" => weights.pokemon_special_defense_boost = value,
                "pokemon_speed_boost" => weights.pokemon_speed_boost = value,
                "pokemon_frozen" => weights.pokemon_frozen = value,
                "pokemon_asleep" => weights.pokemon_asleep = value,
                "pokemon_paralyzed" => weights.pokemon_paralyzed = value,
                "pokemon_toxic" => weights.pokemon_toxic = value,
                "pokemon_poisoned" => weights.pokemon_poisoned = value,
                "pokemon_burned" => weights.pokemon_burned = value,
                "leech_seed" => weights.leech_seed = value,
                "substitute" => weights.substitute = value,
                "confusion" => weights.confusion = value,
                "reflect" => weights.reflect = value,
                "light_screen" => weights.light_screen = value,
                "aurora_veil" => weights.aurora_veil = value,
                "safe_guard" => weights.safe_guard = value,
                "tailwind" => weights.tailwind = value,
                "healing_wish" => weights.healing_wish = value,
                "stealth_rock" => weights.stealth_rock = value,
                "spikes" => weights.spikes = value,
                "toxic_spikes" => weights.toxic_spikes = value,
                "sticky_web" => weights.sticky_web = value,
                "has_item" => weights.has_item = value,
                "sigmoid_scale" => weights.sigmoid_scale = value,
                "exploration" => weights.exploration = value,
                "playout_turns" => weights.playout_turns = value,
                "playout_policy" => weights.playout_policy = value,
                "playout_share" => weights.playout_share = value,
                "opponent_model" => weights.opponent_model = value,
                "opponent_boost_scale" => weights.opponent_boost_scale = value,
                "opponent_switch_penalty" => weights.opponent_switch_penalty = value,
                "own_switch_penalty" => weights.own_switch_penalty = value,
                other => return Err(format!("unknown evaluation weight: {}", other)),
            }
        }
        Ok(weights)
    }
}

static EVAL_WEIGHTS: OnceLock<EvalWeights> = OnceLock::new();

/// Replaces the evaluation weights. Only the first call has any effect, and it must come before
/// the first evaluation.
pub fn set_eval_weights(weights: EvalWeights) -> bool {
    EVAL_WEIGHTS.set(weights).is_ok()
}

pub fn eval_weights() -> &'static EvalWeights {
    EVAL_WEIGHTS.get_or_init(EvalWeights::default)
}

const POKEMON_BOOST_MULTIPLIER_6: f32 = 3.3;
const POKEMON_BOOST_MULTIPLIER_5: f32 = 3.15;
const POKEMON_BOOST_MULTIPLIER_4: f32 = 3.0;
const POKEMON_BOOST_MULTIPLIER_3: f32 = 2.5;
const POKEMON_BOOST_MULTIPLIER_2: f32 = 2.0;
const POKEMON_BOOST_MULTIPLIER_1: f32 = 1.0;
const POKEMON_BOOST_MULTIPLIER_0: f32 = 0.0;
const POKEMON_BOOST_MULTIPLIER_NEG_1: f32 = -1.0;
const POKEMON_BOOST_MULTIPLIER_NEG_2: f32 = -2.0;
const POKEMON_BOOST_MULTIPLIER_NEG_3: f32 = -2.5;
const POKEMON_BOOST_MULTIPLIER_NEG_4: f32 = -3.0;
const POKEMON_BOOST_MULTIPLIER_NEG_5: f32 = -3.15;
const POKEMON_BOOST_MULTIPLIER_NEG_6: f32 = -3.3;

fn evaluate_poison(pokemon: &Pokemon, base_score: f32) -> f32 {
    match pokemon.ability {
        Abilities::POISONHEAL => 15.0,
        Abilities::GUTS
        | Abilities::MARVELSCALE
        | Abilities::QUICKFEET
        | Abilities::TOXICBOOST
        | Abilities::MAGICGUARD => 10.0,
        _ => base_score,
    }
}

fn evaluate_burned(pokemon: &Pokemon, w: &EvalWeights) -> f32 {
    // burn is not as punishing in certain situations

    // guts, marvel scale, quick feet will result in a positive evaluation
    match pokemon.ability {
        Abilities::GUTS | Abilities::MARVELSCALE | Abilities::QUICKFEET => {
            return -2.0 * w.pokemon_burned
        }
        _ => {}
    }

    let mut multiplier = 0.0;
    for mv in pokemon.moves.into_iter() {
        if mv.choice.category == MoveCategory::Physical {
            multiplier += 1.0;
        }
    }

    // don't make burn as punishing for special attackers
    if pokemon.special_attack > pokemon.attack {
        multiplier /= 2.0;
    }

    multiplier * w.pokemon_burned
}

fn get_boost_multiplier(boost: i8) -> f32 {
    match boost {
        6 => POKEMON_BOOST_MULTIPLIER_6,
        5 => POKEMON_BOOST_MULTIPLIER_5,
        4 => POKEMON_BOOST_MULTIPLIER_4,
        3 => POKEMON_BOOST_MULTIPLIER_3,
        2 => POKEMON_BOOST_MULTIPLIER_2,
        1 => POKEMON_BOOST_MULTIPLIER_1,
        0 => POKEMON_BOOST_MULTIPLIER_0,
        -1 => POKEMON_BOOST_MULTIPLIER_NEG_1,
        -2 => POKEMON_BOOST_MULTIPLIER_NEG_2,
        -3 => POKEMON_BOOST_MULTIPLIER_NEG_3,
        -4 => POKEMON_BOOST_MULTIPLIER_NEG_4,
        -5 => POKEMON_BOOST_MULTIPLIER_NEG_5,
        -6 => POKEMON_BOOST_MULTIPLIER_NEG_6,
        _ => panic!("Invalid boost value: {}", boost),
    }
}

fn evaluate_hazards(pokemon: &Pokemon, side: &Side, w: &EvalWeights) -> f32 {
    let mut score = 0.0;
    let pkmn_is_grounded = pokemon.is_grounded();
    if pokemon.item != Items::HEAVYDUTYBOOTS {
        if pokemon.ability != Abilities::MAGICGUARD {
            score += side.side_conditions.stealth_rock as f32 * w.stealth_rock;
            if pkmn_is_grounded {
                score += side.side_conditions.spikes as f32 * w.spikes;
                score += side.side_conditions.toxic_spikes as f32 * w.toxic_spikes;
            }
        }
        if pkmn_is_grounded {
            score += side.side_conditions.sticky_web as f32 * w.sticky_web;
        }
    }

    score
}

fn evaluate_pokemon(pokemon: &Pokemon, w: &EvalWeights) -> f32 {
    let mut score = 0.0;
    score += w.pokemon_hp * pokemon.hp as f32 / pokemon.maxhp as f32;

    match pokemon.status {
        PokemonStatus::BURN => score += evaluate_burned(pokemon, w),
        PokemonStatus::FREEZE => score += w.pokemon_frozen,
        PokemonStatus::SLEEP => score += w.pokemon_asleep,
        PokemonStatus::PARALYZE => score += w.pokemon_paralyzed,
        PokemonStatus::TOXIC => score += evaluate_poison(pokemon, w.pokemon_toxic),
        PokemonStatus::POISON => score += evaluate_poison(pokemon, w.pokemon_poisoned),
        PokemonStatus::NONE => {}
    }

    if pokemon.item != Items::NONE {
        score += w.has_item;
    }

    // without this a low hp pokemon could get a negative score and incentivize the other side
    // to keep it alive
    if score < 0.0 {
        score = 0.0;
    }

    score += w.pokemon_alive;

    score
}

pub fn evaluate(state: &State) -> f32 {
    let w = eval_weights();
    let table = table_evaluate(state);
    // a learned model, if one has been set, takes the place of the table. It may build on the
    // table's score, and its chance of winning is put on the table's scale, so that the search
    // compares positions as before.
    if let Some(chance) = learned_win_chance(state, table) {
        let chance = chance.clamp(0.001, 0.999);
        return (chance / (1.0 - chance)).ln() / w.sigmoid_scale;
    }
    table
}

/// The hand-written table of material and conditions, from side one's point of view.
pub fn table_evaluate(state: &State) -> f32 {
    table_evaluate_with(state, eval_weights())
}

/// The table with the weights given, rather than those set for the search.
pub fn table_evaluate_with(state: &State, w: &EvalWeights) -> f32 {
    let mut score = 0.0;

    let mut iter = state.side_one.pokemon.into_iter();
    let mut s1_used_tera = false;
    while let Some(pkmn) = iter.next() {
        if pkmn.hp > 0 {
            score += evaluate_pokemon(pkmn, w);
            score += evaluate_hazards(pkmn, &state.side_one, w);
            if iter.pokemon_index == state.side_one.active_index {
                if state
                    .side_one
                    .volatile_statuses
                    .contains(&PokemonVolatileStatus::LEECHSEED)
                {
                    score += w.leech_seed;
                }
                if state
                    .side_one
                    .volatile_statuses
                    .contains(&PokemonVolatileStatus::SUBSTITUTE)
                {
                    score += w.substitute;
                }
                if state
                    .side_one
                    .volatile_statuses
                    .contains(&PokemonVolatileStatus::CONFUSION)
                {
                    score += w.confusion;
                }

                score += get_boost_multiplier(state.side_one.attack_boost) * w.pokemon_attack_boost;
                score +=
                    get_boost_multiplier(state.side_one.defense_boost) * w.pokemon_defense_boost;
                score += get_boost_multiplier(state.side_one.special_attack_boost)
                    * w.pokemon_special_attack_boost;
                score += get_boost_multiplier(state.side_one.special_defense_boost)
                    * w.pokemon_special_defense_boost;
                score += get_boost_multiplier(state.side_one.speed_boost) * w.pokemon_speed_boost;
            }
        }
        if pkmn.terastallized {
            s1_used_tera = true;
        }
    }
    if s1_used_tera {
        score += w.used_tera;
    }
    let mut iter = state.side_two.pokemon.into_iter();
    let mut s2_used_tera = false;
    while let Some(pkmn) = iter.next() {
        if pkmn.hp > 0 {
            score -= evaluate_pokemon(pkmn, w);
            score -= evaluate_hazards(pkmn, &state.side_two, w);

            if iter.pokemon_index == state.side_two.active_index {
                if state
                    .side_two
                    .volatile_statuses
                    .contains(&PokemonVolatileStatus::LEECHSEED)
                {
                    score -= w.leech_seed;
                }
                if state
                    .side_two
                    .volatile_statuses
                    .contains(&PokemonVolatileStatus::SUBSTITUTE)
                {
                    score -= w.substitute;
                }
                if state
                    .side_two
                    .volatile_statuses
                    .contains(&PokemonVolatileStatus::CONFUSION)
                {
                    score -= w.confusion;
                }

                // side two's boosts may count for more than side one's, for a search that should
                // fear an opponent setting up without itself setting up more
                let mut boosts =
                    get_boost_multiplier(state.side_two.attack_boost) * w.pokemon_attack_boost;
                boosts +=
                    get_boost_multiplier(state.side_two.defense_boost) * w.pokemon_defense_boost;
                boosts += get_boost_multiplier(state.side_two.special_attack_boost)
                    * w.pokemon_special_attack_boost;
                boosts += get_boost_multiplier(state.side_two.special_defense_boost)
                    * w.pokemon_special_defense_boost;
                boosts += get_boost_multiplier(state.side_two.speed_boost) * w.pokemon_speed_boost;
                score -= boosts * w.opponent_boost_scale;
            }
        }
        if pkmn.terastallized {
            s2_used_tera = true;
        }
    }
    if s2_used_tera {
        score -= w.used_tera;
    }

    score += state.side_one.side_conditions.reflect as f32 * w.reflect;
    score += state.side_one.side_conditions.light_screen as f32 * w.light_screen;
    score += state.side_one.side_conditions.aurora_veil as f32 * w.aurora_veil;
    score += state.side_one.side_conditions.safeguard as f32 * w.safe_guard;
    score += state.side_one.side_conditions.tailwind as f32 * w.tailwind;
    score += state.side_one.side_conditions.healing_wish as f32 * w.healing_wish;

    score -= state.side_two.side_conditions.reflect as f32 * w.reflect;
    score -= state.side_two.side_conditions.light_screen as f32 * w.light_screen;
    score -= state.side_two.side_conditions.aurora_veil as f32 * w.aurora_veil;
    score -= state.side_two.side_conditions.safeguard as f32 * w.safe_guard;
    score -= state.side_two.side_conditions.tailwind as f32 * w.tailwind;
    score -= state.side_two.side_conditions.healing_wish as f32 * w.healing_wish;

    score
}
