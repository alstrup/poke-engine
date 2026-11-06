// Signals that describe a position for a learned evaluation, after the way backgammon programs
// feed a small network with hand-made features next to the raw position.
// The same signals are listed for the side whose view is taken and then for the other side,
// followed by the state of the field.
use super::damage_calc::{calculate_damage, type_effectiveness_modifier, DamageRolls};
use super::generate_instructions::get_effective_speed;
use super::items::Items;
use super::state::{PokemonVolatileStatus, Terrain, Weather};
use crate::choices::{Choices, MoveCategory, MoveTarget};
use crate::state::{Pokemon, PokemonStatus, Side, SideReference, State};

pub const SIDE_FEATURES: usize = 62;
pub const FIELD_FEATURES: usize = 9;
pub const NUM_FEATURES: usize = 2 * SIDE_FEATURES + FIELD_FEATURES;

fn flag(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}

fn has_move(pokemon: &Pokemon, test: impl Fn(&crate::state::Move) -> bool) -> bool {
    pokemon
        .moves
        .into_iter()
        .any(|mv| mv.id != Choices::NONE && test(mv))
}

fn is_recovery(mv: &crate::state::Move) -> bool {
    match &mv.choice.heal {
        Some(heal) => heal.target == MoveTarget::User && heal.amount > 0.0,
        None => false,
    }
}

fn is_setup(mv: &crate::state::Move) -> bool {
    if mv.choice.category != MoveCategory::Status {
        return false;
    }
    match &mv.choice.boost {
        Some(boost) => {
            boost.target == MoveTarget::User
                && (boost.boosts.attack > 0
                    || boost.boosts.special_attack > 0
                    || boost.boosts.speed > 0)
        }
        None => false,
    }
}

fn is_priority(mv: &crate::state::Move) -> bool {
    mv.choice.priority > 0 && mv.choice.base_power > 0.0
}

fn removes_hazards(mv: &crate::state::Move) -> bool {
    matches!(
        mv.id,
        Choices::RAPIDSPIN
            | Choices::DEFOG
            | Choices::MORTALSPIN
            | Choices::TIDYUP
            | Choices::COURTCHANGE
    )
}

/// The most damage the active pokemon of `side_ref` does to the other active pokemon with one move,
/// allowing for accuracy, and whether it has a damaging move with priority that knocks out.
fn best_damage(state: &State, side_ref: &SideReference, target_hp: i16) -> (f32, bool) {
    let side = state.get_side_immutable(side_ref);
    let mut best: f32 = 0.0;
    let mut priority_knockout = false;
    for mv in side.get_active_immutable().moves.into_iter() {
        if mv.id == Choices::NONE || mv.disabled || mv.pp <= 0 {
            continue;
        }
        if let Some((damage, _)) =
            calculate_damage(state, side_ref, &mv.choice, DamageRolls::Average)
        {
            let expected = damage as f32 * (mv.choice.accuracy / 100.0).min(1.0);
            if expected > best {
                best = expected;
            }
            if mv.choice.priority > 0 && damage >= target_hp {
                priority_knockout = true;
            }
        }
    }
    (best, priority_knockout)
}

/// Roughly what share of the defender's full health the attacker's best move takes, from stats and
/// types alone. Unlike the exact calculation it works for pokemon that are not on the field, so it
/// can say who beats whom across the two teams.
fn rough_damage(attacker: &Pokemon, defender: &Pokemon) -> f32 {
    let mut best: f32 = 0.0;
    for mv in attacker.moves.into_iter() {
        let choice = &mv.choice;
        if mv.id == Choices::NONE || choice.base_power <= 0.0 || mv.pp <= 0 {
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
        damage *= type_effectiveness_modifier(&choice.move_type, defender);
        damage *= match attacker.item {
            Items::CHOICEBAND if choice.category == MoveCategory::Physical => 1.5,
            Items::CHOICESPECS if choice.category == MoveCategory::Special => 1.5,
            Items::LIFEORB => 1.3,
            _ => 1.0,
        };
        // the average roll, and the chance of missing
        damage *= 0.925 * (choice.accuracy / 100.0).min(1.0);
        best = best.max(damage / defender.maxhp.max(1) as f32);
    }
    best
}

fn rough_speed(pokemon: &Pokemon) -> f32 {
    let mut speed = pokemon.speed as f32;
    if pokemon.item == Items::CHOICESCARF {
        speed *= 1.5;
    }
    if pokemon.status == PokemonStatus::PARALYZE {
        speed *= 0.5;
    }
    speed
}

/// The rough damage of every pokemon of each side on every pokemon of the other, worked out once
/// for a position: `one_on_two[i][j]` is side one's pokemon i hitting side two's pokemon j, and
/// `two_on_one[j][i]` the hit coming back.
struct Matchups {
    one_on_two: [[f32; 6]; 6],
    two_on_one: [[f32; 6]; 6],
}

fn matchups(state: &State) -> Matchups {
    let mut result = Matchups {
        one_on_two: [[0.0; 6]; 6],
        two_on_one: [[0.0; 6]; 6],
    };
    for (i, mine) in state.side_one.pokemon.pkmn.iter().enumerate() {
        if mine.hp <= 0 {
            continue;
        }
        for (j, theirs) in state.side_two.pokemon.pkmn.iter().enumerate() {
            if theirs.hp <= 0 {
                continue;
            }
            result.one_on_two[i][j] = rough_damage(mine, theirs);
            result.two_on_one[j][i] = rough_damage(theirs, mine);
        }
    }
    result
}

/// Who beats whom one on one, from where each pokemon's health stands, in seven signals.
/// `hit[me][them]` is the damage this side does and `taken[them][me]` what comes back.
fn matchup_features(
    side: &Side,
    other: &Side,
    hit_table: &[[f32; 6]; 6],
    taken_table: &[[f32; 6]; 6],
    out: &mut [f32],
) {
    let my_active = side.active_index as usize;
    let their_active = other.active_index as usize;
    let mut mine_alive = 0;
    let mut theirs_alive = 0;
    let mut won_pairs = 0.0;
    let mut answers = 0.0;
    let mut active_wins = 0.0;
    let mut active_beats = 0.0;
    let mut best_sweep: f32 = 0.0;
    let mut useless = 0.0;
    let mut best_hit: f32 = 0.0;
    for (i, me) in side.pokemon.pkmn.iter().enumerate() {
        if me.hp <= 0 {
            continue;
        }
        mine_alive += 1;
        theirs_alive = 0;
        let my_health = me.hp as f32 / me.maxhp.max(1) as f32;
        let my_speed = rough_speed(me);
        let mut beaten = 0.0;
        for (j, them) in other.pokemon.pkmn.iter().enumerate() {
            if them.hp <= 0 {
                continue;
            }
            theirs_alive += 1;
            let hit = hit_table[i][j];
            let taken = taken_table[j][i];
            let their_health = them.hp as f32 / them.maxhp.max(1) as f32;
            let my_turns = if hit > 0.0 {
                (their_health / hit).ceil()
            } else {
                f32::MAX
            };
            let their_turns = if taken > 0.0 {
                (my_health / taken).ceil()
            } else {
                f32::MAX
            };
            let wins = my_turns < their_turns
                || (my_turns == their_turns && my_turns < f32::MAX && my_speed > rough_speed(them));
            if wins {
                won_pairs += 1.0;
                beaten += 1.0;
                if j == their_active {
                    answers += 1.0;
                }
                if i == my_active {
                    active_beats += 1.0;
                    if j == their_active {
                        active_wins = 1.0;
                    }
                }
            }
            if j == their_active {
                best_hit = best_hit.max(hit);
            }
        }
        best_sweep = best_sweep.max(beaten);
        if beaten == 0.0 {
            useless += 1.0;
        }
    }
    let pairs = (mine_alive * theirs_alive).max(1) as f32;
    out[0] = won_pairs / pairs;
    out[1] = answers / 6.0;
    out[2] = active_wins;
    out[3] = best_sweep / 6.0;
    out[4] = useless / 6.0;
    out[5] = best_hit.min(1.0);
    out[6] = active_beats / 6.0;
}

fn side_features(state: &State, side_ref: &SideReference, table: &Matchups, out: &mut [f32]) {
    let (side, other): (&Side, &Side) = state.get_both_sides_immutable(side_ref);
    let active = side.get_active_immutable();
    let other_active = other.get_active_immutable();
    let active_index = side.active_index as usize;

    let mut alive = 0.0;
    let mut health = 0.0;
    let mut statused = 0.0;
    let mut boots = 0.0;
    let mut recovery = 0.0;
    let mut setup = 0.0;
    let mut priority = 0.0;
    let mut pivot = 0.0;
    let mut removal = false;
    let mut used_tera = false;
    let mut bench: [f32; 5] = [0.0; 5];
    let mut bench_count = 0;
    let mut bench_faster = 0.0;
    let other_speed = get_effective_speed(state, &side_ref.get_other_side()) as f32;
    for (index, pokemon) in side.pokemon.pkmn.iter().enumerate() {
        if pokemon.terastallized {
            used_tera = true;
        }
        if pokemon.hp <= 0 {
            continue;
        }
        let share = pokemon.hp as f32 / pokemon.maxhp.max(1) as f32;
        alive += 1.0;
        health += share;
        if pokemon.status != PokemonStatus::NONE {
            statused += 1.0;
        }
        if pokemon.item == Items::HEAVYDUTYBOOTS {
            boots += 1.0;
        }
        if has_move(pokemon, is_recovery) {
            recovery += 1.0;
        }
        if has_move(pokemon, is_setup) {
            setup += 1.0;
        }
        if has_move(pokemon, is_priority) {
            priority += 1.0;
        }
        if has_move(pokemon, |mv| mv.choice.flags.pivot) {
            pivot += 1.0;
        }
        if has_move(pokemon, removes_hazards) {
            removal = true;
        }
        if index != active_index {
            if bench_count < 5 {
                bench[bench_count] = share;
                bench_count += 1;
            }
            if other_active.hp > 0 && pokemon.speed as f32 > other_speed {
                bench_faster += 1.0;
            }
        }
    }
    // the bench has no order of its own, so the healthiest comes first
    bench.sort_by(|a, b| b.total_cmp(a));

    let active_alive = active.hp > 0;
    let my_speed = get_effective_speed(state, side_ref) as f32;
    let (damage, priority_knockout) = if active_alive && other_active.hp > 0 {
        best_damage(state, side_ref, other_active.hp)
    } else {
        (0.0, false)
    };
    let mut outsped = 0.0;
    for pokemon in other.pokemon.pkmn.iter() {
        if pokemon.hp > 0 && active_alive && my_speed > pokemon.speed as f32 {
            outsped += 1.0;
        }
    }
    let faster = if !active_alive || other_active.hp <= 0 {
        0.5
    } else if my_speed > other_speed {
        1.0
    } else if my_speed < other_speed {
        0.0
    } else {
        0.5
    };
    let volatile = |status: PokemonVolatileStatus| flag(side.volatile_statuses.contains(&status));
    let conditions = &side.side_conditions;
    let boosted = [
        PokemonVolatileStatus::PROTOSYNTHESISATK,
        PokemonVolatileStatus::PROTOSYNTHESISDEF,
        PokemonVolatileStatus::PROTOSYNTHESISSPA,
        PokemonVolatileStatus::PROTOSYNTHESISSPD,
        PokemonVolatileStatus::PROTOSYNTHESISSPE,
        PokemonVolatileStatus::QUARKDRIVEATK,
        PokemonVolatileStatus::QUARKDRIVEDEF,
        PokemonVolatileStatus::QUARKDRIVESPA,
        PokemonVolatileStatus::QUARKDRIVESPD,
        PokemonVolatileStatus::QUARKDRIVESPE,
    ]
    .iter()
    .any(|status| side.volatile_statuses.contains(status));

    const BASIC: usize = 55;
    let values: [f32; BASIC] = [
        alive / 6.0,
        health / 6.0,
        if active_alive {
            active.hp as f32 / active.maxhp.max(1) as f32
        } else {
            0.0
        },
        bench[0],
        bench[1],
        bench[2],
        bench[3],
        bench[4],
        statused / 6.0,
        flag(active_alive && matches!(active.status, PokemonStatus::SLEEP | PokemonStatus::FREEZE)),
        flag(active_alive && active.status == PokemonStatus::PARALYZE),
        flag(active_alive && active.status == PokemonStatus::BURN),
        flag(active_alive && active.status == PokemonStatus::POISON),
        flag(active_alive && active.status == PokemonStatus::TOXIC),
        flag(!used_tera),
        flag(active.terastallized),
        side.attack_boost as f32 / 6.0,
        side.defense_boost as f32 / 6.0,
        side.special_attack_boost as f32 / 6.0,
        side.special_defense_boost as f32 / 6.0,
        side.speed_boost as f32 / 6.0,
        volatile(PokemonVolatileStatus::SUBSTITUTE),
        volatile(PokemonVolatileStatus::LEECHSEED),
        volatile(PokemonVolatileStatus::CONFUSION),
        volatile(PokemonVolatileStatus::TAUNT),
        volatile(PokemonVolatileStatus::ENCORE),
        flag(conditions.stealth_rock > 0),
        conditions.spikes as f32 / 3.0,
        conditions.toxic_spikes as f32 / 2.0,
        flag(conditions.sticky_web > 0),
        (conditions.reflect as f32 / 8.0).min(1.0),
        (conditions.light_screen as f32 / 8.0).min(1.0),
        (conditions.aurora_veil as f32 / 8.0).min(1.0),
        (conditions.tailwind as f32 / 4.0).min(1.0),
        boots / 6.0,
        flag(removal),
        flag(active_alive && has_move(active, is_recovery)),
        recovery / 6.0,
        setup / 6.0,
        priority / 6.0,
        pivot / 6.0,
        flag(side.wish.0 > 0),
        flag(side.future_sight.0 > 0),
        faster,
        if other_active.hp > 0 {
            (damage / other_active.hp as f32).min(1.0)
        } else {
            0.0
        },
        (damage / other_active.maxhp.max(1) as f32).min(1.0),
        flag(
            active_alive
                && other_active.hp > 0
                && (priority_knockout || (damage >= other_active.hp as f32 && faster >= 1.0)),
        ),
        flag(active_alive && has_move(active, is_priority)),
        outsped / 6.0,
        bench_faster / 6.0,
        flag(active_alive && active.item != Items::NONE),
        flag(active_alive && matches!(active.item, Items::LEFTOVERS | Items::BLACKSLUDGE)),
        flag(
            active_alive
                && matches!(
                    active.item,
                    Items::CHOICEBAND | Items::CHOICESPECS | Items::CHOICESCARF
                ),
        ),
        flag(boosted),
        (conditions.toxic_count as f32 / 8.0).min(1.0),
    ];
    out[..BASIC].copy_from_slice(&values);
    let (hit, taken) = match side_ref {
        SideReference::SideOne => (&table.one_on_two, &table.two_on_one),
        SideReference::SideTwo => (&table.two_on_one, &table.one_on_two),
    };
    matchup_features(side, other, hit, taken, &mut out[BASIC..SIDE_FEATURES]);
}

/// Fills `out` with the signals of a position as seen from `perspective`.
pub fn features(state: &State, perspective: &SideReference, out: &mut [f32; NUM_FEATURES]) {
    let table = matchups(state);
    side_features(state, perspective, &table, &mut out[..SIDE_FEATURES]);
    side_features(
        state,
        &perspective.get_other_side(),
        &table,
        &mut out[SIDE_FEATURES..2 * SIDE_FEATURES],
    );
    let field = &mut out[2 * SIDE_FEATURES..];
    let weather = state.weather.weather_type;
    field[0] = flag(matches!(weather, Weather::SUN | Weather::HARSHSUN));
    field[1] = flag(matches!(weather, Weather::RAIN | Weather::HEAVYRAIN));
    field[2] = flag(weather == Weather::SAND);
    field[3] = flag(matches!(weather, Weather::SNOW | Weather::HAIL));
    let terrain = state.terrain.terrain_type;
    field[4] = flag(terrain == Terrain::ELECTRICTERRAIN);
    field[5] = flag(terrain == Terrain::GRASSYTERRAIN);
    field[6] = flag(terrain == Terrain::PSYCHICTERRAIN);
    field[7] = flag(terrain == Terrain::MISTYTERRAIN);
    field[8] = flag(state.trick_room.active);
}
