# HIGH NOON: Game mechanics

This describes the game as of commit `1cbbea0` ("grenade + face_layering"), including practice mode,
the scenery damage, the hat, the realistic facial damage, palm aiming and the grenade rebalance
(section 5.4), plus the removal of the day-to-dusk progression made on top of it.
Every number comes from `GameConfig` in `highnoon/config.py` or from a constant in the module named
next to it. Distances given as a fraction of the screen use the frame height `H` or width `W` in the
mirrored camera frame.

## 1. Overview

HIGH NOON is a two-player duel played in front of one webcam. Each player:

- aims by pointing a palm,
- fires by closing the aiming hand into a fist,
- ducks behind a wall that covers the bottom 25% of the screen to dodge shots and to reload.

The first player to reach 0 HP loses the round. Wins are counted per player for the whole session.

There is also a solo **practice mode** (`--practice`, section 9) for testing the controls: one
player shoots bottles instead of an opponent.

## 2. Game flow (`highnoon/game/game.py`)

```
idle ──(2 players visible)──► countdown (3 s) ──► duel ──(someone hits 0 HP)──► K.O. + winner screen (5 s)
  ▲                               │                                                    │
  └──(a player lost)──────────────┘                         countdown (3 s) ◄──────────┘
```

| Mode | What happens |
|---|---|
| **idle** | Fewer than two players are visible. The screen shows "TWO PLAYERS / STEP INTO VIEW". |
| **countdown** | 3-2-1 (`countdown_s = 3.0`), with a click sound per number. Nobody can fire. If a player leaves view, the game goes back to idle. When it ends, the round is reset, the announcer says "FIGHT!" (voice and on-screen text) and the duel starts. |
| **duel** | The round is live. Shots, damage, powerups and the camping penalty all apply. The mode does not change when a player leaves. |
| **round over** | Starts when a player's HP reaches 0. "K.O.!" is announced (1.3 s, with voice), then the winner screen ("PLAYER n WINS THE ROUND", "NEXT ROUND IN n") is shown. The whole phase lasts `round_end_s = 5.0` s from the moment of the K.O., then a new countdown begins. Nobody can deal damage, and the duck, reload and status text is hidden. |

- **A player who leaves during a duel:** nothing is paused and there is no timer. A player who has been
  out of view for more than `come_back_after_s = 5.0` s gets "P*n*: COME BACK ON SCREEN" in their
  colour. The game keeps running. Once they are visible again the message disappears.
- **Round reset:** HP goes back to 100 and ammo to 3. Rockets, grenades, bullets in flight and the
  falling powerup are all removed. Scores are kept.
- **`r` key:** resets the round immediately without changing the mode.

## 3. Players and identity (`highnoon/tracking/identity.py`)

- There are two player slots, **Player 1** and **Player 2**, each with its own colour.
- Each frame, detected bodies are matched to slots by the distance between body centres, using
  Hungarian assignment (`linear_sum_assignment`). A body can only be matched to a slot if it moved
  less than `max_match_distance = 0.25 W` since the last frame.
- New bodies take free slots in order from left to right.
- A body is followed by position only while it stays in view (gaps up to `track_gap_s = 5` s, e.g.
  ducking). A player who was out of view longer gets their slot back by face (below).
- **Enrolment:** until the duel starts, a slot is kept for `lost_timeout_s = 5` s after its player
  leaves (they can come back to it by face), then released, and whoever claims it next starts fresh. Meanwhile each slot collects its player's
  face embeddings (OpenCV SFace, 128-d, from a face aligned on the face mesh's eyes, nose and mouth;
  only faces turned less than about 30° with eyes at least 18 px apart).
- **Locked ids:** once the duel starts, slots are never released. A player who leaves keeps their slot
  and gets it back only when a new body's face matches theirs (mean cosine similarity `≥ 0.36` over
  their latest 2–3 embeddings, seen for `0.15` s) and beats every other player's by `0.1`. Until then the body gets no
  id, so a third person never gets one, even when a player is away. Identity is decided by face only:
  someone whose face cannot be seen waits without an id until it can.
- **Face check** (before and during the duel): a body tracked by position whose face keeps failing to match the slot's owner
  (below `0.36`) for `0.3` s is released. This catches a stranger stepping into the spot a player just left,
  and ids swapped by two players crossing (both are released, then re-identified by face).
- One more body than there are players is detected, so a bystander is seen and refused rather than
  hiding a player from the pose detector.
- Faces are matched to the nearest nose, within one face size. Hands are matched to the nearest pose
  wrist, within `max(60 px, 0.8 × shoulder width)`.

## 4. Controls

### 4.1 Aiming (`highnoon/vision/motion.py`)

- **Pointer:** the aim runs from the wrist to the middle of the four knuckles, measured from the hand
  landmarks. The elbow and shoulder play no part, so turning the hand is enough. It uses the knuckles
  rather than the fingertips because they move far less as the fingers close into a fist.
- **Requirements:** the hand must be tracked, the palm must look at least `0.08` torso lengths long
  on screen (a shorter one is pointing at the camera), and the aim must point between **−50° and
  +80°** of elevation. Outside that range the player is not aiming.
- **Tracking dropouts:** if the hand is lost or points at the camera, the last aim is held for
  `0.3` s. After that the player is not aiming until the hand is back.
- **Choice of hand:** the right hand is the default. The game switches to the left hand only while
  the right hand cannot be used for aiming.
- **Smoothing:** the aim angle goes through a One Euro filter (`min_cutoff 0.8`, `beta 0.6`). The
  shot starts at the wrist.
- **Held weapon:** while aiming, the weapon the player would fire (revolver, rocket launcher or
  grenade) is drawn in their hand, pointing along the aim and outlined in their colour. There is no
  laser line. With a rocket or grenade loaded, a dotted arc with a reticle shows where it will land.
  The shot leaves the muzzle, not the wrist: `gun_reach = 0.27 H`, `rocket_reach = 0.38 H`,
  `grenade_reach = 0.1 H` from the wrist along the aim. The weapon is drawn starting
  `hand_reach = 0.06 H` from the wrist.
- **Hands tracked:** up to `max_hands = 4` hands are detected, so both players' hands are found.

### 4.2 Firing (`highnoon/game/actions.py`)

**Hand openness** = mean distance from the four fingertips to the wrist ÷ palm length (wrist to
middle knuckle). It is measured from MediaPipe's 3D hand landmarks. The hand state has hysteresis:

- **fist** when openness < `fist_closed = 1.25`
- **open** when openness > `fist_open = 1.45`

A shot fires at the moment the aiming hand changes from open to fist, but only if all of these hold.
If one fails, the fist is ignored and the reason is shown in the debug overlay and in the signal log.

| Check | Value | Reason shown if it fails |
|---|---|---|
| The aim is valid | Section 4.1 | `not aiming` |
| The player is not ducked | — | `ducked` |
| The player has aimed for at least | `armed_after_s = 0.3` s | `aim settling` |
| Time since the last shot is at least | `fire_cooldown_s = 0.6` s | `cooldown` |

With a bullet weapon and an empty magazine the fist is a **dry fire**: a click sound is played and
`no ammo` appears in the debug overlay.

**Stable aim:** the shot uses the oldest aim sample from the last `aim_lookback_s = 0.1` s. This
cancels the small hand movement that comes with closing the fist.

**Recoil spread:**

- Each shot is rotated by a random angle between −spread and +spread, using the spread before that
  shot.
- Each shot then adds `recoil_per_shot_deg = 10°`, up to a maximum of `recoil_max_deg = 20°`.
- The spread recovers at about 16.7° per second, so it is back to 0 about `recoil_recovery_s = 1.2` s
  after the last shot.
- The first shot after a pause is therefore exactly where you aim. Quick follow-up shots spread out.
  The reticle is a fixed-size crosshair; the spread is not drawn.

### 4.3 Ducking and cover

- **The wall:** a static rectangle over the bottom `cover_height = 25%` of the screen. It stops
  bullets and rockets.
- **Head position:** the centre of the face box, or the nose keypoint if no face is detected.
- **Ducked:** the head goes below the top of the wall (the line at 75% of the screen height).
- **Standing up:** the head must rise `duck_margin = 1.5% H` above that line. This hysteresis stops
  the state from flickering.
- **While ducked:** the player cannot fire, and their head and body count as cover, so any bullet or
  rocket that reaches them is blocked. Rocket splash still hurts them (section 5.3).

### 4.4 Reloading

- Stay ducked for `reload_s = 0.8` s to refill the magazine to `max_ammo = 3`.
- This happens once per duck. To reload again, stand up and duck again.
- A "RELOADING" bar shows the progress, and a reload sound plays when the magazine is refilled
  (only if it was not already full).
- When a player's magazine is empty and they hold no powerup, a "DUCK TO RELOAD!" warning appears on
  that player's side of the screen (25% or 75% of the width).

### 4.5 Camping penalty

- **GET UP!** is shown after `camp_warning_s = 5` s ducked.
- After `camp_grace_s = 7` s ducked, **GET OUT CAMPER!** is shown, and the player loses
  `5% of max HP = 5 HP` every `camp_drain_interval_s = 1` s until they stand up. The first loss
  happens right at 7 s.
- The drain only applies during a live duel while an opponent is visible. It counts as damage from
  the opponent, so camping down to 0 HP gives the opponent the round.

## 5. Weapons and damage

The game starts at `max_hp = 100`. When a player fires, the weapon is chosen in this order:

1. **grenade**, if they hold one and an opponent is visible
2. **rocket**, if they hold one
3. **bullet**, if they have ammo
4. otherwise a **dry fire** (a bullet weapon with an empty magazine)

### 5.1 Hitboxes (`hitbox()` in `game.py`)

- **Head:** a circle at the face-box centre, with radius `0.6 × the face box's longer side`. Without
  a face it is a circle at the nose, with radius `0.35 × torso length`.
- **Body:** the quadrilateral formed by the shoulders and hips, scaled up by `1.15` around its
  centre.

### 5.2 Bullets

- **Damage:** **body 10**, **head 25**. From full HP that is 10 body hits or 4 headshots.
- **Origin:** the muzzle (`gun_reach` from the wrist, along the aim). The bullet is drawn as a copper
  bullet with a faint streak.
- **Travel time:** a bullet arrives `bullet_travel_s = 0.6` s after the shot. When it arrives, the
  ray is cast again from the original muzzle position, in the original direction, against the **current**
  positions. A player who ducks or moves away during that time dodges it.
- **What it hits:** whatever the ray meets first:
  - the wall: "BLOCKED"
  - a ducked player: also cover, "BLOCKED"
  - a head: "HEADSHOT!" in the shooter's colour
  - a body
  - the powerup: the shooter collects it
- **Bounds:** the ray stops at the edge of the screen.

### 5.3 Rocket launcher (powerup)

- **Flight:** the rocket follows a ballistic arc. It launches at `1.6 H/s` in the aim direction,
  falls with gravity `1.3 H/s²`, and expires after 4 s or when it leaves the screen horizontally.
- **Impact:** it explodes at the first thing it touches: the wall, a player, or the powerup. Hitting
  the powerup also collects it.
- **Blast:** an explosion (with sound) of radius `rocket_blast_radius = 0.16 H`. Damage falls off linearly from
  `50` at the centre to `20` at the edge, measured to the nearest point of the head or body.
- **Over the wall:** the blast reaches past the wall, so it can damage a ducked player. It never
  damages the shooter.
- **Ammo:** one powerup gives one rocket. Rockets do not stack, and the bullet magazine is not used.

### 5.4 Grenade (powerup)

- **Throw:** the grenade is lobbed from `grenade_reach` past the wrist to where the player aims: the
  first point on the aim line that meets the wall or the opponent, or the edge of the screen if it
  meets neither. A dotted arc and a reticle preview the landing spot. It flies for
  `grenade_flight_s = 1.4` s with an arc peaking at `0.45 H`. The spot is fixed when it is thrown, so
  the grenade does not follow anyone.
- **Warning:** while it flies, the landing spot flashes "DUCK!" with a ring showing the blast radius,
  or shows "SAFE" while the target is ducked.
- **Blast:** radius `grenade_blast_radius = 0.5 H`, which is wider than the fireball drawn for it.
  Damage falls off linearly from `grenade_damage = 80` at the centre to `grenade_min_damage = 10` at
  the edge, measured to the nearest point of the head or body. A player still on the landing spot takes the full 80 wherever it
  lands on them, which leaves 20 HP from full health and kills anyone at 80 HP or less.
- **Avoiding it:** a ducked player takes no damage, and a ducked target shows "SAFE!". Moving away
  during the flight lowers the damage, and getting clear of the radius avoids it. A player out of view
  when it lands takes no damage.
- **Splash on the thrower:** the blast also hurts the thrower if they are standing within its radius,
  so a grenade aimed nearby is risky. They can duck before it lands to avoid this. The damage counts
  as coming from the opponent, so a thrower who dies this way loses the round. If the blast would kill
  both players, the target dies first and the thrower wins.

### 5.5 Powerups

- **When they drop:** only during a live duel. The first one comes 15 s after the duel starts
  (`powerup_interval_s = 15`), then one every 15 s. A new one never drops while another is still on
  screen.
- **Type:** a **GRENADE** (green) with chance `grenade_drop_chance = 25%`, otherwise a **ROCKET
  LAUNCHER** (orange). The pickup shows only its sprite, with no text label.
- **Fall:** each one drops straight down from just above the screen, starting at a random horizontal
  position between 45% and 55% of the width. It falls at constant speed for `powerup_fall_s = 5` s
  until it reaches the top of the wall and disappears.
- **Size:** the radius is `0.08 H`, and the hit radius is `1.3×` larger than the drawn sprite.
- **Collecting:** the first player to shoot it with a bullet, or touch it with a rocket, gets it. The
  pickup announcement is shown in that player's colour. A player holds at most one rocket and one
  grenade.

## 6. HUD, sound and feedback (`highnoon/render/`)

- **No banner:** there is no top title or hint bar. Only the idle screen shows "TWO PLAYERS / STEP
  INTO VIEW".
- **Per player:**
  - a name tag that follows the player
  - an animated HP bar
  - ammo pips
  - the reload bar and the camping warnings
  - the weapon in their hand (section 4.1)
  - the duck, reload and status text is hidden while the round-over screen is shown
- **Announcements:** large text that zooms in and shrinks out. They are queued (at most two waiting,
  and the current one is shortened if another is waiting): "FIGHT!" (1.0 s), "K.O.!" (1.3 s) and
  the powerup pickup announcements in the player's colour.
- **Effects:**
  - muzzle flash
  - floating damage numbers
  - BLOCKED and SAFE!
  - explosions with particles and screen shake
  - the scenery, facial damage and hat described in sections 6b and 6c
- **Sound** (`render/sound.py`, `--mute` turns it off): recorded effects for the revolver, rocket
  launch, grenade throw, explosion, reload, dry-fire click and the 3-2-1 click, plus announcer
  voices for "FIGHT!" and "K.O.". Credits are in `assets/CREDITS.md`.
- **Drawing rate:** the game is drawn at `render_hz = 60` even on a 30 fps camera. Between camera
  frames the game clock is advanced (by at most 0.1 s past the last frame) and the scene is redrawn,
  so projectiles and effects move smoothly. Perception still runs once per camera frame.
- **Webcam inset:** a small view of the raw, mirrored camera picture (`14%` of the screen height) sits
  on the wall, centred between the two player cards, so the input can be compared with the game's
  output. It updates once per camera frame and is hidden while the debug overlay is on, because the
  stats panel uses the same spot.
- **Debug overlay (`d`):** the skeleton, the live signal values against their thresholds (palm length,
  elevation, hand openness, aim, head position), and the reason for the last ignored fist.

Keys: `q`/`esc` quit, `r` restart the round, `d` debug overlay, `b` toggle the background,
`f` fullscreen. The `--mute` flag turns the sound off.

## 6a. Evaluation log (`highnoon/core/match_log.py`)

One row is appended to `results/games.csv` (`--games-csv PATH`, or `--no-games-csv` to disable) when
a round ends in a K.O., or when the program quits during a live round. Each row has the result,
winner, duration, fps, latency/inference/render timings (mean and p95), and per-player stats: shots,
bullet hits, headshots, accuracy, rockets, grenades, damage dealt, camping damage taken, blocked
shots, powerups, reloads, dry fires, ducks, time ducked, time aiming, the share of aiming time with a
tracked hand, fists (with how many fell in cooldown and the share that fired) and the counts of
ignored fists by reason. Practice mode does not write this log.

## 6b. Scenery (`render/scenery.py`)

These effects only change the picture; none of them affect the rules.

- **Sunset:** during a duel the sky moves from midday towards sunset, and never gets to night. How far
  it goes is the larger of (time since the round started ÷ `SUNSET_AFTER_S = 60` s) and (the HP lost
  by the lowest player ÷ max HP). It never jumps: the sky eases towards that at `0.15` per second, so a
  25 HP hit fades in over about 1.7 s and full sunset takes about 7 s at least. It is a keyframe
  animation: five skies from midday to sunset are drawn once, and the current one is a cross-fade of
  the two nearest, in 120 small steps. The sun sinks and reddens, the land and the wall warm up.
  Outside the duel and at the start of the next round it eases back to midday the same way.
- **Wall damage:** a bullet that is BLOCKED by the wall leaves a bullet hole with splinters where it
  hit. An explosion scorches the wall, and one near the top edge also burns a charred notch into
  it. A shattered bottle scatters glass on the wall's top rail. All of it is cleared when a new
  round starts.
- **Ground damage:** an explosion scorches the desert behind the players and leaves a crater in the
  sand. This only shows with the background replacement on (`b`). It is also cleared each round.

## 6c. Face effects (`render/face_damage.py`, `render/face_fx.py`)

Face tracking uses Google's **MediaPipe Face Landmarker** (`models/face_landmarker.task`, float16),
not a custom model. It runs in VIDEO (tracking) mode for up to `max_players` faces and gives 478
points per face: 468 for the face mesh and 10 for the irises. The expression (blendshape) output is
not enabled.

The effects stop at the top of the wall. The face is still tracked behind it, but a ducked player's
hat and injuries are hidden, and a face partly behind the wall only shows them above the wall's top
edge. Weapons, projectiles and the HUD are drawn over the face effects.

### Facial degradation

The face picks up realistic injuries as HP drops. The look depends only on current HP, so it builds
up from any kind of damage (bullets, rocket splash, grenades, camping, bombs) and is gone when HP is
reset for the next round.

| HP | Injuries |
|---|---|
| 100–76 | none |
| 75–51 | a scraped cheek on one side, and grime on the jaw and lower cheek on the other |
| 50–26 | adds a bruise under one eye (dark centre, purple, then a yellow-green rim) and a cut with swollen skin on that cheek |
| 25–0 | adds a black eye on the other side, a nosebleed running over the lips to the chin, a split lower lip and a pale, drained complexion (20% towards gray) |

- **Warped onto the mesh:** the injuries are painted once into a flat texture of the face, in the UV
  layout of MediaPipe's canonical face model (`assets/face_mesh.npz`, built by
  `tools/build_face_mesh.py`). Each of the 852 mesh triangles maps its part of the texture onto the
  tracked landmarks, so the injuries bend, stretch and turn with the face. Eyes and mouth are holes
  in the mesh, so they are never painted over.
- **Blended into the skin:** the texture multiplies the camera image instead of being drawn over it,
  so the player's own skin detail and lighting show through. The marks fade out towards the edge of
  the mesh, so no outline shows.
- **Hit flash:** all damage also reddens the face, including each camping drain tick. Blue and green
  drop by up to 18% and recover over `0.25` s (the same time as the HUD card flash).
- **Turning away:** triangles facing away from the camera are not drawn, and the nose covers the far
  cheek, so injuries on the far side disappear as the head turns.
- **Sides:** Player 1's bruise is on the left of the image and Player 2's on the right (the same
  texture mirrored), so the two players look different.
- **Skipped when:** no face is detected, or the face is less than 24 px wide.
- **Cost:** the warp runs at a face width of at most 110 px and is scaled up. A new camera frame
  takes about 2.5–5 ms per face, and a redraw between camera frames about 0.3–2.7 ms, because the
  warp is reused (measured on a photo with faces 119 and 238 px wide). The textures are painted when
  the renderer starts, so the first hit does not stall a frame.

### Hat

The landmarks are also used for a **cowboy hat** in the player's colour, drawn on top of the
facial damage.

- **Where:** it is drawn for every visible player whose face is detected, in every mode (idle,
  countdown, duel, practice). It sits on the top edge of the face box, `1.1 ×` the face width
  wide, and is drawn at 90% opacity under the name tag and HP bar.
- **Tilt:** the hat is rotated by the head roll, measured as the angle of the line between the two
  iris centres (landmarks 468 and 473).
- **No effect on play:** the hat adds nothing to the hitbox. Its brim and lower crown lie inside the
  head circle (section 5.1), so a shot there is a headshot. The top of the crown reaches about
  `0.19 ×` the face height above the face box and the circle only about `0.1 ×`, so a shot through
  only the top of the crown misses.
- **When it is missing:** no face detected (the head hitbox then uses the nose), or a face box
  smaller than 8 px.

The facial degradation and the hat are the only effects drawn from face landmarks. The other
effects around a player's head or body (the name tag and HP bar, the red flash of the HUD card,
`-N` damage numbers, HEADSHOT! and the camping drain) come from the face box or the pose. Like the facial degradation, most of them need the player to take damage (see section 9 for
why that only happens through bombs in practice mode).

## 7. Tunables at a glance (`GameConfig`)

| Group | Parameter | Default |
|---|---|---|
| Trigger | `fist_closed` / `fist_open` | 1.25 / 1.45 |
| | `armed_after_s`, `fire_cooldown_s`, `aim_lookback_s` | 0.3, 0.6, 0.1 s |
| Recoil | `recoil_per_shot_deg`, `recoil_max_deg`, `recoil_recovery_s` | 10°, 20°, 1.2 s |
| Cover | `cover_height`, `duck_margin` | 0.25, 0.015 |
| Ammo | `max_ammo`, `reload_s` | 3, 0.8 s |
| Camping | `camp_warning_s`, `camp_grace_s`, `camp_drain_fraction`, `camp_drain_interval_s` | 5 s, 7 s, 0.05, 1 s |
| Damage | `max_hp`, `body_damage`, `head_damage`, `bullet_travel_s` | 100, 10, 25, 0.6 s |
| Flow | `countdown_s`, `round_end_s`, `come_back_after_s` | 3, 5, 5 s |
| Weapon reach | `hand_reach`, `gun_reach`, `rocket_reach`, `grenade_reach` | 0.06, 0.27, 0.38, 0.1 (x H) |
| Display | `render_hz` | 60 |
| Powerups | `powerup_interval_s`, `powerup_fall_s`, `powerup_radius`, `powerup_hit_scale`, `powerup_center_spread` | 15 s, 5 s, 0.08, 1.3, 0.05 |
| Rocket | `rocket_speed`, `rocket_gravity`, `rocket_blast_radius`, `rocket_damage`, `rocket_min_damage` | 1.6, 1.3, 0.16, 50, 20 |
| Grenade | `grenade_flight_s`, `grenade_arc`, `grenade_blast_radius`, `grenade_damage`, `grenade_min_damage`, `grenade_drop_chance` | 1.4 s, 0.45, 0.5, 80, 10, 0.25 |
| Practice | `practice_powerup_interval_s`, `bomb_damage` | 10 s, 25 |
| Bottles | `bottle_count`, `bottle_radius`, `bottle_respawn_s` | 4, 0.06 (x H), 0.6 s |

## 8. Behaviour worth knowing

These follow from the code but are easy to miss:

- **Camping can lose the round.** The camping drain counts as damage from the opponent, so a player
  who camps down to 0 HP hands the opponent the win.
- **Aiming needs a tracked hand.** A player whose hand stays lost for more than `0.3` s stops aiming,
  however their arm is held, and must then aim for `armed_after_s` again before they can fire. While
  the hand is lost a fist cannot be seen, so a fist made then only counts when the hand reappears
  (the shot fires at that moment, if the checks in section 4.2 pass).
- **Grenades beat rockets.** A player holding both throws the grenade first, as long as an opponent
  is visible.
- **The reload is free.** Ducking refills the magazine even when it is not empty.
- **No hit-check at the trigger.** Whether a bullet hits is decided only when it arrives. Aiming at
  the target when you fire does not guarantee a hit.
- **Leaving does not pause the game.** The absent player is only asked to come back; the round keeps
  running.
- **Re-tracking resets a player.** If a slot is lost for more than 1.5 s and claimed again, that
  player restarts with full HP and ammo, even in the middle of a round.

## 9. Practice mode (`--practice`)

A solo target range for trying the controls. Run it with
`python -m highnoon --practice`. It uses the same aiming, firing, ducking, reloading and recoil as
the duel; only the targets change.

- **Setup:** only one person is detected (`max_players = 1`), and the games CSV is not written.
- **Flow:** the idle screen says "PRACTICE / STEP INTO VIEW". One visible player starts the 3-2-1
  countdown, then "FIGHT!" and the live round, exactly as in the duel.
- **Bottles:** up to `bottle_count = 4` green bottles stand in the air above the wall, at random
  positions between 8% and 92% of the width and 12% and 90% of the wall height, at least three
  bottle radii apart. Each has radius `0.06 H`. A bullet or a rocket blast that reaches one
  shatters it (green glass, "+1", glass on the wall). Missing bottles reappear together
  `bottle_respawn_s = 0.6` s after the last one broke.
- **Score:** each bottle is worth one point, shown as "n BOTTLES" on the player card. The count
  carries over between rounds.
- **Powerups:** every `10` s, either a **ROCKET LAUNCHER** or a **BOMB** (red) drops, using the same
  fall as in the duel. Grenades are never dropped, and could not be thrown anyway because there is
  no opponent.
- **Bombs:** shooting the bomb, or hitting it with a rocket, sets it off where it was hit and costs
  the shooter `bomb_damage = 25` HP. The bomb's own explosion does not break bottles (a rocket's
  does). The fourth bomb takes the player to 0 HP: the round ends with "BLOWN UP / DON'T SHOOT THE
  BOMBS", no win is scored, and a new countdown starts after 5 s.
- **What is switched off:** the player is never a target. Bullets and rocket blasts pass through
  them, there is no opponent for the camping drain (the GET UP! warnings still appear), and there
  are no grenades.

**Why damage effects only come from bombs in practice mode.** Practice mode removes the player
from every hit test (`self.cfg.practice` checks in `_hit_fractions` and `_explode` in `game.py`),
and `_opponent_of()` returns `None`. So no bullet, rocket, grenade or camping drain can hurt them.
The player effects that react to damage only appear after the player shoots bombs. These are the
red face and HUD-card flash, the `-N` numbers on the body and the facial degradation. Each bomb
(25 HP) moves the face exactly one degradation stage. HEADSHOT! never appears in practice. The hat does not depend on damage, so it is always drawn. The scenery
effects (bullet holes, scorch marks, glass on the rail) do appear in practice whenever the player
hits the wall, fires a rocket or breaks a bottle.
