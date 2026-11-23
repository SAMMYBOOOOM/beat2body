# Physical plausibility in 2D

`beat2body/physics.py`; `python reports/2026-11-22_physics/physics_check.py` writes `physics.json`.

The camera is fixed and level, so image "down" is gravity. All motion is in body units (torso lengths, 30 fps). The
same contact rule (ankle speed) is used for real and generated motion.

## Measures

- **Centre of mass (CoM):** segment-weighted from the COCO-17 joints (Winter's mass fractions).
- **Jump (flight):** neither ankle moving slowly, both ankles at least 0.15 torso lengths above the local floor
  (the lowest planted ankle within ±0.5 s), for at least 6 frames, with the CoM rising then falling. Per jump, a
  parabola fitted to the CoM gives its vertical acceleration (should be g) and horizontal acceleration (should be 0).
- **Floor penetration:** how far a swinging ankle goes below the local floor.
- **Off-balance:** when standing, the share of frames whose CoM is horizontally outside the planted feet (± 0.25).

**Depth caveat.** A foot swinging toward the camera projects lower, so it reads as penetration; balance flattens
the support onto one image axis. These two are only meaningful against real dancers, not as training losses.
Gravity in flight is robust: it uses the whole-body CoM over a short jump.

## Calibration: real dancers fall at g

g ≈ 9.81 m/s² / 0.5 m (torso) / 30² = **0.0218** torso lengths per frame².

| Jump detector, all 210 advanced videos | Jumps | Share of frames | CoM fall rate |
|---|---:|---:|---:|
| Loose: both ankles above the floor at all | 6,564 | 37% | 0.0028 (catches toe stance, shuffles) |
| **Used:** ≥ 0.15 above the floor, no slow ankle, rise then fall | 1,061 | 4.3% | **0.0179** |
| + DWPose heel and toe both moving (real data only) | 795 | 3.2% | 0.0197 |
| + ≥ 0.3 above the floor | 289 | 1.3% | 0.0207 |

The stricter the detector, the closer real jumps come to g. The check is valid in 2D.

## Real vs models: the 10 held-out test songs (20 s each)

| | Jumps (share of frames) | CoM fall rate in flight (g ≈ 0.0218) | Horizontal acc. in flight | Penetration (mean / > 0.05) | Off-balance |
|---|---:|---:|---:|---:|---:|
| Real | 32 (4.1%) | **0.0203** | 0.0014 | 0.004 / 1.7% | 45% |
| 2.1b | 19 (3.2%) | 0.0107 | 0.0046 | 0.069 / 17.2% | 46% |
| 2.1b.1 | 22 (5.1%) | 0.0100 | 0.0022 | 0.028 / 9.6% | 46% |
| 2.1b.2 | 34 (4.9%) | 0.0092 | 0.0022 | 0.025 / 10.9% | 35% |
| 4.1 | 12 (1.8%) | 0.0125 | 0.0025 | 0.007 / 2.8% | 24% |
| 4.4 | 10 (1.4%) | 0.0140 | 0.0013 | 0.0006 / 0.4% | 13% |

- **Generated jumps are floaty.** In the air, the models' CoM falls at 42–64% of the real rate: they hang in the
  air, like a jump on the Moon. Real dancers fall at g. 4.4 is closest (64%) but still far off. No loss constrains
  this yet; it is the clearest physics target.
- **The foot fix removed the "sinking feet".** 2.1b's swinging feet dip below the floor in 17% of swing frames
  (real 1.7%); 4.1 2.8%, 4.4 0.4%.
- **Horizontal drift in the air:** 2.1b accelerates sideways in flight 3× more than real; 4.4 is at real.
- **The 4.x models jump less** (1.4–1.8% of frames vs 4.1% real) and are off-balance less often than real dancers
  (13–24% vs 45%). Real dance is dynamic (the CoM often leaves the support). Read with the depth caveat.
- Jump counts on 10 songs are small (10–34); the real-dancer calibration uses 1,061 jumps.

## Next: a gravity loss

In the model's own flight phases (both ankles off the floor and moving), penalize the CoM's vertical acceleration
for deviating from g, and its horizontal acceleration for deviating from 0. The flight mask comes from the generated
motion and is not differentiated. This is one physical term instead of a hand-made rule. Before adding it as a loss,
recheck the calibration of g per dancer (torso length in metres varies by dancer).
