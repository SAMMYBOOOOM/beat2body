# Plan: better foot-contact labels and predictions

Background: [SLIDING_ISSUE.md](SLIDING_ISSUE.md). Goal: the training penalty uses reliable contact labels from the
real dancer, the model predicts contact well, and foot locking at inference uses those predictions softly.

## Stage A: align the existing ankle-speed labels (done, 2026-11-20)

- `contact[t]` = the ankle moved less than 0.012 torso lengths over the step t−1 → t. The first label foot loss
  applied it to the step t → t+1, so it was one step late: it penalized every lift-off and missed every touchdown.
  That is 12.4% of observed ankle steps on the 10 test videos, and 1,302 lift-offs.
- `--foot-contact label` now uses `contact[t+1]` for the step t → t+1, with the ankle observed at both ends.
  `label-prev` keeps the old version, used by runs 4.1 and 4.2.
- Foot locking uses the same convention: the foot lands at frame a−1 of a contact starting at a. A fully planted
  foot alone drives the root correction, so a fading foot no longer drags it.
- Tests: the touchdown step is penalized and the lift-off step is not (`label-prev`: the reverse). The real motion
  scores zero foot loss. Locking a real stance leaves it still.
- Run **4.3** = 4.2 with the aligned labels (jobs `v43-*`).
- New evaluation measures: stance drift, contact duration, contact share, swing speed.

## Stage B: heel / toe stance labels

**Data.** DWPose outputs 6 foot points (big toe, small toe, heel, per foot) with confidence. The training skeleton
stays COCO-17. The raw files exist for 1,200 of 1,410 videos. The 210 advanced (sFM) videos lost theirs after
cleaning, so DWPose is re-run on them (`aws pose`; source videos in `2d/raw_video/`).

**Labels** (`2d/features/contact/<video>.npz`, new prep step `python -m beat2body contact`):
- Use the same image → body-unit transform as the cleaned keypoints (pelvis-centred, divided by the dancer's
  torso), so the speeds are comparable with `CONTACT_THRESHOLD`.
- Per foot and step t → t+1: heel speed and toe speed (the mean of the two toe points), each only where the point's
  confidence is ≥ 0.6 at both ends (to be calibrated on the confidence distribution).
- `heel[t]`, `toe[t]` = planted if speed < threshold. Stance per foot =
  - `flat` (heel and toe planted), `heel` (heel only), `toe` (toe only: a pivot on the ball of the foot),
    `swing` (neither);
  - `unknown` where a needed point is low-confidence, or the foot points disagree with the ankle by more than a
    margin (a likely detection error).
- Hysteresis: a planted phase must last ≥ 2 steps, and single-step gaps are filled, so detector jitter does not
  flip labels.
- A sanity report per genre: stance shares, agreement with the ankle-speed labels, and the unknown share. Check
  10 clips by eye against the video.

**Masks.** Training ignores `unknown` steps for the contact head and the skating penalty (they count as neither
planted nor swinging).

## Stage C: a separate contact head

- **Model:** a small head on the decoder's last hidden state → per foot, logits for heel-planted and toe-planted
  (4 outputs). It is separate from the 38 motion features, so the diffusion target does not change. The two motion
  contact channels stay, for compatibility and the existing foot lock.
- **Loss:** binary cross-entropy against the stage-B labels, masked by confidence, at low noise only (t < 500, the
  same reason as the foot loss). Weight about 0.1, tuned so it is a few percent of the total loss.
- **Penalty gating stays with the reference labels:** the skating loss uses the real dancer's labels, never the
  head's output, so the model cannot lower the penalty by predicting fewer contacts. With stage-B labels: penalize
  ankle motion on `flat` and `heel` steps. On `toe` steps, penalize only translation of the projected toe, which is
  approximated from the ankle and the shin direction because COCO-17 has no toe. `unknown` steps are not penalized.
- **Augmentation:** a left/right flip swaps the feet's labels; time-stretch resamples them (nearest).
- **Sampling:** at the final denoising step, the head outputs contact probabilities for the generated motion.

## Stage D: soft locking driven by the predicted scores

- Lock strength per foot and frame = the head's probability (heel or flat → ankle anchor; toe → softer, rotation
  allowed), instead of a hard 0.5 threshold.
- **Sequence solve instead of greedy accumulation:** the root path minimizes
  |root − original|² + λ_acc·|root acceleration|² + Σ p·|ankle − anchor|²,
  a small banded least-squares problem per sequence. Leg IK takes the residual, and its strength is reduced when
  the anchor is out of reach. This replaces "smooth the root, then accumulate".
- This addresses the current over-planting: on 2.1b.2 with locking, the slower ankle moves 0.002 vs 0.008 for real.
  Real planted ankles are not perfectly still, partly because of pose-estimation jitter.

## Evaluation (every stage)

Same seed, songs and windows for all models. Each run is evaluated raw, and with the current and the soft lock.
Measures:
- FID_k and FID_p;
- on-beat movement;
- PFC;
- planted-foot speed and the share of frames with no foot planted;
- root jitter;
- stance drift;
- contact duration and share;
- swing speed (feet must still move between contacts);
- for stage C, the head's accuracy and calibration on real windows (labels vs predictions).

Beat timing is checked separately with the beat-shift probe.

## Order and status

| Step | What | Status |
|---|---|---|
| A | aligned labels, foot-lock convention, tests, run 4.3 | done; 4.3 queued |
| A-eval | 2.1b.2 / 4.2 / 4.2-allt (and 4.1, 4.3), raw and locked | running |
| B1 | re-run DWPose on the 210 advanced videos | next |
| B2 | stance labels + masks + sanity report | after B1 |
| C | contact head, BCE loss, label-gated penalty with stage-B labels, run 4.4 | after B2 |
| D | soft sequence lock from the head's scores | with C |
