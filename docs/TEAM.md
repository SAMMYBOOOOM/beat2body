# Team runbook: evaluations and analysis

How to help run the per-genre evaluations of the beat2body dance models and the analysis afterwards. No local GPU
is needed: evaluations run on Google Colab, and the results go to the Hugging Face repo
[SizheZ03/beat2body-runs](https://huggingface.co/SizheZ03/beat2body-runs) (private).

Background: [MODELS.md](MODELS.md) lists every model. The current base model is **6.1**. The candidates compared
now are 4.1, 4.4b, 6.1, 6.2 and 6.3.

## One-time setup

1. **Hugging Face access.** You need write access to `SizheZ03/beat2body-runs`: ask Sizhe. Then create a
   token at <https://huggingface.co/settings/tokens> with write access to that repo.
2. **Colab.** Open the team notebook: **[open in Colab](https://colab.research.google.com/github/novaz03/beat2body/blob/claude/relaxed-knuth-fc6tqf/notebooks/beat2body_team.ipynb)**
   ([`notebooks/beat2body_team.ipynb`](../notebooks/beat2body_team.ipynb)). It has one cell per task below. Use
   File → Save a copy in Drive to keep your own copy.
   - Runtime → Change runtime type → **GPU** (T4 works; L4 or A100 is faster).
   - Key icon on the left → add a secret named `HF_TOKEN` with your token, and allow notebook access.
   - **Never paste a token into a notebook cell or a file**: notebooks get shared and committed.

## Task A: run the evaluation queue

1. In the notebook, set `WORKER = "yourname"` in the **Setup** cell and run it.
2. Run the **Task A** cell. It fetches the latest [`scripts/colab_eval.py`](../scripts/colab_eval.py) from GitHub
   each time, so you never run an old copy. The first time, allow Google Drive access: the data (~150 MB) and
   checkpoints are cached there, so later runs start faster.

What it does: it reads the queue `evals/queue.json`, skips finished entries, and **claims** each entry before
running it. Several people can run the cell at the same time, and each entry is run once. A claim older than 3 hours
(a closed session) is taken over by the next person. Each evaluation takes about 6–8 minutes on an L4. The results
are uploaded as `runs/<job>/eval_test_<tag>.json` and `.log`.

The output, in order:
- `worker: yourname-xxxx`;
- `N queued evaluations, M to run`;
- for each entry: `-- 6.1 feet2_raw_genre`, then either `... is running it: skipped` or the evaluation's
  progress, with `[evaluate] generating test windows ...` and `long dance i/10`;
- at the end: `finished: ... uploaded ...; ... FAILED ...`.

A failed evaluation uploads its log as `eval_test_<tag>.failed.log`. Send Sizhe the name of the entry.

Then the video queue runs, if there is one. The videos come out as `reports/<folder>/videos/*.mp4` on Hugging Face.

## Task B: per-genre analysis (after Task A; no GPU)

The notebook's **Task B** cell does the steps below and shows the tables. To run them on your own machine instead:

The question: which genres are hardest for the models (is ballet jazz harder?), and where does each model fail?

```bash
pip install huggingface_hub           # in a clone of the repo; no GPU or torch needed
hf auth login                         # or: export HF_TOKEN=...
python scripts/summarize_genres.py --out reports/2026-11-24_genres/README.md
```

This writes:
- a **genre difficulty** table: each model's FID divided by its genre's real reference, so 1 means as close as
  real dance;
- one table per genre, with every model next to the real values.

How to read the tables:
- **Do not compare raw FIDs across genres.** With 50–120 windows per genre, FID is biased upwards by different
  amounts. Compare a model with its genre's reference (the ratio).
- Correctness measures should be close to real:
  - stance drift: a planted foot sliding;
  - jump fall rate: real jumps fall at about 0.019;
  - height drift: body proportions changing over a 20 s dance;
  - root jitter.
- Style measures (number of jumps, how often no foot is planted, speed) may differ from real without being errors.
- All measures are on the 2D image. A dancer who turns away from the camera or moves upstage looks smaller, and that
  lowers on-screen sliding and jump measures without the motion being better.

## Task C: video review (no GPU)

The notebook's **Task C** cell plays one genre's video in the browser: set `GENRE` first. To download them instead:

The videos are in `reports/2026-11-23_v6/videos/` on Hugging Face. There is one per test song (genre), 20 s with
music: the real dancer, then 6.1, 6.2 and 6.3. The panels flash yellow on beats.

```bash
hf download SizheZ03/beat2body-runs --include "reports/2026-11-23_v6/videos/*" --local-dir .
```

For each video, note per model, with timestamps:
- **feet sliding** while a foot should be planted;
- **body collapse or tangles**: limbs crossing, the body folding flat;
- **shrinking or turning away**: the figure getting small or narrow (6.3 seems to do this; how often, and in
  which genres?);
- **on the beat?** Big moves on the yellow flashes?
- **genre fit**: does it look like the genre?

Write the notes in `reports/2026-11-24_genres/REVIEW.md`, one section per genre. Short is fine. Timestamps are
the most useful part.

## Who does what

Sizhe queues the work: new entries in `evals/queue.json` and `evals/videos.json`, and training on AWS. Pick tasks
from this table and put your name in the first column so two people don't do the same thing.

| Who | Task | Status |
|---|---|---|
| | A: per-genre evaluations of 6.1, 6.2, 6.3, 4.4b, 4.1 (`feet2_raw_genre`) | queued |
| | B: per-genre analysis → `reports/2026-11-24_genres/README.md` | after A |
| | C: video review, genres ballet jazz, break, house, krump, la hiphop | ready |
| | C: video review, genres lock, middle hiphop, pop, street jazz, waack | ready |
