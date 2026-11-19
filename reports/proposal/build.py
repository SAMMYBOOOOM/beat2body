"""Build the Beat2Body project proposal PDF (ELEC/COMP 576)."""
import os
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "Beat2Body_Proposal.pdf")
W = letter[0] - 1.4 * inch

base = ParagraphStyle("base", fontName="Times-Roman", fontSize=10, leading=12.2, alignment=TA_JUSTIFY,
                      spaceAfter=3.5)
small = ParagraphStyle("small", parent=base, fontSize=8.6, leading=10.2, spaceAfter=0)
cap = ParagraphStyle("cap", parent=base, fontSize=8.6, leading=10.2, alignment=TA_JUSTIFY, spaceBefore=2,
                     spaceAfter=6, textColor=colors.HexColor("#222222"))
h1 = ParagraphStyle("h1", parent=base, fontName="Times-Bold", fontSize=11.5, leading=13.5, spaceBefore=5,
                    spaceAfter=2.5, alignment=0)
h2 = ParagraphStyle("h2", parent=base, fontName="Times-Bold", fontSize=10, leading=12, spaceBefore=3,
                    spaceAfter=1.5, alignment=0)
title = ParagraphStyle("title", parent=base, fontName="Times-Bold", fontSize=17, leading=20, alignment=TA_CENTER,
                       spaceAfter=3)
sub = ParagraphStyle("sub", parent=base, fontSize=10.5, leading=13, alignment=TA_CENTER, spaceAfter=1)
cell = ParagraphStyle("cell", parent=base, fontSize=8.3, leading=9.6, alignment=0, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Times-Bold")
ref = ParagraphStyle("ref", parent=base, fontSize=8.2, leading=9.5, spaceAfter=0.8, leftIndent=14,
                     firstLineIndent=-14, alignment=0)
bul = ParagraphStyle("bul", parent=base, leftIndent=11, firstLineIndent=-8, spaceAfter=1.5)

S = []


def P(t, st=base):
    S.append(Paragraph(t, st))


def B(t):
    S.append(Paragraph("&bull;&nbsp;" + t, bul))


def fig(path, width, caption):
    from PIL import Image as PI
    w, h = PI.open(path).size
    S.append(KeepTogether([Image(path, width=width, height=width * h / w), Paragraph(caption, cap)]))


def table(rows, widths, header=True, fs=None):
    data = [[Paragraph(str(c), cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa4ae")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


# ----------------------------------------------------------------------------- title
P("Beat2Body: Music- and Genre-Conditioned 2D Dance Generation", title)
P("ELEC / COMP 576: Introduction to Deep Learning &mdash; Final Project Proposal", sub)
P("YinHsuan Hsu &nbsp;&middot;&nbsp; Yi Xin &nbsp;&middot;&nbsp; Sizhe Zhang "
  "&nbsp;&nbsp;|&nbsp;&nbsp; November 19, 2026", sub)
S.append(Spacer(1, 4))

# ----------------------------------------------------------------------------- 1
P("1. Abstract", h1)
P("Beat2Body generates a dancer's 2D skeleton sequence from a music clip and a chosen dance genre. We build "
  "a 2D dataset from the AIST Dance Video Database [1] (1,410 front-camera videos, 10 genres, 60 songs, "
  "5.2 h) by running three pose estimators (DWPose [6], OpenPose [7], MediaPipe [8]) and aligning their "
  "COCO-17 skeletons with beat-aware music features. Our main model is a 2D adaptation of EDGE [4], a "
  "transformer diffusion model with music cross-attention and classifier-free guidance, compared against "
  "an autoregressive GRU baseline. We add a projected bone-length loss that removes body-size drift, a "
  "frequency-band metric that measures movement on the beat, and we compare the three pose tools as data "
  "sources. A first 7.1M-parameter model already produces stable, realistic poses (FID<sub rise='1.8' size='7'>p</sub> 23.1); "
  "its main weakness, too little movement on the beat, is the focus of the remaining work, after which the "
  "same model is applied to 3D motion from AIST++ [2].")

# ----------------------------------------------------------------------------- 2
P("2. Background and Motivation", h1)
P("Dance is one of the most natural responses to music, yet creating choreography is hard: even professional "
  "dancers run out of ideas when improvising to unfamiliar music, and no one is an expert in every genre. "
  "Games, animation and virtual performers need large amounts of natural dance motion, which is expensive to "
  "record with motion capture. A model that turns any music clip into a plausible dance of a chosen genre "
  "would give choreographers a source of ideas and give animators motion at almost no cost.")
P("We represent the dancer as a <i>skeleton</i>: 17 body joints (nose, shoulders, elbows, wrists, hips, knees, "
  "ankles, ...) whose 2D image positions are tracked in every video frame by a pose-estimation network "
  "(Fig. 1). The task is then sequence generation: given per-frame music features and a genre label, output "
  "the joint positions frame by frame. A good result must (i) look like real dancing, (ii) move on the beat, "
  "(iii) match the genre, and (iv) stay physically plausible over long pieces (no foot sliding, no body that "
  "grows or shrinks).")
P("<b>Prior work.</b> Lee <i>et al.</i> [3] generated 2D dance from web videos with OpenPose for 3 genres "
  "using a GAN-based decomposition of dance into movement units. Later work moved to 3D motion capture: "
  "FACT [2] (cross-modal transformer, AIST++), Bailando [5] (VQ-VAE + actor-critic GPT) and EDGE [4] "
  "(diffusion), all trained on AIST++, which required a 9-camera studio setup to reconstruct. "
  "<b>Gap.</b> 3D methods cannot use the vast number of ordinary single-camera dance videos; 2D generation "
  "from off-the-shelf pose estimators scales to such data but is noisier, depends on the camera's "
  "perspective, and has not been studied with modern diffusion models or across pose estimators. "
  "<b>Our extensions:</b> 10 genres instead of 3 [3]; the shared COCO-17 format [16] for three pose tools, "
  "so we can measure how the choice of tool affects the generated dance; a diffusion model [4, 9] adapted to "
  "2D with explicit beat phase and genre conditioning; and 2D-specific constraints and metrics. After the 2D "
  "stage, the same model is trained on AIST++ 3D data [2] for a direct comparison with published results.")

fig(os.path.join(HERE, "fig1_pose.png"), W,
    "<b>Figure 1.</b> From video to training data: (a) a frame of an AIST break-dance video, (b) the DWPose "
    "skeleton (COCO-17) found on it, (c) the cleaned skeleton the model learns to generate.")

# ----------------------------------------------------------------------------- 3
P("3. Proposed Experimentation and Implementation", h1)
P("3.1 Dataset (completed for 2D)", h2)
P("<b>Source.</b> The AIST Dance Video Database [1] (free for academic research) contains 13,940 videos. We "
  "use all single-dancer videos from the front camera (c01) in 10 genres (Break, Pop, Lock, Waack, Middle "
  "Hip-hop, LA Hip-hop, House, Krump, Street Jazz, Ballet Jazz): 210 <i>advanced</i> dances (each dancer's own "
  "choreography, about 35 s) and 1,200 <i>basic</i> dances (10 genre-basic routines per dancer, each danced to "
  "4 songs, 7&ndash;12 s), 5.2 h in total, on 60 songs at 80&ndash;135 BPM; the basic dances were added "
  "after our first experiments showed overfitting. <b>Processing.</b> Videos are resampled from 59.94 to "
  "30 fps. DWPose (134 points), MediaPipe (33) and OpenPose (25) are mapped to COCO-17 with tool-specific "
  "confidence thresholds (0.3 / 0.5 / 0.1); outlier jumps are removed, short gaps interpolated and jitter "
  "smoothed with a Savitzky&ndash;Golay filter. Each frame is centred on the mid-hip and scaled by the "
  "dancer's torso length. Music features (librosa [13]): EDGE's 35 per-frame features (onset envelope, "
  "20 MFCC, 12 chroma, onset peaks, beats) plus the phase within the beat. Because librosa's beat tracker "
  "picked the wrong tempo for 10 of the 60 songs, beats are placed on a grid at AIST's published tempo; an "
  "audit command checks labels, lengths, beats and splits against the raw sources. <b>Split.</b> One song per "
  "genre is held out entirely as the test set; the rest is split into train / validation by video (basic "
  "dances: by routine, so a routine's tempo versions never cross splits). Clips are 8 s windows with a 2 s "
  "stride; the model trains on 5 s (150-frame) windows. The processed advanced-dance data and the code are already public (Hugging Face "
  "datasets <font face='Courier' size=8>SizheZ03/AISTDance_*</font>).")
S.append(table([
    ["Split", "Videos (advanced + basic)", "Hours", "Songs", "Use"],
    ["Train", "145 + 868 = 1,013", "3.7", "50", "fit the model"],
    ["Validation", "29 + 132 = 161", "0.6", "50 (shared with train)", "model selection, early stopping"],
    ["Test", "36 + 200 = 236", "0.8", "10 (never seen in training)", "final numbers only"],
], [0.85 * inch, 1.75 * inch, 0.55 * inch, 1.75 * inch, W - 4.9 * inch]))
S.append(Spacer(1, 5))

P("3.2 Design Requirements", h2)
B("<b>Temporal continuity:</b> 15&ndash;30 s dances without pose collapse or body-size drift (variation of "
  "torso length and body height over the piece within 15% of a real dancer's).")
B("<b>Beat alignment:</b> movement peaks within 50 ms (1.5 frames) of musical beats; as much movement in the "
  "beat band (1&ndash;4 Hz) as real dancers.")
B("<b>Realism, diversity, genre:</b> low FID on kinetic and pose features, diversity close to real dance, "
  "genre recognizable by a classifier trained on real data.")
B("<b>Efficiency:</b> each training run fits on one GPU in a few hours (AWS SageMaker g5/g6, Google Colab).")

P("3.3 Architecture", h2)
P("Each frame is a 38-dimensional vector: 2 ankle-contact flags, the 2D root (hip) position in the image, "
  "and 17 pelvis-centred joints (Fig. 2). <b>Baseline:</b> an autoregressive GRU that predicts the next pose "
  "from the previous pose, a 1D-CNN music encoding and a 32-dimensional learned genre embedding. "
  "<b>Main model:</b> a 2D re-implementation of EDGE [4]. A diffusion model [9] learns to undo noise added "
  "to whole 5 s motion sequences; its denoiser is an 8-layer transformer decoder [14] (width 512, 46.9M "
  "parameters; a 7.1M version for fast experiments) that attends to a music encoder through cross-attention "
  "and is conditioned on the noise level and genre. Music and genre are dropped 25% of the time so that "
  "classifier-free guidance [11] can strengthen them at sampling time; sampling uses 50 DDIM steps [10], and "
  "long dances are built from overlapping windows kept consistent while denoising. EDGE's losses "
  "(reconstruction, velocity, joint positions, foot contact) are kept; for 2D we add a <i>projected "
  "bone-length loss</i> (each bone's on-screen length must match the same bone in the real frame, so natural "
  "foreshortening is kept) and test an <i>energy loss</i> on each joint's speed and range, v-prediction [15] "
  "and Min-SNR weighting [12].")
fig(os.path.join(HERE, "fig2_method.png"), W,
    "<b>Figure 2.</b> (a) Data pipeline from AIST videos to per-frame motion and music features. (b) The "
    "diffusion transformer: it denoises a 5 s motion sequence conditioned on music (cross-attention) and "
    "genre; at test time it starts from pure noise.")

P("<b>Evaluation.</b> Metrics are computed on 5 s windows (validation split during development, the held-out "
  "test songs for final numbers) against the real dancers on the same music. Following AIST++ [2] and Bailando [5], FID<sub rise='1.8' size='7'>k</sub> "
  "and FID<sub rise='1.8' size='7'>p</sub> are Fr&eacute;chet distances between generated and real "
  "<i>kinetic</i> (joint speed and acceleration) and <i>pose</i> statistics, and Div<sub rise='1.8' size='7'>k</sub> / Div<sub rise='1.8' size='7'>p</sub> measure variety between generated dances; "
  "BAS [2] scores how close movement minima fall to musical beats; PFC [4] scores foot sliding. We add three "
  "2D-specific measures: <i>speed</i> and <i>BeatPow</i> (motion power at 1&ndash;4 Hz) relative to real, "
  "to catch damped dancing, and a <i>drift check</i> (variation of torso, thigh and body height over a "
  "20 s dance) to catch a body that grows or shrinks. Genre accuracy uses a classifier trained on real "
  "skeletons. The GRU baseline and the diffusion model are compared on the same split and windows.")

_ms = Paragraph("3.4 Milestones and Measures of Success", h2)
_tab = table([
    ["Stage", "Time", "What we will do", "How we know it works", "Lead"],
    ["1. Data", "Done (2D); Wk 1 (3D)",
     "2D dataset with 3 pose tools, audio features, beat grid, clips; add basic dances. Download AIST++ 3D.",
     "Audit passes; clips render back to the source video with poses and beats aligned.", "YinHsuan Hsu"],
    ["2. Baseline", "Wk 1&ndash;2", "Train the autoregressive GRU on music + genre.",
     "Val loss decreases; 5 s of motion that follows the beat.", "Sizhe Zhang"],
    ["3. Main model", "Wk 2&ndash;3",
     "Diffusion transformer, 2D losses, larger model and data; long-form generation.",
     "Speed and beat-band power near real over 15&ndash;30 s; smooth transitions; better FID/BAS than "
     "baseline.", "YinHsuan Hsu, Sizhe Zhang"],
    ["4. Evaluation", "Wk 3", "FID<sub rise='1.8' size='7'>k</sub>/FID<sub rise='1.8' size='7'>p</sub>, diversity, BAS, foot-skating (PFC), BeatPow, "
     "drift; compare the 3 pose tools.", "Main model beats baseline; pose tools ranked.", "Yi Xin"],
    ["5. 3D, demo, report", "Wk 4", "Train on AIST++ 3D; stick-figure videos; report and poster.",
     "Finished report, poster and video demo.", "Yi Xin &amp; all"],
], [0.85 * inch, 0.8 * inch, 2.05 * inch, 1.95 * inch, W - 5.65 * inch])
S.append(KeepTogether([_ms, _tab]))
S.append(Spacer(1, 5))

P("3.5 Preliminary Results", h2)
P("The 7.1M diffusion model trained for 20k steps (on the advanced dances only) gives realistic poses and a "
  "stable body, but too little rhythmic movement (validation set, 440 windows):")
S.append(table([
    ["", "FID<sub rise='1.8' size='7'>k</sub> &darr;", "FID<sub rise='1.8' size='7'>p</sub> &darr;", "Foot skating (PFC) &darr;",
     "Speed (real = 1)", "Beat-band power (real = 1)", "Torso drift, 20 s (real 0.103)"],
    ["First model (no size fix)", "52.8", "55.5", "0.048", "0.58", "&ndash;", "0.134"],
    ["+ per-video scale, bone-length loss", "43.7", "23.1", "0.038", "0.53", "0.26", "0.111"],
], [1.95 * inch, 0.55 * inch, 0.55 * inch, 0.95 * inch, 0.75 * inch, 0.95 * inch, W - 5.7 * inch]))
S.append(Spacer(1, 4))
P("A frequency analysis (Fig. 3b) shows why: the slow sway is all there (0&ndash;1 Hz), but only a fifth "
  "of the real movement in the 1&ndash;4 Hz beat band, and the deficit is the same on training music, so it "
  "is a modelling issue rather than overfitting. Next, three runs on the full 1,410-video dataset will "
  "separate the effects of our corrections and of model size: the small model with our corrections, the "
  "EDGE-size model without corrections, and the small model without corrections.")
fig(os.path.join(HERE, "fig3_results.png"), W,
    "<b>Figure 3.</b> (a) A generated 20 s dance (orange) next to the real dancer (green) on the same music. "
    "(b) The model sways (0&ndash;1 Hz) but lacks movement on the beat (blue bars).")

P("3.6 Team Member Task Breakdown", h2)
B("<b>YinHsuan Hsu:</b> data pipeline, librosa audio features, GPU / cluster setup, dataset loader.")
B("<b>Sizhe Zhang:</b> model implementation (GRU baseline, diffusion transformer), training on AWS "
  "SageMaker and Colab, experiments.")
B("<b>Yi Xin:</b> evaluation framework (FID, BAS, diversity, foot skating), 3D visualization, results "
  "documentation. All members contribute to the report and poster.")

# ----------------------------------------------------------------------------- 4
P("4. Feasibility and Limitations", h1)
P("<b>Endpoint.</b> We stop at skeleton generation: 2D for all 10 genres, then 3D joint sequences on AIST++. "
  "We do not render meshes or textures and do not simulate physics; these do not change the deep-learning "
  "question and do not fit in one semester. <b>Challenges and mitigations:</b>", base)
B("<i>Foot sliding:</i> velocity loss, ankle-contact labels and a foot-contact loss; measured with PFC.")
B("<i>Body-size drift in 2D</i> (no fixed bone lengths on screen): per-video scale and the projected "
  "bone-length loss (drift already down from 0.134 to 0.111, real 0.103).")
B("<i>Averaged, low-energy motion:</i> beat-band metric to detect it; v-prediction, energy loss, more data "
  "(basic dances, 2.5&times; the hours) and a larger model to fix it.")
B("<i>Overfitting on a small dataset:</i> dropout, weight decay, time-stretch and geometric augmentation, the "
  "three pose tools as extra views, best-checkpoint selection.")
B("<i>Compute for long sequences</i> (attention is O(T<super>2</super>)): train on 5 s windows and chain them; "
  "mixed precision; AWS spot instances and Colab.")
P("<b>Open questions.</b> (1) Whether 2D metrics agree with human judgement; BAS is unreliable in 2D (a "
  "frozen pose scores well), so we will need a better rhythm metric. (2) How the model behaves on "
  "out-of-distribution music (classical, ambient). (3) Whether learned audio features (wav2vec 2.0 [17]) "
  "beat hand-crafted ones. (4) How much camera perspective limits 2D training, and whether the pose tools' "
  "differences matter for generation.")

# ----------------------------------------------------------------------------- 5
P("5. Potential Impact", h1)
P("Beat2Body can give choreographers and dance learners new ideas across genres and supply games, "
  "animation and virtual performances with dance motion without motion capture. Because it learns from "
  "ordinary single-camera video, the approach extends to any dance style filmed online. Our comparison of "
  "three pose estimators tells future work which tool yields the best training data, and the 2D constraints "
  "and beat-band metric apply to any generated human motion. The same method could drive music-paced "
  "fitness training and rehabilitation exercises.")

# ----------------------------------------------------------------------------- refs
P("References", h1)
refs = [
    "S. Tsuchida, S. Fukayama, M. Hamasaki, and M. Goto, &ldquo;AIST Dance Video Database: Multi-genre, "
    "multi-dancer, and multi-camera database for dance information processing,&rdquo; in <i>Proc. ISMIR</i>, 2019.",
    "R. Li, S. Yang, D. A. Ross, and A. Kanazawa, &ldquo;AI Choreographer: Music conditioned 3D dance "
    "generation with AIST++,&rdquo; in <i>Proc. IEEE/CVF ICCV</i>, 2021, pp. 13401&ndash;13412.",
    "H.-Y. Lee, X. Yang, M.-Y. Liu, T.-C. Wang, Y.-D. Lu, M.-H. Yang, and J. Kautz, &ldquo;Dancing to "
    "music,&rdquo; in <i>Proc. NeurIPS</i>, 2019.",
    "J. Tseng, R. Castellon, and C. K. Liu, &ldquo;EDGE: Editable dance generation from music,&rdquo; in "
    "<i>Proc. IEEE/CVF CVPR</i>, 2023, pp. 448&ndash;458.",
    "L. Siyao, W. Yu, T. Gu, C. Lin, Q. Wang, C. Qian, C. C. Loy, and Z. Liu, &ldquo;Bailando: 3D dance "
    "generation by actor-critic GPT with choreographic memory,&rdquo; in <i>Proc. IEEE/CVF CVPR</i>, 2022.",
    "Z. Yang, A. Zeng, C. Yuan, and Y. Li, &ldquo;Effective whole-body pose estimation with two-stages "
    "distillation,&rdquo; in <i>Proc. IEEE/CVF ICCV Workshops</i>, 2023.",
    "Z. Cao, G. Hidalgo, T. Simon, S.-E. Wei, and Y. Sheikh, &ldquo;OpenPose: Realtime multi-person 2D pose "
    "estimation using part affinity fields,&rdquo; <i>IEEE TPAMI</i>, vol. 43, no. 1, 2021.",
    "V. Bazarevsky, I. Grishchenko, K. Raveendran, T. Zhu, F. Zhang, and M. Grundmann, &ldquo;BlazePose: "
    "On-device real-time body pose tracking,&rdquo; arXiv:2006.10204, 2020.",
    "J. Ho, A. Jain, and P. Abbeel, &ldquo;Denoising diffusion probabilistic models,&rdquo; in <i>Proc. "
    "NeurIPS</i>, 2020.",
    "J. Song, C. Meng, and S. Ermon, &ldquo;Denoising diffusion implicit models,&rdquo; in <i>Proc. ICLR</i>, 2021.",
    "J. Ho and T. Salimans, &ldquo;Classifier-free diffusion guidance,&rdquo; arXiv:2207.12598, 2022.",
    "T. Hang, S. Gu, C. Li, J. Bao, D. Chen, H. Hu, X. Geng, and B. Guo, &ldquo;Efficient diffusion training via "
    "Min-SNR weighting strategy,&rdquo; in <i>Proc. IEEE/CVF ICCV</i>, 2023.",
    "B. McFee <i>et al.</i>, &ldquo;librosa: Audio and music signal analysis in Python,&rdquo; in <i>Proc. "
    "14th Python in Science Conf.</i>, 2015.",
    "A. Vaswani <i>et al.</i>, &ldquo;Attention is all you need,&rdquo; in <i>Proc. NeurIPS</i>, 2017.",
    "T. Salimans and J. Ho, &ldquo;Progressive distillation for fast sampling of diffusion models,&rdquo; in "
    "<i>Proc. ICLR</i>, 2022.",
    "T.-Y. Lin <i>et al.</i>, &ldquo;Microsoft COCO: Common objects in context,&rdquo; in <i>Proc. ECCV</i>, 2014.",
    "A. Baevski, H. Zhou, A. Mohamed, and M. Auli, &ldquo;wav2vec 2.0: A framework for self-supervised learning "
    "of speech representations,&rdquo; in <i>Proc. NeurIPS</i>, 2020.",
]
for i, r in enumerate(refs, 1):
    P(f"[{i}]&nbsp; {r}", ref)


def footer(c, doc):
    c.saveState()
    c.setFont("Times-Roman", 8.5)
    c.setFillColor(colors.HexColor("#555555"))
    c.drawCentredString(letter[0] / 2, 0.42 * inch, f"Beat2Body project proposal  -  {doc.page}")
    c.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                        topMargin=0.6 * inch, bottomMargin=0.65 * inch, title="Beat2Body: Project Proposal",
                        author="YinHsuan Hsu, Yi Xin, Sizhe Zhang", subject="ELEC/COMP 576 final project proposal")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print(OUT)
