# What Auto Trace can and cannot read from a photograph

Auto Trace turns a picture of an artefact into a map symbol. It does that
well for some finds and not at all for others, and the difference is not
about picture quality — it is about **where the artefact's identity lives**.

This page records what was measured, so the same ground is not covered
again. Every figure below comes from nine photographs published by
국립중앙박물관 under 공공누리; their provenance is in
`tests/fixtures/real_sources.json`.

## Identity in the outline: this works

A stone dagger, a Liaoning bronze dagger, a slender bronze dagger — the
silhouette *is* the type, so tracing the silhouette gives a symbol that sits
correctly beside the drawn catalogue entry.

Two things had to be right for this:

- **Standing the find up.** Excavated material is photographed lying down.
  Traced as-is, the slender dagger came out as a horizontal lens. An
  elongated find (2.2:1 or longer) is now turned so its long axis is
  vertical, the way archaeological illustration draws it. See
  `stand_upright` in `generators/autotrace/segment.py`.
- **Picking one find.** Record shots routinely hold several objects and a
  label card. The mask already selects a single component and ignores the
  card; no change was needed.

## Identity in the decoration: this does not work from a photograph

A bronze mirror and a roof tile end are plain discs without their
decoration — indistinguishable from a posthole. That decoration is **shallow
relief**, which a photograph carries only as shading under whatever light
the object happened to be under.

Seven approaches were tried and measured, and all seven read the lotus tile
as 10, 8 or 12 petals depending on which pixel the image started at:

| input | fold count found | score |
| --- | --- | --- |
| drawn six-fold control, shifted 0–4 px | 6, 6, 6, 6, 6 | 0.3097 every time |
| photographed lotus tile, shifted 0–4 px | 10, 8, 12, 8, 8 | 0.033 – 0.100 |

A one-pixel shift cannot change how many petals a tile has, so this was not
a weak signal — it was the wrong instrument. **The cause was the frame.**
`find_rotational_frame` searched a grid of centres and kept whichever one
scored best, which solves a geometry problem with a noisy objective: the
best score is the maximum of a noisy field, and it moves whenever the input
does.

The frame is now found by geometry before any fold is counted:

1. a circle fitted to the silhouette with the cast-shadow skirt trimmed off
   (`trimmed_face_circle`), then
2. the centre walked downhill on the **one-cycle wave** the decoration makes
   in the radius its contrast sits at (`recentre_on_decoration`) — off
   centre, the same feature returns at a different radius on the far side of
   the face, and nulling that swing needs no fold count at all.

Only then do folds get counted, and every plausible frame votes, weighed by
its own score (`survey_folds`). The same photograph now reads **8 petals on
three of four nudges** — the tile has eight — eleven times faster than the
search it replaced, and every control drops:

| input (4 one-pixel nudges each) | folds | score |
| --- | --- | --- |
| plain drawn disc | 4 | 0.000 |
| drawn discs of scattered blobs, four seeds | wanders | 0.003 – 0.005 |
| **photographed dragon-motif tile** | wanders | 0.006 – 0.008 |
| **photographed bronze mirror** | wanders | 0.013 – 0.015 |
| two comb-pattern jars, dagger, ground stone | wanders | 0.001 – 0.015 |
| drawn six-fold disc | 6, 6, 6, 6 | 0.068 |
| **photographed lotus tile** | 8, 8, 8, 9 | 0.017 – 0.061 |
| drawn eight-fold disc | 8, 8, 8, 8 | 0.100 |
| drawn twelve-fold disc | 12, 12, 12, 12 | 0.349 |

So the gate (`FRAME_MIN_SCORE`, `generators/autotrace/round_motif.py`) sits
**above every control** at 0.03 — twice the loudest thing with no repeat in
it, half the weakest thing that has one. It was 0.15 while the score meant
"best fold score found by searching"; the score means something else now, so
the number moved with it. The gate is not weaker: the mirror and the dragon
tile it used to refuse are further below this one than they were below the
old one.

The fourth nudge of the lotus reads 9 at 0.017 and is refused. That is the
right outcome and not a failure of the gate — the tracer declines rather
than stamping nine petals onto an eight-petal tile.

Comb-pattern pottery was checked the same way and still fails: a bronze
dagger with no comb decoration at all scores in the same range (4.75–8.93×)
as the two comb-pattern jars (4.11–10.88×). Banded decoration is not read.

Everything refused is refused out loud, with a log line naming the fold
count and score. Stamping petals onto a dragon-motif tile would be worse
than drawing it plain.

## A second opinion before the decoration is drawn

The frame above is this project's own invention, and the cost of it being
wrong is the worst failure this tracer has. So the answer is checked against
a published method that reaches it a completely different way: Loy and
Eklundh (ECCV 2006), the baseline the CVPR symmetry competitions use for
rotation symmetry. Matched feature pairs each vote — two patches that look
alike are related by some rotation, and a rotation through a known angle
about an unknown centre pins that centre down exactly — so the centre is an
**output** of the vote, never an input to it. That is precisely the part this
project kept getting wrong.

Measured on the same nine photographs, as a fraction of the face radius:

| find | our frame vs. the vote | plain ellipse fit vs. the vote |
| --- | --- | --- |
| lotus roof tile end | **0.008** | 0.21 |
| dragon roof tile end | **0.010** | 0.05 |
| bronze mirror | **0.012** | 0.02 |
| comb-pattern jar | 0.47 | 0.26 |
| bipa-shaped dagger | 0.64 | 0.35 |
| polished stone dagger | 0.81 | 0.76 |
| ground stone tool | 0.90 | 0.73 |
| slender bronze dagger | 1.60 | 0.80 |

Two independent methods land within one percent of a radius of each other on
every decorated disc, and half a radius apart or more on everything that is
not one. So the vote runs as the last check before a motif is committed —
only when the score has already passed, where it costs 0.2–0.5 s — and a
disagreement past 0.12 of a radius refuses the reading. Silence is not
disagreement: a worn or plain surface gives the vote nothing to match, and a
guard that fired on that would refuse the artefacts most in need of help.

On this corpus the guard changes no outcome: the only find that passes the
score gate is the lotus tile, and there the two methods agree. It is a lock,
not an improvement.

**Only the centre is taken from that method.** Its fold-count step was
measured on the same photographs and does not separate an eight-petal tile
from a dragon — both came out at a matched-pair concentration of 0.28, the
tile reading 8 and the dragon 4. Four statistics on those pair angles were
tried and each either named the wrong count for the tile or scored a control
above it: Rayleigh on the descriptor orientation difference, Rayleigh on the
positional rotation, bootstrap stability of the winning count, and agreement
between radial bands (the lotus gave 8 and 7 in its two halves; the dragon
11 and 18, which is the right refusal for the wrong reason — the margin is
one fold, not a gap).

## The direction not taken: relief from one photograph, learned

Recovering shallow relief from a *single* image is an active problem with
published results — MonoRelief and MonoRelief V2 do exactly this, and unlike
the lamp-moving path below they need nothing of the photographer. That would
plug straight into the `relief` argument `run_autotrace` already takes.

It is not used here: no pretrained weights are released, and the method needs
PyTorch and Depth-Anything-V2 behind it, which is not a dependency a QGIS
plugin can carry. If weights appear in a form that runs under the
`onnxruntime` this plugin already uses, this is the first thing to try.

## Moving the lamp: reading relief the way it is meant to be read

There is a way to get the decoration from the object itself rather than from
a rubbing, and it works because it adds the information one photograph
physically cannot hold.

Stains, discolouration and grain — the things that beat every single-image
attempt — do not change when the lamp moves. Relief does. So photograph the
artefact **three to five times with the light moved between shots and the
camera left where it is**, and the per-pixel variation across that stack is
the decoration with the surface colour divided out. This is the simplified
form of the Reflectance Transformation Imaging archaeology already uses for
exactly this problem; a desk lamp and a phone on a stand are enough.

Pass the extra frames as `light_stack` (the dialog takes a multiple
selection). On a synthetic eight-petal tile carrying stains, one frame is
refused at 0.022 and the five-frame stack reads eight petals and traces them.

Two things to get right when shooting:

- **Three or more positions.** The lamps are an arrangement in a circle too,
  and they used to be read as one: with the searched frame, four evenly
  spaced lamps made an eight-petal tile read as four, and `relief.py` carried
  a guard that refused any reading equal to the lamp count. That confound
  came from the frame, not from the lamps. With the frame fixed by geometry
  it does not reproduce — six fold counts against five lamp counts all read
  correctly — and the guard is gone with it, since it would now only refuse
  correct answers. Five positions are still advised, because more lamps means
  more relief separated from more stain.
- **Do not move the camera.** Frames are aligned by translation, but a frame
  that has drifted more than about 8% of the image is used unaligned and
  said so in the log.

This path has been verified on synthetic stacks built from known height
fields, not on real multi-light photographs of Korean artefacts — none were
available to this work.

## What to feed it instead

The reading works, and works exactly, when the repeat is clean. A **rubbing
(탁본) or a measured drawing** of the same mirror or tile scores twice the
gate where the photograph scores a fifth of it, and traces to a proper
motif — eight petals around a boss.

Set **Input type → Drawing / rubbing** in the dialog for these. Note the
input has to be the sheet itself: a rubbing mounted as a hanging scroll and
then photographed is a photograph, and is treated as one (three such items
are catalogued in the manifest as a warning to the next reader).

## Summary

| artefact | from a photograph | from a rubbing / drawing |
| --- | --- | --- |
| blades, stone tools | works | works |
| pottery vessels | outline only, no surface pattern | pattern read |
| mirrors, roof tile ends | outline only — declined, drawn plain | motif read and replayed |

With a multi-light stack, mirrors and roof tile ends move into the last
column without needing a rubbing at all.
