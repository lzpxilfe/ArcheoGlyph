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
its own score (`survey_folds`). That is eleven times faster than the search
it replaced and leaves every control below the gate.

### What was measured on a mask that changed every time

This section used to report that the same photograph now reads **8 petals on
three of four nudges at 0.061**. That was wrong, and the way it was wrong is
worth recording.

`get_mask_opencv` was not deterministic. `cv2.grabCut` seeds its colour
models with k-means, and OpenCV's k-means draws from one global RNG whose
state advances with every call, so **the same photograph returned a different
silhouette every time it was traced** — on eight of nine finds, three calls in
a row gave three different masks (the lotus tile: 364941, 389658 and 368993
pixels), and the same three in the same order in a fresh process. A user
pressing the button twice got two different symbols.

Every real-photograph number reported here was therefore one draw of a
lottery. GrabCut now votes over three fixed seeds and the mask is a function
of the image again. Re-measured on that basis, over four one-pixel crops each:

| input | folds | score |
| --- | --- | --- |
| plain drawn disc | 4 | 0.000 |
| drawn discs of scattered blobs, four seeds | wanders | 0.003 – 0.005 |
| **photographed dragon-motif tile** | 8/8/14/4 | 0.005 – 0.009 |
| **photographed bronze mirror** | 5/5/5/15 | 0.013 – 0.015 |
| two comb-pattern jars, daggers, ground stone | wanders | 0.001 – 0.018 |
| **photographed lotus tile** | 7/9/7/8 | 0.013 – 0.060 |
| drawn six-fold disc | 6, 6, 6, 6 | 0.068 |
| drawn eight-fold disc | 8, 8, 8, 8 | 0.100 |
| drawn twelve-fold disc | 12, 12, 12, 12 | 0.349 |

The gate (`FRAME_MIN_SCORE`) still sits **above every control** at 0.03 —
above the loudest thing with no repeat in it, well below every drawn repeat.
That number was set from the controls and the controls have not moved, so it
stands. What does not stand is the claim about the tile: reproducibly it
clears the gate on **one of four crops**, not three.

### Why, and what has to improve

The cause is measurable, and it is the most useful thing this line of work
turned up:

- a drawn repeat keeps its answer while the frame's centre is moved up to
  **0.03 of the face radius** (0.05 for six folds, 0.03 for eight and twelve);
- a **one-pixel change of crop** moves the frame by **0.059 of a radius** on
  the lotus photograph — twice the width of the basin the reading lives in.

The frame is not repeatable to the precision the fold reader needs. Until it
is, no threshold can make the reading consistent, because the thing that
moves is upstream of the score.

So the tracer now refuses a reading that sits on the edge of its basin.
`reading_is_stable` pushes the frame as far as a drawn repeat can be pushed
and requires the same fold count back from every nudge; the pipeline runs it
between the score gate and the feature-vote check. Drawn six-, eight- and
twelve-fold discs pass it, under a lighting gradient and under grain. The
lotus tile's one gate-clearing crop passes it too — so today the tile is still
drawn on one crop in four, and that is stated rather than fixed. What would
fix it is a frame that lands in the same place twice.

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
| dragon roof tile end | **0.011** | 0.050 |
| bronze mirror | **0.017** | 0.020 |
| lotus roof tile end | **0.072** | 0.246 |
| comb-pattern jar (b) | 0.235 | 0.085 |
| comb-pattern jar | 0.333 | 0.261 |
| bipa-shaped dagger | 0.579 | 0.353 |
| polished stone dagger | 0.924 | 0.759 |
| slender bronze dagger | 0.973 | 0.797 |
| ground stone tool | 1.016 | 0.734 |

(These are the deterministic figures. The first version of this table read
0.008 / 0.010 / 0.012 against 0.35 and up, measured on the shifting masks
described above.)

Two independent methods land inside 0.072 of a radius of each other on every
decorated disc, and 0.235 or more apart on everything that is not one. The
gap is real but narrower than it first appeared, and it rests on one awkward
case at each end — the lotus tile at 0.072 and a comb-pattern jar at 0.235.
So the vote runs as the last check before a motif is committed —
only when the score has already passed, where it costs 0.2–0.5 s — and a
disagreement past 0.12 of a radius refuses the reading. Silence is not
disagreement: a worn or plain surface gives the vote nothing to match, and a
guard that fired on that would refuse the artefacts most in need of help.

On this corpus the guard changes no outcome: the only find that passes the
score gate is the lotus tile, on one crop, and there the two methods agree
to 0.072. It is a lock, not an improvement.

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

## The silhouette lump was the test harness, not the tracer

For several rounds this document and its author treated a lumpy silhouette on
the roof tiles as a defect in the tracer. It was not. The render script that
produced those sheets pinned `mask_backend` to `opencv`. The plugin's default
is `auto`, which prefers the ONNX salient-object model and cross-checks it
against the OpenCV mask:

| find | opencv | onnx (what `auto` picks) |
| --- | --- | --- |
| lotus roof tile end | 615269 px, solidity 0.972, circularity 0.782 | **324000 px, 0.994, 0.876** |
| dragon roof tile end | 369142 px, 0.969, 0.767 | **354364 px, 0.996, 0.889** |
| comb-pattern jar | 483113 px, 0.945, 0.659 | **458769 px, 0.995, 0.716** |
| bronze mirror | 455613 px, 0.994, 0.869 | **450488 px, 0.998, 0.897** |

The white support block under the lotus tile is **43 percent of the OpenCV
mask** and none of the ONNX one. `auto` chose the model mask on all of them.

This matters for a user only if they run `opencv` — the "no extra download"
setting — where the lump is real. It is a reason to install the model, not a
bug in the tracer.

## Twenty-four marks nobody could see

With the default backend the silhouettes came out clean and a different
defect was left in plain sight: Line and Measured were drawing **24 interior
marks** inside a round artefact, and half of them were specks.

| find / style | median mark span (of the symbol box) | at 64 px |
| --- | --- | --- |
| comb-pattern jar, Line | 0.031 | **2.0 px** |
| dragon tile, Line | 0.050 | **3.2 px** |
| bipa-shaped dagger, Measured | 0.044 | **2.8 px** |
| bronze mirror, Line | 0.063 | 4.0 px |

A symbol is 64 grid units and a legend shows it at 64 pixels, so a unit is a
legend pixel and `icon_grid.DETAIL` — the internal line weight — is exactly
one. A mark two or three pixels across is not a line at that size. Two rules
follow, both taken from measurements rather than taste:

- `geometry.LEGEND_MARK_MIN_SPAN` = four detail-widths. Shorter than the least
  that can read as a stroke, and it goes.
- `geometry.MAX_INTERIOR_MARKS` = 11, the busiest symbol in the drawn
  catalogue (whose median artefact carries 2 and whose ninetieth percentile
  is 5). A traced symbol may be as busy as the busiest drawn one, no busier.
  What survives is kept largest first.

A folded rotational motif is exempt from the count — it is stamped once per
fold, and trimming it would leave the face decorated round part of its turn
and bare for the rest. A drawing is exempt too, because there the ink strokes
are the content rather than an inference about it; its specks still go.

### The blobs, fixed by changing what is read rather than what is kept

Three attempts to tell decoration from lighting *after* extraction all failed
on the same nine photographs, and a fourth was measured and rejected:

| rule tried | what killed it |
| --- | --- |
| thickness at legend size | a good mark on the stone dagger is 8.2 px, a bad one on the dragon tile 3.0 px |
| size of a closed region | the drawn catalogue uses closed interior shapes up to 0.88 of the tile |
| stability under a re-crop | the pots score 0.08 and 0.33, the discs 0.42 to 0.58 - backwards |
| concentric radial bands | works (a lotus tile gives the same five bands over three re-crops, a dragon tile a different set each time) but only ever yields rings |

The information needed is not in the extracted marks, because decoration and
lighting arrive there in the same shapes. What was wrong was **what was being
read**, not what was being kept.

A roof tile end's decoration is *height*, and a photograph carries height only
as shading. But the shading is local: subtract a wide blur and the lamp goes,
because a lamp is broad and a groove is not. What remains is **a rubbing of
the object** — and a rubbing is an input this tracer already reads well.

So for a round photograph, `enhance.relief_ink_sheet` renders the relief as
ink on paper and the existing ink-centreline tracer works on that. The
silhouette still comes from the photograph; only the ink comes from the
relief. On the lotus roof tile end this yields the petal ring, the boss with
its ring of beads and the outer bead ring — the drawing an archaeologist would
make — where reading the photograph directly gave a diagonal band across three
quarters of the face.

Those strokes are the content, as a real rubbing's are, so they are exempt
from `MAX_INTERIOR_MARKS`: that cap is the busiest drawn *legend symbol*, and
Line and Measured are documentation plates. Capping them there cut a
125-stroke rosette down to ten arcs. The size floor still applies — a speck is
unreadable whatever drew it.

### What this does not fix

The comb-pattern jar's Line output is clean and the daggers keep the marks
they should; neither is round, so the relief route leaves them alone.

The **bronze mirror** gains little. Its surface is worn and dark and the
relief map finds mostly its rim, which is honest — a plain disc is what that
photograph supports. The **rotational motif** is still read separately and
still refused on all three of these photographs; nothing here changes that
gate, and the fold count remains unrepeatable across crops for the reason
recorded above.

## The unsupported-boundary measurement, not shipped

A cast shadow fused to the silhouette puts a lump on every traced symbol that
the artefact does not have. There is a clean way to see it: an artefact's edge
has an image gradient under it and a shadow's does not, because a shadow's own
boundary is a soft gradient somewhere out on the paper. Measured along the
mask contour, the share with no edge beneath it is **0.00–0.01** on the five
finds photographed without a visible shadow and **0.09–0.30** on the four with
one, and it lands on the skirt and nowhere else. Cutting those runs and
closing each with a chord lifts a comb-pattern jar's silhouette from 0.934 to
0.984 solidity.

It is not in the code. On the lotus tile the same cut takes 0 to 15 percent of
the mask depending on which pixel the crop starts at, and that killed the one
motif reading that worked. A mask step that is unstable under a one-pixel
crop is disqualified whatever it gains elsewhere — that is the same standard
this document applies to the fold reader. The measurement is recorded here so
the next attempt starts from it rather than from scratch.

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
