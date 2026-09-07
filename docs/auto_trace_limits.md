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

Seven approaches were tried and measured. The last one is the informative
one:

| input | fold count found | score |
| --- | --- | --- |
| drawn six-fold control, shifted 0–4 px | 6, 6, 6, 6, 6 | 0.3097 every time |
| photographed lotus tile, shifted 0–4 px | 10, 8, 12, 8, 8 | 0.033 – 0.100 |

A one-pixel shift cannot change how many petals a tile has. The reading
moves because the signal is on the noise floor, not because the method is
weak — on the drawn control the same method is exact and perfectly stable.

Comb-pattern pottery was checked the same way and fails the same test: a
bronze dagger with no comb decoration at all scores in the same range
(4.75–8.93×) as the two comb-pattern jars (4.11–10.88×).

So the gate (`FRAME_MIN_SCORE`, `generators/autotrace/round_motif.py`) sits
**above** that noise band at 0.15. Photographs of relief decoration are
declined and the artefact is drawn plain, with a log line naming the fold
count and score that were refused. Stamping petals onto a dragon-motif tile
would be worse than drawing it plain.

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
