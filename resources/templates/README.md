# Template SVG files

This directory is empty on purpose. Every one of the catalogue's 188 symbols
is drawn in code, on the 64-unit grid in `generators/icon_grid.py`, so that
they read as one set and can be published without depending on anyone else's
artwork.

An SVG dropped in here is still picked up: `TemplateGenerator._template_file`
looks for the `file` name a catalogue entry carries and colourises it. That is
a way to override a symbol locally, not the way the catalogue works. An entry
that also carries a `draw` function keeps its drawn symbol - the file does not
win, because a silent replacement of a deliberate drawing is not an override,
it is a surprise.
