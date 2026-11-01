<!-- Please read CONTRIBUTING.md first. One change per pull request. -->

## What and why

<!-- Symptom (what was wrong in the game or the launcher) and the cause on the emulator side. -->

## Switch

<!-- Name, values (0 = old path, 1 = new path, 2 = verify) and the default in this pull request.
     For launcher-only, documentation or build changes write "none". -->

## Verify mode (`=2`)

<!-- What the verify mode compares, the number of differences, and where in the game it ran. -->

## Cost or gain

<!-- ms per frame of the affected pass (KYTY_TRACE_GPU_TIME) and/or live A/B in the same run (KYTY_LIVE_AB):
     fps / frame time with the switch off and on, GPU model and driver, resolution. -->

## Screenshots

<!-- Before and after, same place and settings. Required for changes to the image. -->

## Checklist

- [ ] The change has its own switch, documented in a comment where it is read (not needed for launcher-only, documentation or build changes).
- [ ] Verify mode (`=2`) runs the old and the new path and reports **0 differences** in the game.
- [ ] The description gives the cost or gain in ms per frame, or a live A/B from the same run.
- [ ] Screenshots before and after are attached (for changes to the image).
- [ ] No workaround is added, or it is off by default and described as a diagnostic.
- [ ] Anything beyond the PS5's result is a separate switch, off by default, with its cost stated.
- [ ] No game code, game files, shader dumps, decompiler output, or addresses or symbols from the game's executable are included or quoted.
- [ ] Code is formatted with clang-format, and code adapted from another project names its origin and has a GPL-2.0-only compatible license.
