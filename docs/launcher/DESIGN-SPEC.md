# Launcher design spec (v1)

The launcher look of the Yōtei fork: snow-paper pages, ink sidebar and bars, one ginkgo-gold accent per screen, pastel
ukiyo-e landscapes. Source: the project's Figma file (pages Foundations, Components, Screens), October 2026. This file is
the reference for the Qt launcher (`src/launcher`); the values below are the ones used in code.

| What | Where in the repo |
|---|---|
| Colour, spacing, radius and type tokens | `src/launcher/styles/tokens.json` (read at start-up by `launcherTheme.cpp`) |
| Style sheet (uses `@token` names) | `src/launcher/styles/launcher.qss` |
| Layout | `src/launcher/forms/main_dialog.ui` |
| Painted parts (art backgrounds, art buttons, nav items) | `src/launcher/src/launcherWidgets.cpp` |
| Game tiles (Play) | `src/launcher/src/gameTileView.cpp` |
| Landscapes | `src/launcher/images/ukiyoe-*.jpg` |
| Icons | `src/launcher/icons/` (toggle, chevron, status glyphs) |

## Rules

1. **One gold accent per screen.** If the screen has a primary action (PLAY, Choose game folder) that button is gold
   and the active sidebar item is neutral. Otherwise (Display, Graphics, Shaders, Advanced) the active sidebar item is
   gold. Everything else is ink, stone and mist. On Play the selected game's tile has a gold frame too: it is what
   PLAY starts.
2. **Hairlines, not shadows.** 1 px lines in `border/default`; no drop shadows; no gradients except over the artwork.
3. **No game characters or game marks** in the art. The landscapes are generic ukiyo-e scenes (mountain, fields, ginkgo
   forest, misty ridges, lake) made for this project. Game covers and game pictures on Play are read from the
   player's own game folder while the launcher runs; no game file is in the repository or in the resources.
4. **Licence text only in About** (Advanced page) and in the repository files, never on the other screens.

## Colour

Palette (`palette` in tokens.json):

| Token | Hex | Use |
|---|---|---|
| paper | `#ECE8E0` | page background (snow paper) |
| paper-light | `#F4F1EB` | cards, fields, toggle knob |
| ink | `#171513` | sidebar, bottom bar, primary text, strong borders |
| ink-2 | `#24211E` | active sidebar item |
| gold | `#E0A526` | the one accent (ginkgo gold) |
| ochre | `#C2922F` | accent hover, warning, art-button frame |
| stone | `#6F6A62` | secondary text, inactive nav dot |
| mist | `#C9C6BF` | hairlines, track, muted text on ink |
| success | `#5E8C55` | check OK |
| error | `#9A3B2C` | errors, danger button |

Semantic names (`color` in tokens.json, written `@bg-page`, `@text-secondary`… in the style sheet): bg/page → paper,
bg/card and bg/field → paper-light, bg/sidebar and bg/bar → ink, bg/sidebar-active → ink-2, text/primary → ink,
text/secondary → stone, text/on-dark → paper, text/on-dark-muted → mist, text/on-dark-strong → paper-light,
text/on-accent → ink, text/error → error, border/default → mist, border/strong → ink, border/on-dark → mist,
border/error → error, accent/default → gold, accent/hover → ochre, status/success|warning|error → success|ochre|error,
control/on → ink, control/off → mist, control/knob → paper-light, track/bg → mist, track/bg-dark → stone,
track/fill → ink, icon/on-status → paper-light, nav/dot → stone, nav/dot-active → paper, nav/bar → mist.

## Type

Families: **Cormorant Garamond** (headings, brand, PLAY), **Manrope** (text, labels), **JetBrains Mono** (values,
paths). All three are under the SIL Open Font License. Until the font files are bundled (`src/launcher/fonts`, loaded from
`:/fonts`), the launcher falls back to Georgia / Segoe UI / Consolas (`fallback` in tokens.json).

| Style | Font | Size / line (px) | Tracking | Use |
|---|---|---|---|---|
| Heading/Title | Cormorant Garamond SemiBold | 52 / 58 | 0 | page title |
| Heading/Section | Cormorant Garamond SemiBold | 26 / 32 | 0 | card and section titles |
| Brand/Wordmark | Cormorant Garamond SemiBold | 32 / 38 | 0 | "KytyPS5" in the sidebar, loading screen |
| Brand/Play | Cormorant Garamond Bold | 26 / 32 | +14 % | PLAY |
| Body/Default | Manrope Regular | 14 / 20 | 0 | text, select values |
| Body/Strong | Manrope Medium | 14 / 20 | 0 | setting labels |
| Body/Description | Manrope Regular | 13 / 18 | 0 | descriptions |
| Body/Lead | Manrope Regular | 15 / 24 | 0 | page lead paragraph (max 520 px) |
| Label/Eyebrow | Manrope SemiBold | 12 / 16 | +10 %, caps | eyebrow above titles, "YŌTEI FORK" |
| Label/Button | Manrope SemiBold | 14 / 20 | 0 | buttons |
| Label/Nav | Manrope Medium | 15 / 20 | 0 | sidebar items |
| Label/Caption | Manrope Regular | 12 / 16 | 0 | sidebar footer |
| Mono/Value | JetBrains Mono Regular | 13 / 18 | 0 | values, status line |
| Mono/Path | JetBrains Mono Regular | 12 / 16 | 0 | paths, progress numbers |
| Overlay/Label 2x, Overlay/Mono 2x | Manrope / JetBrains Mono Regular | 24 / 32 | 0 | in-game badge at 2160p |

## Space and radius

Spacing scale: 4, 8, 12, 16, 24, 32, 48. Radius: sm 6 (art buttons, progress bars), md 8 (buttons, fields, selects),
lg 10 (cards, status icons).

## Window

1280 × 800 (minimum 1120 × 740). Left: **Sidebar** 248 px, ink. Right: the page, 1032 px wide at the design size.

### Sidebar

Padding 32 top and bottom. Brand block: padding 0 24 32, gap 4: "KytyPS5" (Brand/Wordmark, text/on-dark) and
"YŌTEI FORK" (Label/Eyebrow, text/on-dark-muted). Nav items, then a spacer, then the footer (padding 0 24, gap 4,
Label/Caption): "by Erlou" (on-dark), the fork version "v0.1" (`FORK_VERSION`, on-dark-muted), "F2 overlay ·
Shift+F2 details" (on-dark-muted). Nav items: Play, Display, Graphics, Shaders, Advanced.

**NavItem** — 248 × 46, padding left 24, dot 6 × 6, gap 12, Label/Nav.

| State | Background | Marker (2 px, left edge) | Dot | Text |
|---|---|---|---|---|
| inactive | none | none | stone | mist |
| hover (launcher addition) | ink-2 at 50 % | none | stone | mist |
| active (Play) | ink-2 | mist | paper | paper |
| active-accent (other pages) | ink-2 | gold | gold | paper |

### Page header and banner

Settings pages start with a **Banner** 1032 × 200: the page's landscape (cover), a 56 px fade into paper at the bottom,
and the PageHeader at padding 48 48: eyebrow "SETTINGS" (Label/Eyebrow, stone), gap 12, title (Heading/Title, ink).
Body below: padding 16 48 32.

| Page | Image | Crop of the image (x, y, w, h as fractions) |
|---|---|---|
| Play (background 1032 × 716, no game selected) | ukiyoe-play.jpg (1264 × 848) | 0.0601, 0.0448, 0.8797, 0.9104 |
| Display banner | ukiyoe-display.jpg (1584 × 672) | 0, 0.1786, 1, 0.4568 |
| Graphics banner | ukiyoe-graphics.jpg (1584 × 672) | 0, 0.1935, 1, 0.4568 |
| Shaders banner | ukiyoe-shaders.jpg (1584 × 672) | 0, 0.2976, 1, 0.4568 |
| Advanced banner | ukiyoe-advanced.jpg (1584 × 672) | 0, 0.1190, 1, 0.4568 |
| First start (background 1032 × 800) | ukiyoe-first-start.jpg (1264 × 848) | 0.0671, 0, 0.8658, 1 |
| Loading screen (1280 × 720) | ukiyoe-loading.jpg (1376 × 768) | cover |
| Art buttons (ginkgo branch) | ukiyoe-play.jpg | 0.6171, 0.0826, 0.3639, 0.1238 (stretched into the button) |

At other window sizes the crop keeps its centre and is trimmed to the new aspect ratio (cover).

### Play

The game grid is the main element (launcher v2, October 2026; the v1 table was too small). Content padding 48 48 32:
PageHeader (eyebrow "<title id> · found", title "Ready to play"; no lead paragraph — the grid needs the height), 12 px,
then the cards row (gap 24, top-aligned): the **Games** card (fills the width) and the **Checks** card (320 wide).
**Bottom bar**: 84 px, ink, padding 0 48, gap 16: status line (Mono/Value, mist: "3840×2160 · window · tessellation
on · cache 41%"), spacer, "Precompile shaders" art button (height 40), PLAY art button (220 × 48). The status line
shows the settings the selected game starts with (its own settings or the global ones); the Figma "Current settings"
card gave way to it (its values are also on Display / Graphics).

**Background.** With a game selected: that game's picture from its own folder, `sce_sys/pic0.png` (else
`sce_sys/pic1.png`; Ghost of Yōtei has both at 3840 × 2160), read when the game is selected, decoded and scaled to
2048 px wide off the UI thread, drawn whole as cover under an **ink layer at 62 %**. The page is then "dark": eyebrow,
title and card text in paper (secondary text mist), cards are dark glass (ink at 59 %, 1 px paper hairline at 18 %),
the search field is ink at 47 % with a paper hairline, tool icons paper. Without a selected game, or when the game
has no picture: image B under a left-to-right paper scrim (paper 55 % at the left edge → 0 at 62 % of the width),
paper cards and ink text as everywhere else. The previous picture stays until the next one is decoded.

**Games card** (Card, padding 16, gap 12): the tool row, then the tiles. Tool row: 32 × 32 icon buttons with 20 px
icons, 4 apart — refresh, add game folder (the folder with eboot.bin and sce_sys, or a folder of games; same as First
start), global settings, input mapping, edit game settings, clear game settings, trophies — then the search field
(200–280 × 32, "Search name or serial") on the right. The card is as tall as its rows (the picture shows below it);
with more games than fit, it reaches the bottom of the page and the tiles scroll (8 px scroll bar).

**GameTile** — 172 × 216, tiles 8 px apart (three in a row at 1280 × 800, two at the minimum window size):
- cover 156 × 156 at 8 px, radius 6, 1 px hairline (mist; paper at 14 % when dark): the game's `sce_sys/icon0.png`
  (square crop). No icon: the cover is mist (ink-2 when dark) with the initials of the name in Cormorant Garamond
  SemiBold 44, stone (mist when dark);
- name 6 px below the cover: Body/Strong, ink (paper-light when dark), one line, elided;
- 2 px below: "serial · version" (`PPSA26344 · 01.512.000`), Mono/Path with −0.2 px tracking, stone (mist when dark);
- states: hover — radius 8 fill, mist at 35 % (paper at 7 % when dark); selected — fill paper (paper at 12 % when
  dark) and a 2 px gold frame, radius 8; running — a "Playing" pill on the cover (ink 85 %, Label/Caption, paper);
- click selects, double click = PLAY, right click = the game menu (run, open folder, trophies, cheats, save data,
  game settings). The tooltip lists name, serial, versions, path.
- empty: "No games yet" (Heading/Section) and how to add a folder (Body/Description) in the middle; while a search
  hides every tile: "No game matches “…”".

**Checks card**: emulator — only when something is wrong (error: "Emulator not found", or "Emulator not responding"
when it printed no version; the description says where it must be), game folder found (path · eboot.bin, sce_sys;
paths break after their separators), game version (from the game's `sce_sys/param.json`, as on the tile; with a game
selected): "Game version 01.512.000 · tested" (success) for the version the fork is tested on, else "Game version … ·
untested" (warning) with what the fork is tuned for, PLAY stays on (the tested game and version are one constant,
`TESTED_TITLE_ID` / `TESTED_GAME_VERSION` in `src/launcher/include/forkInfo.h`), graphics card · VRAM (the card the emulator will use: the chosen one, or for
Auto the last discrete GPU as the emulator picks it), RAM, and "Shader cache: N% precompiled" — the share of the
recorded shaders compiled at the last precompile (stored per title id from its `PRECOMPILE_*` lines; the emulator
has no way yet to report it without compiling). Overlay and Ray tracing pool rows wait for the emulator (see below).

**Without the emulator** (no `kyty_emulator.exe` in the launcher's folder or the one above, or it prints no version)
the launcher stays open: the red row in Checks (and in the First start System card), the title says "Emulator not
found", PLAY and Precompile are off, Advanced shows "Emulator not found" and no log folder.

The game table of v1 is still there for `--local` mode, where the compatibility status and comment are edited.

**Card** — paper-light, 1 px mist border, radius 10, padding 24, gap 16; title Heading/Section. KeyValueRow: label
Body/Default stone (left), value Mono/Value ink (right), rows 20 px apart 12. **CheckRow**: StatusIcon 20, gap 12,
title Body/Default, description Body/Description stone 2 px below.

### First start

Shown instead of Play while no game is found (no sidebar item active). Background image (first start crop) under a
paper scrim (85 % → 0 at 70 % of the width) and a 300 px header wash (paper 95 % → 0). PageHeader "FIRST START" /
"Select your game" / lead, then two cards (gap 24, top-aligned): **Game folder** (540) — CheckRow "No game found",
path field ("No folder selected"; error look when the chosen folder holds no game), "Choose game folder" art button;
**System** (372) — graphics card · VRAM, RAM. A chosen folder may be the game's own folder (eboot.bin, sce_sys) or a
folder of games; it is added to the game folders, its first game is selected and Play opens.

### Display, Graphics

Banner page. They edit the selected game's own settings when it has them (eyebrow "SETTINGS · PPSA26344 ONLY"), else
the global ones ("SETTINGS · ALL GAMES"); every change is saved at once and used at the next start.
- Display: Resolution (select 240), Full screen (toggle), Present mode (select 240: Fifo, Mailbox, Immediate; a
  launcher option the Figma frame does not show), Overlay (select 240: Off / On = `--perf-overlay`, the performance
  panel from the start; F2 hides or shows it, Shift+F2 its details).
- Graphics: Tessellation (toggle; off passes `--no-tessellation`), Graphics card (select 400: Auto (<card>) and every
  Vulkan device with its VRAM, in the emulator's `--gpu` order).
- Not shown until the emulator has them: Aspect ratio (Display), Ray tracing pool (Graphics), overlay starting compact.

### Shaders

Section "Precompile" (Heading/Section). **SettingRow**: padding-top 12; label column 280 px (Body/Strong); gap 24;
control column (gap 8): control, then description (Body/Description, stone); then a 1 px divider 12 px below.
Last row of a section has no divider; the next section title has 24 px above it.

- Precompile all shaders: "Precompile shaders" / "Continue precompiling" (after a pause) and, while it runs, "Pause"
  (the emulator finishes the shaders being compiled, saves and stops; a second press is "Force stop"); progress bar
  (6 px, track mist, fill ink, radius 3); progress line (Mono/Path, stone): "17 840 / 28 700 shaders" left, "~31 min
  left" right; status line for pausing / saving / the result; description.
- Precompile on start (toggle, on by default): off starts the game with `KYTY_SHADER_PRECOMPILE=0` (no background
  compiles and no shader journals — the emulator has one switch for both).
- Precompile threads: select-like spin box 160 × 40; also used for the game's background compiles
  (`KYTY_SHADER_PRECOMPILE_THREADS`).
- Section "Cache" — Shader cache: "Size on disk" (+ "· N% precompiled"), "Open folder", "Clear cache" (danger): after a
  confirmation that names the folder, the file count and the size, every file of `_PipelineCache` goes to the Recycle
  Bin (nothing is deleted for good); off while the game or a precompile runs. "Share shader hints / Export hints"
  waits for precompile step 3.

### Advanced

Sections "Folders" (Game folder with "Change...", Log folder with "Open" — one log per game run in `_Logs` next to the
emulator: a header with the command line, then the emulator's stderr; the newest 10 are kept — Emulator, Settings
file), "Support" (Bug report: "Export bug report"), "Updates" (shown only where the update check is supported — off in
the fork until it has its own release feed, `forkInfo.h`), "About": Version ("KytyPS5 · Yōtei fork · v0.1",
description "by Erlou · emulator build <build>"), Licence ("GPL-2.0 (source code on GitHub)" and a "Source code"
button once `FORK_REPOSITORY_URL` is set), Credits.

**Export bug report** (`bugReport.cpp`): a dialog lists every file of the zip — `report/system.txt` (versions, OS,
CPU, RAM, cards and drivers, the selected game and its settings), the launcher settings file, the newest 3 run logs and
the last precompile log — with a preview of exactly what goes in. Before packing, the Windows user name (in paths, and
as a word), the computer name and the PS5 user names are replaced with `<user>`, `<computer>`, `<user-name>`. It says
that nothing is sent and that GitHub issues are public, saves the zip where the user chooses, then offers "Open GitHub
Issues" (`FORK_REPOSITORY_URL`/issues/new) — or, while the repository is not public (the constant is empty), only
"Show folder".

## Components and states

| Component | Default | Hover | Pressed | Disabled | Other |
|---|---|---|---|---|---|
| Button primary (gold) | gold fill, ink label | ochre fill | — | 40 % opacity | radius 8, height 40, padding 0 16 |
| Button secondary | 1 px ink border, ink label, no fill | paper-light fill | mist fill | ink at 40 % (label and border) | radius 8, height 40 |
| Button danger | 1 px error border, error label | paper-light fill | — | 40 % opacity | |
| Button secondary-on-dark | ink fill, 1 px mist border, paper label | ink-2 fill | — | 40 % opacity | on the bottom bar |
| Button image / PLAY image (art button) | art crop + ink tint 60 %, 1.5 px ochre frame, paper label | tint 48 %, gold frame, paper-light label | tint 70 % (launcher addition), gold frame | whole button 40 % opacity | radius 6; PLAY 220 × 48 with Brand/Play |
| Toggle | on: ink track, off: mist track; knob paper-light r 7 | — | — | 40 % opacity | 36 × 20 |
| Select | paper-light, 1 px mist border, chevron | — | open: 1 px ink border, menu below | 40 % opacity | radius 8, height 40 |
| Field (path) | paper-light, 1 px mist border, Mono/Path | — | — | — | empty: stone placeholder; error: error border and text |
| ProgressBar | track mist, fill ink, 6 px | — | — | — | **progress**: fill width = value %; busy (no total yet): moving fill |
| ProgressBarGold | track stone, fill gold, 4 px | — | — | — | for dark surfaces (loading window, in-game badge) |
| CheckRow / StatusIcon | 20 px circle: success ✓, ochre !, error × (12 px glyph, paper-light) | — | — | — | |

## Not in the launcher yet (designed in Figma)

The loading window (dark and light, stages cache / shaders / start, gold bar) and the in-game shader badge (top-right,
margin 24 px at 1080p, 48 px at 2160p, translucent ink 88 %, Manrope + JetBrains Mono 12 px, 3 px gold bar): both are
drawn by the emulator. Aspect ratio, Ray tracing pool and "overlay starts compact" need emulator options first; "Share
shader hints" waits for precompile step 3. Their landscapes are already in `src/launcher/images`.

## Checking the look

`KYTY_LAUNCHER_SCREENSHOT_DIR=<folder>` makes the launcher save each page at 1280 × 800 (launcher-play.png,
launcher-play-hover.png when there are two or more games — the second tile under the mouse, launcher-display.png,
launcher-graphics.png, launcher-shaders.png, launcher-shaders-precompiling.png, launcher-advanced.png,
launcher-first-start.png) and quit. It looks for the emulator file as usual (so the Checks show it found or missing)
but never starts it; the Play picture waits for the selected game's picture. As always, a `Kyty.ini` in the working
folder is used instead of the user's settings, which is how a test list of games is shown. The precompiling picture
runs sample `PRECOMPILE_*` lines through the same parser as the emulator's output. It also writes a sample run log
with this PC's real paths into `bug-report-sample/`, shows the bug report dialog on it (launcher-bug-report.png) and
saves `bug-report-sample.zip`; when `<folder>/test-cache/_PipelineCache` exists (a copy made for the test), it shows
the Clear cache question (launcher-clear-cache.png), moves that copy's files to the Recycle Bin and writes
`clear-cache-test.txt`. The emulator's own cache is never touched in this mode.
