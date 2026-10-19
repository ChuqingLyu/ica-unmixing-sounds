# Unmixing School Sounds

A small ICA (Independent Component Analysis) project for K-12 learners: kids collect sounds around school, the computer mixes them as if several microphones in one room heard everything at once, and ICA pulls them apart again using only the mixed recordings.

**Live page:** https://lucylyu.surge.sh/vibe/ica/

Vanilla JavaScript, no dependencies. Everything runs in the browser.

## What's on the page

- **The ICA visualizer** (2 sounds, 2 mics). Each dot is one moment of time. Step through: the sounds as dots → mixed up (2×2 sliders) → move to the middle (centering) → make it round (whitening) → turn the dial (rotation slider with a live independence meter, histograms of the two outputs, and Play buttons). "Let the computer turn it" runs 2D FastICA round by round. Honors `prefers-reduced-motion`.
- **The full demo** (3 sounds, 3 mics): play the sounds, edit or randomize the 3×3 loudness table, press Separate, and see which sound each recovered track matches.
- **Behind the scenes**: figures from a real FastICA run on the example sounds with the page's default mix.

URL options: `?autorun=1` presses Separate once the sounds load; `?vizstep=4` opens the visualizer at step 4 (`&vizangle=30` sets the dial).

## Sounds

Right now the page uses **example sounds**: openly licensed recordings standing in until our classroom recordings are added. They were not recorded by the kids. Each is a 9-second excerpt, mono, 16 kHz, loudness-normalized, 40 kbps MP3. Full details are in `audio/credits.json`.

| File | Stands in for | Source | Author | License |
|---|---|---|---|---|
| `audio/kitchen.mp3` | school kitchen / canteen | [Kitchen tidy up.ogg](https://commons.wikimedia.org/wiki/File:Kitchen_tidy_up.ogg) (0:22–0:31) | starlite (via PDSounds) | Public domain |
| `audio/hall.mp3` | hall / corridor | [Koulu-ambience1.wav](https://commons.wikimedia.org/wiki/File:Koulu-ambience1.wav) (0:05–0:14), a school hallway in Helsinki | Suihkuturbiini8k | CC0 1.0 |
| `audio/classroom.mp3` | classroom (reading aloud) | [The Aesop for Children, section 32](https://librivox.org/the-aesop-for-children-by-aesop/) ([file](https://archive.org/download/aesopforchildren_1308_librivox/theaesopforchildren_32_aesop_64kb.mp3), 0:40–0:49) | a LibriVox volunteer | Public domain |

To rebuild the clips, download the three originals into a folder and run `tools/make_audio.sh <folder>` (needs ffmpeg). It also writes `test/sources-16k.s16`, the decoded clips used by the test and the figure script.

## Adding the kids' recordings

Open `ica.js` and fill in `CLASSROOM_CLIPS` at the top:

```js
var CLASSROOM_CLIPS = [
  { label: 'Kitchen (Class 4B)',   place: 'kitchen',   src: 'audio/kids/kitchen-4b.mp3' },
  { label: 'Hall (Class 4B)',      place: 'hall',      src: 'audio/kids/hall-4b.mp3' },
  { label: 'Classroom (Class 4B)', place: 'classroom', src: 'audio/kids/classroom-4b.mp3' }
];
```

- `place` is `kitchen`, `hall` or `classroom`. While a place has no recording, the page shows an empty slot such as "[Kids' recording from the kitchen — to be added]".
- Each recording is cut or padded to 9 seconds and resampled to 16 kHz in the browser.
- All recordings appear in the visualizer's sound lists. With 3 or more, the full demo uses the first three instead of the examples.
- Please get permission before putting children's recordings online, and avoid recording names or faces.

## Run it

Serve the folder with any static server (the sounds are loaded with `fetch`, so opening the file directly may not work):

```bash
python3 -m http.server 8000   # then open http://localhost:8000/
```

## Test

```bash
node test/ica-test.js
```

Runs FastICA from `ica.js` on 12 random 3×3 mixes and 18 random 2×2 mixes of the example sounds and checks that every output matches a different sound with |r| > 0.95. The latest output is in `process/separation-test.txt`.

## Regenerate the figures

```bash
python3 tools/make_figures.py   # needs numpy and matplotlib
```

This is a numpy port of `fastICA()` in `ica.js` (same random start, centering, whitening, update and stopping rule), run on the page's default 3×3 mix. It writes `process/convergence.png`, `process/track-snapshots.png`, `process/unmix-times-mix.png` and `process/training-log.json`. On the current example sounds it stops after 7 rounds, the same as the page.

## License

Code: MIT (see `LICENSE`). Sounds: see the table above.
