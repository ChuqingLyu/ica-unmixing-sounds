# Unmixing School Sounds

A small ICA (Independent Component Analysis) project for K-12 learners: kids collect everyday sounds from home and the neighborhood, the computer mixes them as if several microphones in one room heard everything at once, and ICA pulls them apart again using only the mixed recordings.

**Live page:** https://lucylyu.surge.sh/vibe/ica/

## The activity

I ran this as a two-week summer-vacation project assignment: about 6 Grade 7–8 students from Bocuide School, Guangzhou recorded sounds from their neighborhoods, homes, and communities with boom-pole microphones. The page then lets learners mix and unmix sounds hands-on, during both recording and analysis.

Vanilla JavaScript, no dependencies. Everything runs in the browser.

## What's on the page

- **The ICA visualizer** (2 sounds, 2 mics). Each dot is one moment of time. Step through: the sounds as dots → mixed up (2×2 sliders) → move to the middle (centering) → make it round (whitening) → turn the dial (rotation slider with a live independence meter, histograms of the two outputs, and Play buttons). "Let the computer turn it" runs 2D FastICA round by round. Honors `prefers-reduced-motion`.
- **The full demo** (3 sounds, 3 mics): play the sounds, edit or randomize the 3×3 loudness table, then:
  - **You try first:** three rotation dials on the whitened mic recordings, with a live independence meter and Play buttons; your best score is compared with the computer's afterwards (same measure, same moments).
  - **Analysis lab:** Separate records every FastICA round. Play/Pause, step ◀ ▶ or scrub through rounds; the three outputs morph between rounds, an independence-score chart builds up, each output shows live match bars to every source, and Play works at any round. Spectrograms (in-browser FFT/STFT) of mic 1 vs. each output at the current round, plus a drag slider comparing the mix with a separated output.
  - **Which is which?:** tap Kitchen / Hall / Classroom for each final track, then check against the correlation-based match; tracks take their matched colors.
- **Student recordings & submissions**: cards filled from `CLASSROOM_CLIPS` (audio) and `STUDENT_SUBMISSIONS` (images); placeholders while empty.
- **Behind the scenes**: figures from a real FastICA run on the example sounds with the page's default mix.

**Look:** a dark "sound lab" theme with system fonts only, no external assets. Each sound keeps one color everywhere (Kitchen amber, Hall teal, Classroom pink; mic mixes are gray), and each recovered track takes the color of the sound it matches. The animated hero shows a mixed signal splitting into the three sounds. Every Play button opens a live oscilloscope and spectrum (Web Audio `AnalyserNode`) in that track's color, with a playhead moving across the waveform. With `prefers-reduced-motion`, the hero shows its separated state and sections appear without the scroll animation.

URL options: `?autorun=1` presses Separate once the sounds load (and reveals the matches); `?labround=3` runs Separate and opens the Analysis lab at round 3; `?vizstep=4` opens the visualizer at step 4 (`&vizangle=30` sets the dial).

## Sounds

Right now the demo uses openly licensed recordings; students' own recordings will replace them. They were not recorded by the students. Each is a 9-second excerpt, mono, 16 kHz, loudness-normalized, 40 kbps MP3. Full details are in `audio/credits.json`.

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

- `place` is `kitchen`, `hall` or `classroom`. Each entry renders an audio card in "Student recordings & submissions"; while the list is empty, placeholder cards are shown.
- Each recording is cut or padded to 9 seconds and resampled to 16 kHz in the browser.
- All recordings appear in the visualizer's sound lists. With 3 or more, the full demo uses the first three instead of the examples.
- Please get permission before putting children's recordings online, and avoid recording names or faces.

## Adding students' submitted work

Fill in `STUDENT_SUBMISSIONS` in `ica.js` to show image cards (e.g. screenshots of submitted work):

```js
var STUDENT_SUBMISSIONS = [
  { title: 'Sound map (Student A)', src: 'submissions/sound-map-a.png', caption: 'A short caption' }
];
```

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

This is a numpy port of `fastICA()` in `ica.js` (same random start, centering, whitening, update and stopping rule), run on the page's default 3×3 mix. It writes `process/convergence.png`, `process/track-snapshots.png`, `process/unmix-times-mix.png` (dark theme, same colors as the page) and `process/training-log.json`. On the current example sounds it stops after 7 rounds, the same as the page.

## License

Code: MIT (see `LICENSE`). Sounds: see the table above.

## Screenshots

`screenshots/ica-v6-1280-top.png` (top of page), `screenshots/ica-v6-lab.png` (Analysis lab at round 3), `screenshots/ica-v6-lab-final.png` (full demo after separation).
