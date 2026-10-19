# Unmixing the Class Band

A small in-browser ICA demo for music class: a tiny class band (piano, drums and bass) plays "Twinkle Twinkle Little Star", three microphones around the music room each record a blend of all the instruments, and Independent Component Analysis (FastICA) pulls each instrument's own track back out using only the mic recordings. Made for K-12 learners, in vanilla JavaScript with no dependencies.

**Live demo:** https://lucylyu.surge.sh/vibe/ica/

## How to run

Open `index.html` in a browser, or serve the repo root with any static file server (for example `python3 -m http.server`). All sounds are synthesized in the browser; nothing is downloaded.

To check the FastICA math outside the browser:

```bash
node test/ica-test.js
```
