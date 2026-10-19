# Unmixing Sounds

A small in-browser FastICA demo of the cocktail party problem, aimed at K-12 learners: three synthesized sources are mixed at virtual microphones, then separated again with Independent Component Analysis—all in vanilla JavaScript with no dependencies.

**Live demo:** https://lucylyu.surge.sh/vibe/ica/

## How to run

Open `index.html` in a browser, or serve the repo root with any static file server (for example `python3 -m http.server`). All sounds are synthesized in the browser; nothing is downloaded.

To check the FastICA math outside the browser:

```bash
node test/ica-test.js
```
