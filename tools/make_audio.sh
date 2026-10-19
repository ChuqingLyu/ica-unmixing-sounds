#!/usr/bin/env bash
# Rebuild the three example sounds from the original openly licensed files.
# Usage: tools/make_audio.sh <folder with the downloaded originals>
# Originals (see audio/credits.json for full credits):
#   kitchen_src.ogg   = https://upload.wikimedia.org/wikipedia/commons/e/eb/Kitchen_tidy_up.ogg
#   hall_src.wav      = https://upload.wikimedia.org/wikipedia/commons/0/0d/Koulu-ambience1.wav
#   classroom_src.mp3 = https://archive.org/download/aesopforchildren_1308_librivox/theaesopforchildren_32_aesop_64kb.mp3
set -euo pipefail
SRC="${1:-.}"; OUT="$(dirname "$0")/../audio"; DUR=9
clip() { # name file start
  ffmpeg -v error -y -ss "$3" -t "$DUR" -i "$SRC/$2" -ac 1 \
    -af "highpass=f=60,afade=t=in:d=0.05,afade=t=out:st=$((DUR-1)).95:d=0.05,loudnorm=I=-20:TP=-2:LRA=11" \
    -ar 16000 -c:a libmp3lame -b:a 40k "$OUT/$1.mp3"
}
clip kitchen   kitchen_src.ogg   22
clip hall      hall_src.wav      5
clip classroom classroom_src.mp3 40
# Decoded copy for the Node test and the figure script (16 kHz, 16-bit, 3 x 144000 samples).
for n in kitchen hall classroom; do
  ffmpeg -v error -y -i "$OUT/$n.mp3" -ac 1 -ar 16000 -f s16le -t "$DUR" - ; done > "$(dirname "$0")/../test/sources-16k.s16"
ls -l "$OUT" "$(dirname "$0")/../test/sources-16k.s16"
