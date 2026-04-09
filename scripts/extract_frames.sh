#!/bin/bash
# Usage: ./scripts/extract_frames.sh <youtube_url> <output_dir>
set -e
URL=$1; OUTDIR=$2
mkdir -p "$OUTDIR"
yt-dlp -o "/tmp/algerplate_video.%(ext)s" "$URL"
ffmpeg -i /tmp/algerplate_video.* -vf "fps=0.5" \
  "$OUTDIR/frame_%04d.jpg" -q:v 2 -loglevel warning
rm -f /tmp/algerplate_video.*
echo "Done. Frames in $OUTDIR"
