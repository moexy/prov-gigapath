#!/bin/bash
# Download the manifest-selected CAMELYON17 slides with resume + MD5 verification.
# Run on a login node (transfers only). Re-running skips slides that already verify.
set -uo pipefail

MANIFEST=${1:?Usage: fetch_camelyon17.sh MANIFEST DEST}
DEST=${2:?Usage: fetch_camelyon17.sh MANIFEST DEST}
mkdir -p "$DEST"

rc=0
while IFS=$'\t' read -r slide centre label size md5 url; do
    [[ $slide == slide_id ]] && continue
    target=$DEST/$slide.tif
    if [[ -f $target ]] && echo "$md5  $target" | md5sum -c --status -; then
        continue
    fi
    echo "[$(date -u +%FT%TZ)] fetching $slide (centre $centre, $label)"
    curl -fsS -C - -o "$target" "$url" || { echo "download failed: $slide" >&2; rc=1; continue; }
    if ! echo "$md5  $target" | md5sum -c --status -; then
        echo "MD5 mismatch: $slide" >&2
        rm -f "$target"
        rc=1
    fi
done < "$MANIFEST"
exit "$rc"
