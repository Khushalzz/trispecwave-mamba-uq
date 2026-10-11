#!/usr/bin/env bash
set -euo pipefail

destination="${1:-data/sen2naip}"
url="${2:-}"
mkdir -p "$destination"
if [[ -z "$url" ]]; then
  echo "Usage: $0 OUTPUT_DIRECTORY PUBLIC_ARCHIVE_URL" >&2
  echo "Provide the dataset URL after checking its licence and access terms." >&2
  exit 2
fi
archive="$destination/$(basename "${url%%\?*}")"
curl -L --fail --retry 3 "$url" -o "$archive"
echo "Downloaded $archive; extract it manually or add an archive-specific extraction step."

