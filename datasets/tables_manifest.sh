#!/usr/bin/env bash
# The MIMIC-IV tables live outside the checkout (docs/tickets/mimic/01-external-datasets-root.md), so git cannot say
# when they change or whether another machine's copy is current. This keeps their SHA-256 list in two places: in the
# data root (SHA256SUMS, copied with the tables) and in the checkout (datasets/external_tables.sha256, tracked).
#   bash datasets/tables_manifest.sh write   # where the tables were built or changed; then commit and push
#   bash datasets/tables_manifest.sh check   # anywhere, after a pull: fails if the tables differ from the tracked list
# A failed check on a machine that did not build them means: rsync the data root from the one that did.
set -eu
: "${TRIDENT_DATASETS_ROOT:?set TRIDENT_DATASETS_ROOT}"
LIST="$(cd "$(dirname "$0")" && pwd)/external_tables.sha256"
case "${1:-}" in
  write)
    (cd "$TRIDENT_DATASETS_ROOT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum) > "$LIST"
    cp "$LIST" "$TRIDENT_DATASETS_ROOT/SHA256SUMS"
    echo "$(wc -l < "$LIST") files listed in $LIST and $TRIDENT_DATASETS_ROOT/SHA256SUMS" ;;
  check)
    (cd "$TRIDENT_DATASETS_ROOT" && sha256sum --quiet -c "$LIST") && echo "the tables match $(basename "$LIST")" ;;
  *) echo "usage: $0 write|check" >&2; exit 2 ;;
esac
