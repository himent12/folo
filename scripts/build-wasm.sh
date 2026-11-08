#!/bin/sh
# Builds folo-wasm (FOLO detectors in the browser, docs/WASM.md) into web/public/folo-wasm/: folo_wasm.js (ES module),
# folo_wasm_bg.wasm and their TypeScript declarations. Studio's build can run this before `vite build`.
#
#   sh scripts/build-wasm.sh                 # release build into web/public/folo-wasm
#   sh scripts/build-wasm.sh --out DIR       # somewhere else
#   sh scripts/build-wasm.sh --dev           # unoptimised, faster to build
#   sh scripts/build-wasm.sh --locked        # fail instead of updating Cargo.lock (release builds)
#
# Needs the wasm32-unknown-unknown target (added with rustup when missing) and wasm-bindgen-cli at exactly the
# version in Cargo.lock: on PATH, in $FOLO_TOOLS/bin, or installed with
#   cargo install wasm-bindgen-cli --version <version> --locked
# CARGO_TARGET_DIR is honoured. wasm-opt (binaryen) is run on the output when it is on PATH.
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
out="$root/web/public/folo-wasm"
profile=release
locked=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) out=$2; shift 2 ;;
    --dev) profile=dev; shift ;;
    --locked) locked=--locked; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "build-wasm.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

target=wasm32-unknown-unknown
if command -v rustup >/dev/null 2>&1 && ! rustup target list --installed | grep -qx "$target"; then
  rustup target add "$target"
fi

want=$(sed -n '/^name = "wasm-bindgen"$/{n;s/^version = "\(.*\)"$/\1/p;}' "$root/Cargo.lock")
bindgen=""
for c in wasm-bindgen "${FOLO_TOOLS:-.}/bin/wasm-bindgen"; do
  if command -v "$c" >/dev/null 2>&1; then bindgen=$c; break; fi
done
if [ -z "$bindgen" ]; then
  echo "build-wasm.sh: wasm-bindgen not found; install it with: cargo install wasm-bindgen-cli --version $want --locked" >&2
  exit 1
fi
have=$("$bindgen" --version | awk '{print $2}')
if [ "$have" != "$want" ]; then
  echo "build-wasm.sh: $bindgen is $have but Cargo.lock has wasm-bindgen $want; install the matching CLI:" >&2
  echo "  cargo install wasm-bindgen-cli --version $want --locked" >&2
  exit 1
fi

# An explicit RUSTFLAGS replaces .cargo/config.toml's target flags, so pass SIMD128 here as well.
RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128"
export RUSTFLAGS
target_dir=${CARGO_TARGET_DIR:-$root/target}
if [ "$profile" = release ]; then
  cargo build --manifest-path "$root/Cargo.toml" $locked --release --target "$target" -p folo-wasm
  wasm="$target_dir/$target/release/folo_wasm.wasm"
else
  cargo build --manifest-path "$root/Cargo.toml" $locked --target "$target" -p folo-wasm
  wasm="$target_dir/$target/debug/folo_wasm.wasm"
fi

mkdir -p "$out"
"$bindgen" --target web --out-dir "$out" "$wasm"
if command -v wasm-opt >/dev/null 2>&1 && [ "$profile" = release ]; then
  wasm-opt -O3 --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext \
    --enable-mutable-globals -o "$out/folo_wasm_bg.wasm" "$out/folo_wasm_bg.wasm"
fi
echo "folo-wasm ($profile) -> $out"
ls -l "$out"
