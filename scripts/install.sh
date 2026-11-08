#!/bin/sh
# Installs the `folo` CLI from a GitHub release: picks the archive for this platform, verifies its SHA-256 and
# copies the binary into a user bin directory. The binary embeds FOLO Studio, so `folo serve` works right away.
#
#   sh scripts/install.sh                       # latest release into ~/.local/bin
#   sh scripts/install.sh --version v0.2.0 --dir ~/bin
#
# Options (or environment variables):
#   --version TAG    FOLO_VERSION      release tag, e.g. v0.2.0 [default: latest]
#   --dir DIR        FOLO_INSTALL_DIR  where to put `folo` [default: ~/.local/bin]
#   --repo O/R       FOLO_REPO         GitHub repository [default: himent12/folo]
#   --base-url URL   FOLO_BASE_URL     download <URL>/<asset> instead of GitHub (a mirror, or a local folder)
#   --target TRIPLE  FOLO_TARGET       override the detected platform, e.g. x86_64-unknown-linux-gnu
# Private repositories need a token: GH_TOKEN or GITHUB_TOKEN, else the one from `gh auth token` when gh is logged in.
set -eu

usage() {
  cat <<'EOF_USAGE'
usage: install.sh [--version TAG] [--dir DIR] [--repo OWNER/NAME] [--base-url URL] [--target TRIPLE]
Installs the folo CLI from a GitHub release (default: the latest into ~/.local/bin).
Environment: FOLO_VERSION, FOLO_INSTALL_DIR, FOLO_REPO, FOLO_BASE_URL, FOLO_TARGET, GH_TOKEN / GITHUB_TOKEN.
EOF_USAGE
}

repo=${FOLO_REPO:-himent12/folo}
version=${FOLO_VERSION:-latest}
dir=${FOLO_INSTALL_DIR:-${HOME:?HOME is not set}/.local/bin}
base_url=${FOLO_BASE_URL:-}
target=${FOLO_TARGET:-}

say() { printf 'folo-install: %s\n' "$*" >&2; }
die() { say "error: $*"; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --version) version=${2:?--version needs a tag}; shift 2 ;;
    --dir) dir=${2:?--dir needs a directory}; shift 2 ;;
    --repo) repo=${2:?--repo needs owner/name}; shift 2 ;;
    --base-url) base_url=${2:?--base-url needs a URL}; shift 2 ;;
    --target) target=${2:?--target needs a target triple}; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done

if [ -z "$target" ]; then
  case "$(uname -s)/$(uname -m)" in
    Linux/x86_64 | Linux/amd64) target=x86_64-unknown-linux-gnu ;;
    Darwin/arm64 | Darwin/aarch64) target=aarch64-apple-darwin ;;
    MINGW* | MSYS* | CYGWIN*) die "on Windows use scripts/install.ps1" ;;
    *) die "no prebuilt folo for $(uname -s) $(uname -m) yet; build from source with cargo (see README.md)" ;;
  esac
fi
asset="folo-$target.tar.gz"

command -v curl >/dev/null || die "curl is required"
command -v tar >/dev/null || die "tar is required"
if command -v sha256sum >/dev/null; then
  sha256() { sha256sum "$1" | cut -d ' ' -f 1; }
elif command -v shasum >/dev/null; then
  sha256() { shasum -a 256 "$1" | cut -d ' ' -f 1; }
else
  die "sha256sum or shasum is required to verify the download"
fi

tmp=$(mktemp -d 2>/dev/null || mktemp -d -t folo-install)
trap 'rm -rf "$tmp"' EXIT INT TERM

token=${GH_TOKEN:-${GITHUB_TOKEN:-}}
if [ -z "$token" ] && [ -z "$base_url" ] && command -v gh >/dev/null; then
  token=$(gh auth token 2>/dev/null || true)
fi

# fetch NAME DEST: downloads one release asset.
if [ -n "$base_url" ]; then
  fetch() {
    case "$base_url" in
      http://* | https://* | file://*) curl -fsSL --proto '=https,http,file' -o "$2" "$base_url/$1" ;;
      *) cp "$base_url/$1" "$2" ;;
    esac
  }
  from=$base_url
elif [ -n "$token" ]; then
  # The REST API serves assets of private repositories too: find each asset's API URL in the release JSON.
  api="https://api.github.com/repos/$repo/releases"
  if [ "$version" = latest ]; then api="$api/latest"; else api="$api/tags/$version"; fi
  curl -fsSL -H "Authorization: Bearer $token" -H "Accept: application/vnd.github+json" -o "$tmp/release.json" "$api" ||
    die "could not read release $version of $repo (is it published, and can this token read the repository?)"
  # Split the JSON at braces so every asset's own "url" and "name" land on one line.
  # shellcheck disable=SC2020 # both braces become newlines (equal-length sets are the portable spelling)
  tr -d '\n' < "$tmp/release.json" | tr '{}' '\n\n' | sed 's/" *: *"/":"/g' > "$tmp/release.lines"
  version=$(sed -n 's/.*"tag_name":"\([^"]*\)".*/\1/p' "$tmp/release.lines" | head -n 1)
  fetch() {
    url=$(grep -F "\"name\":\"$1\"" "$tmp/release.lines" | sed -n 's/.*"url":"\([^"]*\/releases\/assets\/[0-9]*\)".*/\1/p' | head -n 1)
    [ -n "$url" ] || die "release $version of $repo has no asset $1"
    # curl drops the Authorization header when the asset redirects to another host.
    curl -fsSL -H "Authorization: Bearer $token" -H "Accept: application/octet-stream" -o "$2" "$url"
  }
  from="github.com/$repo ($version)"
else
  if [ "$version" = latest ]; then
    dl="https://github.com/$repo/releases/latest/download"
  else
    dl="https://github.com/$repo/releases/download/$version"
  fi
  fetch() {
    curl -fsSL -o "$2" "$dl/$1" ||
      die "download of $dl/$1 failed (a private repository needs GH_TOKEN or a logged-in gh)"
  }
  from="github.com/$repo ($version)"
fi

say "downloading $asset from $from"
fetch "$asset" "$tmp/$asset"
fetch "$asset.sha256" "$tmp/$asset.sha256"

want=$(cut -d ' ' -f 1 < "$tmp/$asset.sha256" | tr 'A-F' 'a-f')
got=$(sha256 "$tmp/$asset")
[ -n "$want" ] && [ "$want" = "$got" ] || die "checksum mismatch for $asset: expected $want, got $got"
say "checksum ok ($got)"

tar -xzf "$tmp/$asset" -C "$tmp"
bin="$tmp/folo-$target/folo"
[ -f "$bin" ] || die "$asset does not contain folo-$target/folo"

mkdir -p "$dir"
# Copy next to the destination, then rename: an update never leaves a half-written binary behind.
cp "$bin" "$dir/.folo.tmp.$$"
chmod 755 "$dir/.folo.tmp.$$"
mv -f "$dir/.folo.tmp.$$" "$dir/folo"
say "installed $dir/folo"
"$dir/folo" --version >&2 || say "warning: $dir/folo --version failed"

case ":$PATH:" in
  *":$dir:"*) ;;
  *) say "add $dir to your PATH, e.g. echo 'export PATH=\"$dir:\$PATH\"' >> ~/.profile" ;;
esac
