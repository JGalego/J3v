#!/bin/sh
# J3v installer: downloads the static j3v binary for this machine from GitHub releases.
#
#   curl -fsSL https://raw.githubusercontent.com/JGalego/J3v/main/install.sh | sh
#
# Environment:
#   J3V_VERSION=v0.1.1     release to install (default: latest)
#   J3V_INSTALL_DIR=DIR    where to put the binary (default: /usr/local/bin if writable or sudo works, else ~/.local/bin)
#   J3V_MODELS=DIR         also download the shared encoder and the example artifacts into DIR
set -eu

REPO="JGalego/J3v"
say() { printf 'j3v-install: %s\n' "$*" >&2; }
die() { say "error: $*"; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "'$1' is required"; }

need curl
need tar
need uname

[ "$(uname -s)" = "Linux" ] || die "only Linux is supported (got $(uname -s)); build from source: cargo install --path crates/j3v"

arch="${J3V_ARCH:-$(uname -m)}"
case "$arch" in
  x86_64 | amd64) target="x86_64-unknown-linux-musl" ;;
  aarch64 | arm64) target="aarch64-unknown-linux-musl" ;;
  armv7l | armv8l) target="armv7-unknown-linux-musleabihf" ;;
  armv6l) die "ARMv6 (Pi 1 / original Pi Zero) has no prebuilt binary yet" ;;
  *) die "no prebuilt binary for '$arch'" ;;
esac

version="${J3V_VERSION:-}"
if [ -z "$version" ]; then
  # the /releases/latest page redirects to /releases/tag/<version>
  version=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/$REPO/releases/latest" | sed 's|.*/tag/||')
  [ -n "$version" ] || die "could not determine the latest release"
fi
base="https://github.com/$REPO/releases/download/$version"
name="j3v-$version-$target"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT INT TERM

say "downloading $name ($version)"
curl -fsSL "$base/$name.tar.gz" -o "$tmp/$name.tar.gz" || die "download failed: $base/$name.tar.gz"
curl -fsSL "$base/SHA256SUMS" -o "$tmp/SHA256SUMS" || die "download failed: $base/SHA256SUMS"

want=$(grep " $name.tar.gz\$" "$tmp/SHA256SUMS" | cut -d' ' -f1)
[ -n "$want" ] || die "$name.tar.gz is not listed in SHA256SUMS"
if command -v sha256sum >/dev/null 2>&1; then got=$(sha256sum "$tmp/$name.tar.gz" | cut -d' ' -f1)
else got=$(shasum -a 256 "$tmp/$name.tar.gz" | cut -d' ' -f1); fi
[ "$want" = "$got" ] || die "checksum mismatch for $name.tar.gz"

tar xzf "$tmp/$name.tar.gz" -C "$tmp"

dir="${J3V_INSTALL_DIR:-}"
sudo=""
if [ -z "$dir" ]; then
  # sudo -v reads the password from the terminal even when this script is piped into sh
  # shellcheck disable=SC2024
  if [ -w /usr/local/bin ]; then
    dir=/usr/local/bin
  elif command -v sudo >/dev/null 2>&1 && { sudo -n true 2>/dev/null || (sudo -v </dev/tty) 2>/dev/null; }; then
    dir=/usr/local/bin
    sudo=sudo
  else
    dir="$HOME/.local/bin"
  fi
fi
$sudo mkdir -p "$dir"
$sudo install -m 0755 "$tmp/$name/j3v" "$dir/j3v"
say "installed $("$dir/j3v" --version 2>/dev/null || echo "j3v $version") to $dir/j3v"
case ":$PATH:" in *":$dir:"*) ;; *) say "note: $dir is not on your PATH; add it, e.g. export PATH=\"$dir:\$PATH\"" ;; esac

if [ -n "${J3V_MODELS:-}" ]; then
  mkdir -p "$J3V_MODELS"
  for f in minilm.j3a support_triage.pi.j3a support_triage.mcu.j3a; do
    say "downloading $f"
    curl -fsSL "$base/$f" -o "$J3V_MODELS/$f" || die "download failed: $base/$f"
    want=$(grep " $f\$" "$tmp/SHA256SUMS" | cut -d' ' -f1)
    if command -v sha256sum >/dev/null 2>&1; then got=$(sha256sum "$J3V_MODELS/$f" | cut -d' ' -f1)
    else got=$(shasum -a 256 "$J3V_MODELS/$f" | cut -d' ' -f1); fi
    [ "$want" = "$got" ] || die "checksum mismatch for $f"
  done
  say "models in $J3V_MODELS; try:"
  say "  j3v predict --encoder $J3V_MODELS/minilm.j3a $J3V_MODELS/support_triage.pi.j3a '{\"state\": {\"message\": \"refund me\"}}'"
fi
