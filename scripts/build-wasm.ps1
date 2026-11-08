<#
.SYNOPSIS
Builds folo-wasm (FOLO detectors in the browser, docs/WASM.md) into web\public\folo-wasm.

.DESCRIPTION
Produces folo_wasm.js (an ES module), folo_wasm_bg.wasm and their TypeScript declarations. Studio's build can run
this before `vite build`. Needs the wasm32-unknown-unknown target (added with rustup when missing) and
wasm-bindgen-cli at exactly the version in Cargo.lock: on PATH, in $env:FOLO_TOOLS\bin, or installed with
`cargo install wasm-bindgen-cli --version <version> --locked`. CARGO_TARGET_DIR is honoured. wasm-opt (binaryen) is
run on the output when it is on PATH. Works in Windows PowerShell 5.1 and PowerShell 7.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\build-wasm.ps1

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\build-wasm.ps1 -Out D:\tmp\folo-wasm -Dev
#>
[CmdletBinding()]
param(
    # Output directory [default: web\public\folo-wasm].
    [string]$Out,
    # Unoptimised build (faster to compile).
    [switch]$Dev,
    # Fail instead of updating Cargo.lock (release builds).
    [switch]$Locked
)
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
if (-not $Out) { $Out = Join-Path $root 'web\public\folo-wasm' }
$target = 'wasm32-unknown-unknown'

if (Get-Command rustup -ErrorAction SilentlyContinue) {
    $installed = & rustup target list --installed
    if ($installed -notcontains $target) { & rustup target add $target; if ($LASTEXITCODE) { exit $LASTEXITCODE } }
}

$lock = Get-Content -Raw (Join-Path $root 'Cargo.lock')
$m = [regex]::Match($lock, 'name = "wasm-bindgen"\r?\nversion = "([^"]+)"')
if (-not $m.Success) { throw 'build-wasm.ps1: wasm-bindgen is not in Cargo.lock' }
$want = $m.Groups[1].Value

$candidates = @('wasm-bindgen')
if ($env:FOLO_TOOLS) { $candidates += (Join-Path $env:FOLO_TOOLS 'bin\wasm-bindgen.exe') }
$bindgen = $null
foreach ($c in $candidates) {
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if ($cmd) { $bindgen = $cmd.Source; break }
}
if (-not $bindgen) {
    throw "build-wasm.ps1: wasm-bindgen not found; install it with: cargo install wasm-bindgen-cli --version $want --locked"
}
$have = ((& $bindgen --version) -split ' ')[1]
if ($have -ne $want) {
    throw "build-wasm.ps1: $bindgen is $have but Cargo.lock has wasm-bindgen $want; install the matching CLI: cargo install wasm-bindgen-cli --version $want --locked"
}

# An explicit RUSTFLAGS replaces .cargo/config.toml's target flags, so pass SIMD128 here as well.
$env:RUSTFLAGS = (($env:RUSTFLAGS, '-C target-feature=+simd128') -join ' ').Trim()
$targetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $root 'target' }
$manifest = Join-Path $root 'Cargo.toml'
$lockedArg = @(if ($Locked) { '--locked' })
if ($Dev) {
    & cargo build --manifest-path $manifest @lockedArg --target $target -p folo-wasm
    $wasm = Join-Path $targetDir "$target\debug\folo_wasm.wasm"
} else {
    & cargo build --manifest-path $manifest @lockedArg --release --target $target -p folo-wasm
    $wasm = Join-Path $targetDir "$target\release\folo_wasm.wasm"
}
if ($LASTEXITCODE) { exit $LASTEXITCODE }

New-Item -ItemType Directory -Force $Out | Out-Null
& $bindgen --target web --out-dir $Out $wasm
if ($LASTEXITCODE) { exit $LASTEXITCODE }
if ((-not $Dev) -and (Get-Command wasm-opt -ErrorAction SilentlyContinue)) {
    $bg = Join-Path $Out 'folo_wasm_bg.wasm'
    & wasm-opt -O3 --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext `
        --enable-mutable-globals -o $bg $bg
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
}
$kind = if ($Dev) { 'dev' } else { 'release' }
Write-Host "folo-wasm ($kind) -> $Out"
Get-ChildItem $Out | Format-Table Name, Length -AutoSize
