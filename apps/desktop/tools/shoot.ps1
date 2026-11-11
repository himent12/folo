# Render FOLO Desktop in a scenario to a PNG, headless: `folo-desktop --scenario ... --shot <png>` runs the app
# on folo-ui's headless platform (no window, no cursor, no focus), settles it on a fake clock and writes the
# frame. It takes well under a second.
# Usage: tools\shoot.ps1 -Scenario run-live -Out shots\r1\run-live.png [-Width 1280 -Height 800] [-Scale 1.5]
#        [-Exe path] [-ExtraArgs "--server http://127.0.0.1:7878 --open run:run_9d9d"]
# The PNG is Width*Scale x Height*Scale pixels. -Scale defaults to 1.5, the scale of the dev PC's display, at
# which every design review has been shot. -Scenario none passes no --scenario: give -ExtraArgs a source then
# (`--server URL` or `--mock`), or the app looks for, and may start, a local folo serve as a normal launch does.
param(
    [Parameter(Mandatory = $true)][string]$Scenario,
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Width = 1280,
    [int]$Height = 800,
    [double]$Scale = 1.5,
    [string]$Exe = "",
    # Extra arguments for the app, e.g. "--server http://127.0.0.1:7878 --open run:run_9d9d".
    [string]$ExtraArgs = ""
)

$ErrorActionPreference = "Stop"
if (-not $Exe) {
    $target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $PSScriptRoot "../target" }
    $Exe = Join-Path $target "debug/folo-desktop.exe"
}
$Exe = (Resolve-Path $Exe).Path
$dir = Split-Path -Parent $Out
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }
$Out = [System.IO.Path]::GetFullPath($Out)

$scaleText = $Scale.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$argList = @("--width", $Width, "--height", $Height, "--scale", $scaleText, "--shot", $Out)
if ($Scenario -ne "none") { $argList = @("--scenario", $Scenario) + $argList }
if ($ExtraArgs) { $argList += $ExtraArgs.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries) }
& $Exe @argList
if ($LASTEXITCODE -ne 0) { throw "folo-desktop --shot failed for $Scenario (exit $LASTEXITCODE)" }
Add-Type -AssemblyName System.Drawing
$img = [System.Drawing.Image]::FromFile($Out)
try { "$Out $($img.Width)x$($img.Height)" } finally { $img.Dispose() }
