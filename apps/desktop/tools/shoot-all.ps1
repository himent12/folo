# Capture every review scenario (the list comes from `folo-desktop --list-scenarios`) into a folder,
# plus the default and minimum window sizes for the densest screens.
# Usage: tools\shoot-all.ps1 -Dir shots\round1 [-Exe path] [-Only overview,run-live]
param(
    [Parameter(Mandatory = $true)][string]$Dir,
    [string]$Exe = "",
    [string[]]$Only = @()
)

$ErrorActionPreference = "Stop"
if (-not $Exe) {
    $target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $PSScriptRoot "../target" }
    $Exe = Join-Path $target "debug/folo-desktop.exe"
}
$scenarios = & $Exe --list-scenarios
if ($Only.Count -gt 0) { $scenarios = $scenarios | Where-Object { $Only -contains $_ } }
# Warm-up launch (shader and font caches), so the first real capture isn't caught mid-load.
& "$PSScriptRoot\shoot.ps1" -Scenario ($scenarios | Select-Object -First 1) -Exe $Exe -Out (Join-Path $env:TEMP "folo-warmup.png")
# A busy machine can make a window slow to appear: retry a capture once before giving up.
function Shoot([string]$s, [hashtable]$more) {
    try { & "$PSScriptRoot\shoot.ps1" -Scenario $s -Exe $Exe @more }
    catch { Write-Warning "retrying ${s}: $_"; & "$PSScriptRoot\shoot.ps1" -Scenario $s -Exe $Exe @more }
}
foreach ($s in $scenarios) {
    $name = $s -replace '\+', '-'
    Shoot $s @{ Out = (Join-Path $Dir "$name.png") }
}
if ($Only.Count -eq 0) {
    # Minimum window size (960 x 620).
    foreach ($s in @("overview", "overview+empty", "run-live", "dataset", "playground", "models", "new-run")) {
        Shoot $s @{ Width = 960; Height = 620; Out = (Join-Path $Dir "$($s -replace '\+', '-')-min.png") }
    }
    # Large window (1680 x 1050).
    foreach ($s in @("overview", "overview+empty", "run-live", "model")) {
        Shoot $s @{ Width = 1680; Height = 1050; Out = (Join-Path $Dir "$($s -replace '\+', '-')-wide.png") }
    }
}
