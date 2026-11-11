# Render every review scenario (the list comes from `folo-desktop --list-scenarios`) into a folder, plus the
# minimum (960 x 620) and large (1680 x 1050) window sizes for the densest screens. Headless (tools\shoot.ps1):
# no window opens; the whole set takes about a minute.
# Usage: tools\shoot-all.ps1 -Dir shots\round1 [-Exe path] [-Scale 1.5] [-Only overview,run-live]
param(
    [Parameter(Mandatory = $true)][string]$Dir,
    [string]$Exe = "",
    [double]$Scale = 1.5,
    [string[]]$Only = @()
)

$ErrorActionPreference = "Stop"
if (-not $Exe) {
    $target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $PSScriptRoot "../target" }
    $Exe = Join-Path $target "debug/folo-desktop.exe"
}
$Exe = (Resolve-Path $Exe).Path
$scenarios = @(& $Exe --list-scenarios | Where-Object { $_ })
if ($Only.Count -gt 0) { $scenarios = @($scenarios | Where-Object { $Only -contains $_ }) }
$failed = @()
function Shoot([string]$s, [hashtable]$more) {
    try { & "$PSScriptRoot\shoot.ps1" -Scenario $s -Exe $Exe -Scale $Scale @more }
    catch { Write-Warning "${s}: $_"; $script:failed += $s }
}
foreach ($s in $scenarios) {
    $name = $s -replace '\+', '-'
    Shoot $s @{ Out = (Join-Path $Dir "$name.png") }
}
if ($Only.Count -eq 0) {
    # Minimum window size (960 x 620).
    foreach ($s in @("overview", "overview+empty", "run-live", "run-live+charts-focus", "dataset", "playground", "models", "new-run", "settings+cloud", "settings+cloud+trouble", "cloud-connect", "cloud-connect+remote", "new-run+cloud", "run-cloud+syncing", "run-cloud+training")) {
        Shoot $s @{ Width = 960; Height = 620; Out = (Join-Path $Dir "$($s -replace '\+', '-')-min.png") }
    }
    # Large window (1680 x 1050).
    foreach ($s in @("overview", "overview+empty", "run-live", "model", "settings+cloud", "run-cloud+training")) {
        Shoot $s @{ Width = 1680; Height = 1050; Out = (Join-Path $Dir "$($s -replace '\+', '-')-wide.png") }
    }
}
if ($failed.Count) { throw "$($failed.Count) shots failed: $($failed -join ', ')" }
