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
foreach ($s in $scenarios) {
    $name = $s -replace '\+', '-'
    & "$PSScriptRoot\shoot.ps1" -Scenario $s -Exe $Exe -Out (Join-Path $Dir "$name.png")
}
if ($Only.Count -eq 0) {
    # Minimum window size (960 x 620).
    foreach ($s in @("overview", "run-live", "dataset", "playground")) {
        & "$PSScriptRoot\shoot.ps1" -Scenario $s -Exe $Exe -Width 960 -Height 620 -Out (Join-Path $Dir "$s-min.png")
    }
    # Large window (1680 x 1050).
    foreach ($s in @("overview", "run-live")) {
        & "$PSScriptRoot\shoot.ps1" -Scenario $s -Exe $Exe -Width 1680 -Height 1050 -Out (Join-Path $Dir "$s-wide.png")
    }
}
