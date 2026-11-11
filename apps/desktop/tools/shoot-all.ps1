# Renders every review scenario headless with `folo-desktop --shot` (tools\shoot.ps1's way: no window opens,
# nothing on screen, no cursor, no focus), in parallel: all of `--list-scenarios` at 1280 x 800, plus the
# minimum (960 x 620) and wide (1680 x 1050) window sizes for the densest screens. The 98 PNGs take a few
# seconds on a release build. This is the same matrix as the GPUI reference captures and capture-all.ps1,
# so tools\parity.py can compare the trees.
# Layout: <Dir>\<width>x<height>\<scenario>.png (the scenario name exactly, '+' kept).
# Usage: tools\shoot-all.ps1 -Dir shots\round1 [-Exe path] [-Scale 1.5] [-Parallel 8] [-Only overview,run-live]
# -Exe defaults to $CARGO_TARGET_DIR\release\folo-desktop.exe, else the debug build. -Scale is 1.5 by default,
# the dev PC's display (the 1280 x 800 PNG is 1920 x 1200). -Only renders those scenarios (at the extra sizes
# too where they are in those sets). Exits 1 if any shot failed (each failure is printed with its stderr).
param(
    [Parameter(Mandatory = $true)][string]$Dir,
    [string]$Exe = "",
    [double]$Scale = 1.5,
    [int]$Parallel = 8,
    [string[]]$Only = @()
)
$ErrorActionPreference = "Stop"
if (-not $Exe) {
    $target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $PSScriptRoot "..\target" }
    $Exe = Join-Path $target "release\folo-desktop.exe"
    if (-not (Test-Path $Exe)) { $Exe = Join-Path $target "debug\folo-desktop.exe" }
}
$Exe = (Resolve-Path $Exe).Path
New-Item -ItemType Directory -Force $Dir | Out-Null
$Dir = (Resolve-Path $Dir).Path
$scaleText = $Scale.ToString([System.Globalization.CultureInfo]::InvariantCulture)

$scenarios = @(& $Exe --list-scenarios | Where-Object { $_ })
if ($Only.Count -gt 0) { $scenarios = @($scenarios | Where-Object { $Only -contains $_ }) }
$minSet = @("overview", "overview+empty", "run-live", "run-live+charts-focus", "dataset", "playground", "models", "new-run", "settings+cloud", "settings+cloud+trouble", "cloud-connect", "cloud-connect+remote", "new-run+cloud", "run-cloud+syncing", "run-cloud+training")
$wideSet = @("overview", "overview+empty", "run-live", "model", "settings+cloud", "run-cloud+training")
$jobs = New-Object System.Collections.Generic.List[object]
foreach ($s in $scenarios) { $jobs.Add(@($s, 1280, 800)) }
foreach ($s in $minSet) { if ($scenarios -contains $s) { $jobs.Add(@($s, 960, 620)) } }
foreach ($s in $wideSet) { if ($scenarios -contains $s) { $jobs.Add(@($s, 1680, 1050)) } }

$work = {
    param($Exe, $s, $w, $h, $scale, $out)
    New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null
    $err = & $Exe --scenario $s --width $w --height $h --scale $scale --shot $out 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $out)) { return "FAIL $out (exit $LASTEXITCODE) $err" }
    "ok   $out"
}
$pool = [RunspaceFactory]::CreateRunspacePool(1, [Math]::Max(1, $Parallel))
$running = @()
$failed = 0
try {
    $pool.Open()
    $started = Get-Date
    foreach ($j in $jobs) {
        $out = Join-Path $Dir "$($j[1])x$($j[2])\$($j[0]).png"
        $ps = [PowerShell]::Create()
        $ps.RunspacePool = $pool
        $ps.AddScript($work).AddArgument($Exe).AddArgument($j[0]).AddArgument($j[1]).AddArgument($j[2]).AddArgument($scaleText).AddArgument($out) | Out-Null
        $running += [pscustomobject]@{ Ps = $ps; Handle = $ps.BeginInvoke() }
    }
    foreach ($r in $running) {
        $res = "$($r.Ps.EndInvoke($r.Handle))"
        $r.Ps.Dispose()
        if ($res.StartsWith("FAIL")) { $failed++; Write-Output $res }
    }
}
finally {
    foreach ($r in $running) { if (-not $r.Handle.IsCompleted) { $r.Ps.Stop() } }
    $pool.Close()
}
$secs = [Math]::Round(((Get-Date) - $started).TotalSeconds, 1)
Write-Output "$($jobs.Count - $failed)/$($jobs.Count) shots in $secs s -> $Dir"
if ($failed -gt 0) { exit 1 }
