# Captures every scenario of `<exe> --list-scenarios` at 1280x800, plus the minimum (960x620) and wide
# (1680x1050) window sizes for the scenarios shoot-all.ps1 renders there, all with real windows off screen
# (capture.ps1). Layout: <Dir>/<width>x<height>/<scenario>.png (the scenario name exactly, '+' kept).
# Usage: tools\capture-all.ps1 -Dir <dir> [-Exe path] [-Parallel 4] [-Only overview,run-live] [-DelayMs 2200]
# Writes <Dir>/capture-log.txt (one line per capture, failures marked FAIL) and exits 1 if any failed.
param(
    [Parameter(Mandatory = $true)][string]$Dir,
    [string]$Exe = "",
    [int]$Parallel = 4,
    [string[]]$Only = @(),
    [int]$DelayMs = 2200
)
$ErrorActionPreference = "Stop"
if (-not $Exe) {
    $target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $PSScriptRoot "../target" }
    $Exe = Join-Path $target "debug/folo-desktop.exe"
}
$Exe = (Resolve-Path $Exe).Path
New-Item -ItemType Directory -Force $Dir | Out-Null
$Dir = (Resolve-Path $Dir).Path
$capture = Join-Path $PSScriptRoot "capture.ps1"

$scenarios = @(& $Exe --list-scenarios | Where-Object { $_ })
if ($Only.Count -gt 0) { $scenarios = @($scenarios | Where-Object { $Only -contains $_ }) }
# The extra sizes, as in shoot-all.ps1.
$minSet = @("overview", "overview+empty", "run-live", "run-live+charts-focus", "dataset", "playground", "models", "new-run", "settings+cloud", "settings+cloud+trouble", "cloud-connect", "cloud-connect+remote", "new-run+cloud", "run-cloud+syncing", "run-cloud+training")
$wideSet = @("overview", "overview+empty", "run-live", "model", "settings+cloud", "run-cloud+training")
$jobs = New-Object System.Collections.Generic.List[object]
foreach ($s in $scenarios) { $jobs.Add(@($s, 1280, 800)) }
foreach ($s in $minSet) { if ($scenarios -contains $s) { $jobs.Add(@($s, 960, 620)) } }
foreach ($s in $wideSet) { if ($scenarios -contains $s) { $jobs.Add(@($s, 1680, 1050)) } }

# Warm-up launch (font and file caches), so the first real capture isn't caught mid-load.
& $capture -Exe $Exe -Scenario $scenarios[0] -Out (Join-Path $Dir "warmup.png") -DelayMs $DelayMs | Out-Null
Remove-Item -ErrorAction SilentlyContinue (Join-Path $Dir "warmup.png")

$work = {
    param($capture, $Exe, $s, $w, $h, $out, $delay)
    $ErrorActionPreference = "Stop"
    # A busy machine can make a window slow to appear: retry a capture once before giving up.
    for ($i = 0; $i -lt 2; $i++) {
        try { return (& $capture -Exe $Exe -Scenario $s -Out $out -Width $w -Height $h -DelayMs $delay) }
        catch { $err = "$_" }
    }
    "FAIL $out $err"
}
# The runspaces share this process's environment, and Start-Dfw sets and restores variables around each
# launch: set the off-screen switch for the whole process first, so a restore never clears it under
# another runspace's launch. It is put back once every worker has stopped, so a Desktop launched later
# from the same shell opens on screen again.
$offscreen0 = [Environment]::GetEnvironmentVariable("FOLO_UI_OFFSCREEN")
[Environment]::SetEnvironmentVariable("FOLO_UI_OFFSCREEN", "1")
$pool = [RunspaceFactory]::CreateRunspacePool(1, [Math]::Max(1, $Parallel))
$log = @()
$running = @()
try {
    $pool.Open()
    $started = Get-Date
    foreach ($j in $jobs) {
        $out = Join-Path $Dir "$($j[1])x$($j[2])\$($j[0]).png"
        $ps = [PowerShell]::Create()
        $ps.RunspacePool = $pool
        $ps.AddScript($work).AddArgument($capture).AddArgument($Exe).AddArgument($j[0]).AddArgument($j[1]).AddArgument($j[2]).AddArgument($out).AddArgument($DelayMs) | Out-Null
        $running += [pscustomobject]@{ Ps = $ps; Handle = $ps.BeginInvoke() }
    }
    foreach ($r in $running) {
        $res = $r.Ps.EndInvoke($r.Handle)
        foreach ($e in $r.Ps.Streams.Error) { $res += "FAIL $e" }
        $r.Ps.Dispose()
        $log += $res
    }
}
finally {
    # Stops any worker still running (an interrupted run), then restores the variable.
    foreach ($r in $running) { if (-not $r.Handle.IsCompleted) { $r.Ps.Stop() } }
    $pool.Close()
    [Environment]::SetEnvironmentVariable("FOLO_UI_OFFSCREEN", $offscreen0)
}
# Redirect files a parallel capture could not delete (another child still held them): empty ones go.
Get-ChildItem -Recurse $Dir -Include *.stderr.txt, *.stderr.txt.out | Where-Object { $_.Length -eq 0 } | Remove-Item -ErrorAction SilentlyContinue
$log += @(Get-ChildItem -Recurse $Dir -Filter *.stderr.txt | ForEach-Object { "STDERR $($_.FullName): $((Get-Content $_.FullName -TotalCount 3) -join ' | ')" })
$secs = [int]((Get-Date) - $started).TotalSeconds
$fails = @($log | Where-Object { "$_" -like "FAIL*" })
$log += "$($jobs.Count) captures, $($fails.Count) failed, $secs s, parallel $Parallel"
$log | Set-Content -Encoding utf8 (Join-Path $Dir "capture-log.txt")
$log | Select-Object -Last 1
$fails
if ($fails.Count) { exit 1 }
