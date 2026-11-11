# Performance numbers of a FOLO Desktop build, always off screen (FOLO_UI_OFFSCREEN=1): it needs only the
# app's command line (--mock, --open, --width, --height) and, for startup, the FOLO_UI_TRACE=1 stderr lines
# `first-render <ms>` and `first-frame <ms>` (ms since process creation).
# Usage: tools\perf.ps1 -Label main -Out perf\main.json [-Exe path] [-Launches 12] [-SettleS 6] [-LoadS 30]
#        [-RunId run_a19f] [-DatasetId ds_8f2c]
# -Exe defaults to $CARGO_TARGET_DIR\debug\folo-desktop.exe; measure a release build. A run takes about three
# and a half minutes. The numbers swing with the machine's load: compare two builds measured back to back.
# Measures:
#   startup  -Launches launches of --mock; median and min of first-render / first-frame / window-visible
#            (the first launch is the cold one, reported apart and left out of the stats);
#   memory   after -SettleS s on overview, datasets, dataset:<id>, run:<id> (live): working set, private
#            bytes, peak working set, threads, handles, GPU dedicated and shared memory (sum of instances);
#   load     over -LoadS s on settings (static), run:<id> (the mock's simulator keeps it updating) and the
#            frozen `--scenario run-live` (the same page, no new data: what its animations cost):
#            process CPU seconds (kernel + user), average GPU engine utilization (sum of engines);
#   context  logical cores, display refresh rate, total CPU and the busiest other processes.
param(
    [Parameter(Mandatory = $true)][string]$Label,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$Exe = "",
    [int]$Launches = 12,
    [int]$SettleS = 6,
    [int]$LoadS = 30,
    [string]$RunId = "run_a19f",
    [string]$DatasetId = "ds_8f2c",
    [int]$Width = 1280,
    [int]$Height = 800
)
$ErrorActionPreference = "Stop"
if (-not $Exe) {
    $target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $PSScriptRoot "../target" }
    $Exe = Join-Path $target "debug/folo-desktop.exe"
}
. "$PSScriptRoot\win.ps1"
$Exe = (Resolve-Path $Exe).Path
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) "folo-perf-$PID"
New-Item -ItemType Directory -Force $tmp | Out-Null
$size = @("--width", $Width, "--height", $Height)
# --width/--height already keep the launch from reading or writing the remembered window; a scratch state
# file keeps the real profile out of it regardless.
$env0 = @{ FOLO_UI_OFFSCREEN = "1"; FOLO_DESKTOP_STATE = (Join-Path $tmp "desktop.json") }

function Median([double[]]$xs) {
    $s = @($xs | Sort-Object); $n = $s.Count
    if ($n -eq 0) { return $null }
    if ($n % 2) { $s[($n - 1) / 2] } else { ($s[$n / 2 - 1] + $s[$n / 2]) / 2 }
}
function Stats([double[]]$xs) {
    if (-not $xs -or $xs.Count -eq 0) { return $null }
    [ordered]@{ median = [math]::Round((Median $xs), 1); min = [math]::Round(($xs | Measure-Object -Minimum).Minimum, 1);
        max = [math]::Round(($xs | Measure-Object -Maximum).Maximum, 1); n = $xs.Count }
}
function Launch([string[]]$appArgs, [hashtable]$extraEnv, [string]$name) {
    $e = @{} + $env0
    foreach ($k in $extraEnv.Keys) { $e[$k] = $extraEnv[$k] }
    Start-Dfw $Exe ($appArgs + $size) $e (Join-Path $tmp "$name.stderr.txt")
}
function GpuMem([int]$procId) {
    $r = [ordered]@{ dedicated_mb = $null; shared_mb = $null }
    try {
        $c = Get-Counter -ErrorAction SilentlyContinue "\GPU Process Memory(pid_${procId}_*)\Dedicated Usage", "\GPU Process Memory(pid_${procId}_*)\Shared Usage"
        $ded = ($c.CounterSamples | Where-Object { $_.Path -like "*dedicated usage" } | Measure-Object CookedValue -Sum).Sum
        $sha = ($c.CounterSamples | Where-Object { $_.Path -like "*shared usage" } | Measure-Object CookedValue -Sum).Sum
        $r.dedicated_mb = [math]::Round($ded / 1MB, 1); $r.shared_mb = [math]::Round($sha / 1MB, 1)
    } catch { $r.error = "$_" }
    $r
}
function TopOthers([int]$n) {
    # CPU over one second per process, from the Process counters (excluding Idle and _Total).
    $c = Get-Counter -ErrorAction SilentlyContinue "\Process(*)\% Processor Time" -SampleInterval 1 -MaxSamples 1
    @($c.CounterSamples | Where-Object { $_.InstanceName -notin @("_total", "idle") -and $_.CookedValue -gt 1 } |
        Sort-Object CookedValue -Descending | Select-Object -First $n |
        ForEach-Object { "$($_.InstanceName) $([math]::Round($_.CookedValue / [Environment]::ProcessorCount, 1))%" })
}
function Context() {
    $cpu = (Get-Counter -ErrorAction SilentlyContinue "\Processor(_Total)\% Processor Time" -SampleInterval 1 -MaxSamples 3).CounterSamples | Measure-Object CookedValue -Average
    $vc = Get-CimInstance Win32_VideoController | Select-Object -First 1
    [ordered]@{
        time = (Get-Date).ToString("s")
        logical_cores = [Environment]::ProcessorCount
        cpu_name = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name.Trim()
        gpu_name = $vc.Name
        refresh_hz = $vc.CurrentRefreshRate
        display = "$($vc.CurrentHorizontalResolution)x$($vc.CurrentVerticalResolution)"
        total_cpu_pct = [math]::Round($cpu.Average, 1)
        rustc_processes = @(Get-Process rustc -ErrorAction SilentlyContinue).Count
        busiest_processes_pct_of_machine = TopOthers 8
    }
}

$result = [ordered]@{ label = $Label; exe = $Exe; exe_bytes = (Get-Item $Exe).Length; width = $Width; height = $Height }
$result.context_before = Context
"context: $($result.context_before.total_cpu_pct)% CPU, $($result.context_before.rustc_processes) rustc"

# ---------- startup ----------
$launchRows = @()
for ($i = 0; $i -lt $Launches; $i++) {
    $p = Launch @("--mock") @{ FOLO_UI_TRACE = "1" } "startup$i"
    $errFile = Join-Path $tmp "startup$i.stderr.txt"
    $row = [ordered]@{ launch = $i; first_render_ms = $null; first_frame_ms = $null; window_ms = $null }
    try {
        $created = $p.StartTime
        $deadline = (Get-Date).AddSeconds(30)
        while ((Get-Date) -lt $deadline) {
            if (-not $row.window_ms -and [DfwWin]::FindWindow([uint32]$p.Id) -ne [IntPtr]::Zero) {
                $row.window_ms = [math]::Round(((Get-Date) - $created).TotalMilliseconds, 1)
            }
            $txt = ""
            # The app holds the file open for writing: open it sharing read and write.
            try {
                $fs = New-Object System.IO.FileStream($errFile, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
                try { $txt = (New-Object System.IO.StreamReader($fs)).ReadToEnd() } finally { $fs.Dispose() }
            } catch {}
            if ($txt -match "first-frame ([\d.]+)") {
                $row.first_frame_ms = [double]$Matches[1]
                if ($txt -match "first-render ([\d.]+)") { $row.first_render_ms = [double]$Matches[1] }
                break
            }
            Start-Sleep -Milliseconds 10
        }
        Start-Sleep -Milliseconds 500   # let it finish the startup burst before the kill
    }
    finally { Stop-Dfw $p }
    $launchRows += $row
    "startup $i render $($row.first_render_ms) frame $($row.first_frame_ms) window $($row.window_ms)"
    Start-Sleep -Milliseconds 300
}
$warm = @($launchRows | Select-Object -Skip 1)
$result.startup = [ordered]@{
    cold = $launchRows[0]
    first_render_ms = Stats ([double[]]@($warm | Where-Object { $null -ne $_.first_render_ms } | ForEach-Object { $_.first_render_ms }))
    first_frame_ms = Stats ([double[]]@($warm | Where-Object { $null -ne $_.first_frame_ms } | ForEach-Object { $_.first_frame_ms }))
    window_visible_ms = Stats ([double[]]@($warm | Where-Object { $null -ne $_.window_ms } | ForEach-Object { $_.window_ms }))
    window_visible_note = "polled every ~10 ms from outside: first visible top-level window of the process, ms after Process.StartTime"
    launches = $launchRows
}

# ---------- memory ----------
$states = [ordered]@{
    overview = @("--mock")
    datasets = @("--mock", "--open", "datasets")
    dataset = @("--mock", "--open", "dataset:$DatasetId")
    run_live = @("--mock", "--open", "run:$RunId")
}
$result.memory = [ordered]@{}
foreach ($k in $states.Keys) {
    $p = Launch $states[$k] @{} "mem-$k"
    try {
        if ((Wait-DfwWindow $p 30) -eq [IntPtr]::Zero) { throw "no window for $k" }
        Start-Sleep -Seconds $SettleS
        $p.Refresh()
        $m = [ordered]@{
            args = ($states[$k] -join " ")
            working_set_mb = [math]::Round($p.WorkingSet64 / 1MB, 1)
            private_mb = [math]::Round($p.PrivateMemorySize64 / 1MB, 1)
            peak_working_set_mb = [math]::Round($p.PeakWorkingSet64 / 1MB, 1)
            threads = $p.Threads.Count
            handles = $p.HandleCount
        }
        $g = GpuMem $p.Id
        foreach ($gk in $g.Keys) { $m["gpu_$gk"] = $g[$gk] }
        $result.memory[$k] = $m
        "memory $k ws $($m.working_set_mb) MB private $($m.private_mb) MB gpu $($m.gpu_dedicated_mb)/$($m.gpu_shared_mb) MB threads $($m.threads) handles $($m.handles)"
    }
    finally { Stop-Dfw $p }
}

# ---------- CPU and GPU load ----------
$loadStates = [ordered]@{ static_settings = @("--mock", "--open", "settings"); run_live = @("--mock", "--open", "run:$RunId"); run_live_frozen = @("--scenario", "run-live") }
$result.load = [ordered]@{}
foreach ($k in $loadStates.Keys) {
    $p = Launch $loadStates[$k] @{} "load-$k"
    try {
        if ((Wait-DfwWindow $p 30) -eq [IntPtr]::Zero) { throw "no window for $k" }
        Start-Sleep -Seconds $SettleS
        $p.Refresh()
        $cpu0 = $p.TotalProcessorTime; $k0 = $p.PrivilegedProcessorTime; $t0 = Get-Date
        $samples = @()
        $err = $null
        try {
            $c = Get-Counter -ErrorAction SilentlyContinue "\GPU Engine(pid_$($p.Id)_*)\Utilization Percentage" -SampleInterval 1 -MaxSamples $LoadS
            foreach ($set in $c) { $samples += ($set.CounterSamples | Measure-Object CookedValue -Sum).Sum }
        } catch { $err = "$_" }
        # A process that uses no GPU has no engine counters, and Get-Counter returns at once: wait out the rest.
        $left = $LoadS - ((Get-Date) - $t0).TotalSeconds
        if ($left -gt 0) { Start-Sleep -Milliseconds ([int](1000 * $left)) }
        $p.Refresh()
        $wall = ((Get-Date) - $t0).TotalSeconds
        $cpu = ($p.TotalProcessorTime - $cpu0).TotalSeconds
        $l = [ordered]@{
            args = ($loadStates[$k] -join " ")
            seconds = [math]::Round($wall, 2)
            cpu_seconds = [math]::Round($cpu, 3)
            cpu_kernel_seconds = [math]::Round(($p.PrivilegedProcessorTime - $k0).TotalSeconds, 3)
            cpu_pct_of_one_core = [math]::Round(100 * $cpu / $wall, 2)
            gpu_util_pct_avg = if ($samples.Count) { [math]::Round(($samples | Measure-Object -Average).Average, 2) } else { $null }
            gpu_util_pct_max = if ($samples.Count) { [math]::Round(($samples | Measure-Object -Maximum).Maximum, 2) } else { $null }
            gpu_samples = $samples.Count
        }
        if ($err) { $l.gpu_error = $err }
        $result.load[$k] = $l
        "load $k cpu $($l.cpu_seconds) s over $($l.seconds) s ($($l.cpu_pct_of_one_core)% of a core), gpu avg $($l.gpu_util_pct_avg)% max $($l.gpu_util_pct_max)%"
    }
    finally { Stop-Dfw $p }
}
$result.context_after = Context

$dir = Split-Path -Parent $Out
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }
$result | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 $Out
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $tmp
"wrote $Out"
