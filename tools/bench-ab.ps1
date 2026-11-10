# Alternate benchmark binaries run by run on Windows, at a process priority and CPU affinity, and report how many
# cores the rest of the machine used during each run (docs/hill/cpu.md, docs/BENCHMARKS.md "Per-core efficiency").
#
#   powershell -File tools/bench-ab.ps1 -Runs "base=D:\a\bench.exe","new=D:\b\bench.exe" -Rounds 3 `
#       -Threads 8 -Affinity 5555 -Env "FOLO_BENCH_ITERS=60"
#
# -Runs      label=path pairs, run in this order every round
# -Threads   RAYON_NUM_THREADS for the run (0: leave unset)
# -Affinity  hex mask of logical CPUs ("" = all). On an i7-14700KF: 5555 = one thread on each of the 8 P-cores,
#            55 = 4 P-cores, 4 = one P-core, FFFF = both threads of every P-core
# -Priority  process priority class (High by default: it takes cores back from background builds)
# -Env       extra NAME=value pairs for the benchmark, separated by `;`
# Each run prints its own output, then `busy`: the CPU time of everything else over the run's wall time, in cores,
# and the run's peak working set (sampled every 100 ms).
param(
    [string]$Runs,
    [int]$Rounds = 3,
    [int]$Threads = 0,
    [string]$Affinity = "",
    [string]$Priority = "High",
    [string]$Env = "",
    [string]$ExeArgs = "--bench"
)

function Idle {
    $c = Get-CimInstance Win32_PerfRawData_PerfOS_Processor -Filter "Name='_Total'"
    return @([double]$c.PercentProcessorTime, [double]$c.Timestamp_Sys100NS)
}

if ($Threads -gt 0) { $env:RAYON_NUM_THREADS = "$Threads" }
foreach ($e in ($Env.Split(';') | Where-Object { $_ })) {
    $k, $v = $e.Split('=', 2)
    Set-Item "env:$k" $v
}
$cpus = [Environment]::ProcessorCount
for ($r = 1; $r -le $Rounds; $r++) {
    foreach ($run in ($Runs.Split(';') | Where-Object { $_ })) {
        $label, $exe = $run.Split('=', 2)
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $exe
        $psi.Arguments = $ExeArgs
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $i0 = Idle
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $p = [System.Diagnostics.Process]::Start($psi)
        try {
            $p.PriorityClass = $Priority
            if ($Affinity) { $p.ProcessorAffinity = [IntPtr][Convert]::ToInt64($Affinity, 16) }
        } catch {}
        # Read the output asynchronously and sample the peak working set until the process exits.
        $read = $p.StandardOutput.ReadToEndAsync()
        $peak = 0
        while (-not $p.WaitForExit(100)) {
            try { $p.Refresh(); $peak = [Math]::Max($peak, $p.PeakWorkingSet64) } catch {}
        }
        $p.WaitForExit()
        $out = $read.Result
        $wall = $sw.Elapsed.TotalSeconds
        $i1 = Idle
        $own = $p.TotalProcessorTime.TotalSeconds / $wall
        $total = $cpus * (1.0 - ($i1[0] - $i0[0]) / ($i1[1] - $i0[1]))
        $busy = [Math]::Max(0.0, $total - $own)
        Write-Output ("=== round {0} {1} (threads {2}, affinity {3}, {4})" -f $r, $label, $Threads, $Affinity, $Priority)
        Write-Output $out.TrimEnd()
        Write-Output ("busy: others {0:F1} cores, this run {1:F1} cores, wall {2:F1} s, peak working set {3:F0} MB" -f `
            $busy, $own, $wall, ($peak / 1MB))
    }
}
