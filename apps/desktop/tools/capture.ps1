# Launches FOLO Desktop off screen in a scenario, captures the real window's client area to a PNG and kills
# it. It never moves the cursor, never activates the window and injects no input: FOLO_UI_OFFSCREEN=1 makes
# the app open its window outside every display without focus, and PrintWindow renders it from there.
# Design shots are shoot.ps1's (headless, no window); this one shows what the Win32 backend presents, at
# the scale of the display nearest the window (the primary one).
# Usage: tools\capture.ps1 -Scenario run-live -Out shots\run-live.png [-Exe path] [-Width 1280 -Height 800]
#        [-DelayMs 2200] [-ExtraArgs "--mock --open settings"]   (-Scenario none: no --scenario)
# -Exe defaults to $CARGO_TARGET_DIR\debug\folo-desktop.exe.
# Prints "<out> <w>x<h> scale <s> rect <window rect> fg-same <bool>". Throws if the window ever overlaps a
# display (and kills the app first), if it never appears, or if the capture stays black.
param(
    [Parameter(Mandatory = $true)][string]$Scenario,
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Width = 1280,
    [int]$Height = 800,
    [int]$DelayMs = 2200,
    [string]$ExtraArgs = "",
    [string]$Exe = ""
)
$ErrorActionPreference = "Stop"
if (-not $Exe) {
    $target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $PSScriptRoot "../target" }
    $Exe = Join-Path $target "debug/folo-desktop.exe"
}
. "$PSScriptRoot\win.ps1"
$Exe = (Resolve-Path $Exe).Path
$dir = Split-Path -Parent $Out
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }
$Out = [System.IO.Path]::GetFullPath($Out)

$argList = @("--width", $Width, "--height", $Height)
if ($Scenario -ne "none") { $argList = @("--scenario", $Scenario) + $argList }
if ($ExtraArgs) { $argList += $ExtraArgs.Split(" ") }
$errFile = "$Out.stderr.txt"
# Scenario launches neither read nor write the window state; point it away from the real profile anyway.
$stateFile = Join-Path ([System.IO.Path]::GetTempPath()) "folo-capture-desktop.json"
$fgBefore = [DfwWin]::GetForegroundWindow()
$p = Start-Dfw $Exe $argList @{ FOLO_UI_OFFSCREEN = "1"; FOLO_DESKTOP_STATE = $stateFile } $errFile
try {
    $h = Wait-DfwWindow $p 25
    if ($h -eq [IntPtr]::Zero) { throw "window did not appear for $Scenario" }
    $wr = New-Object DfwWin+RECT
    [DfwWin]::GetWindowRect($h, [ref]$wr) | Out-Null
    if (-not [DfwWin]::OffEveryMonitor($wr)) { throw "window of $Scenario is on a display: $([DfwWin]::Str($wr))" }
    Start-Sleep -Milliseconds $DelayMs
    [DfwWin]::GetWindowRect($h, [ref]$wr) | Out-Null
    if (-not [DfwWin]::OffEveryMonitor($wr)) { throw "window of $Scenario moved onto a display: $([DfwWin]::Str($wr))" }
    $r = New-Object DfwWin+RECT
    [DfwWin]::GetClientRect($h, [ref]$r) | Out-Null
    $w = $r.Right - $r.Left; $hh = $r.Bottom - $r.Top
    # A capture can come before the first frame (an all-black image): sample a grid of pixels and capture
    # again, up to three times, before giving up.
    for ($try = 0; $try -lt 4; $try++) {
        # PW_CLIENTONLY | PW_RENDERFULLCONTENT, into a plain DIB (win.ps1: GDI+'s GetHdc blackens RGB(13,11,12)).
        $bmp = [DfwPrint]::Print($h, $w, $hh, 3)
        $lit = $false
        for ($y = 8; $y -lt $hh -and -not $lit; $y += [Math]::Max(1, [int]($hh / 12))) {
            for ($x = 8; $x -lt $w; $x += [Math]::Max(1, [int]($w / 12))) {
                $c = $bmp.GetPixel($x, $y)
                if ($c.R + $c.G + $c.B -gt 0) { $lit = $true; break }
            }
        }
        if ($lit) { break }
        $bmp.Dispose()
        if ($try -eq 3) { throw "capture of $Scenario stayed black" }
        Start-Sleep -Milliseconds 800
    }
    $bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    $scale = [DfwWin]::GetDpiForWindow($h) / 96.0
    $fgSame = ([DfwWin]::GetForegroundWindow() -eq $fgBefore)
    "$Out ${w}x${hh} scale $scale rect $([DfwWin]::Str($wr)) fg-same $fgSame"
}
finally {
    Stop-Dfw $p
    # Keep stderr only when it says something (a panic, an error log). With parallel captures another
    # launch's child can inherit this redirect handle for a while; capture-all sweeps what is left.
    Remove-Item -ErrorAction SilentlyContinue "$errFile.out", $stateFile
    try { if ((Test-Path $errFile) -and ((Get-Item $errFile).Length -eq 0)) { Remove-Item $errFile -ErrorAction Stop } } catch {}
}
