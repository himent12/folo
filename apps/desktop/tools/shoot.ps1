# Launch FOLO Desktop in a scenario, capture the window's client area to a PNG, then close it.
# Usage: tools\shoot.ps1 -Scenario run-live -Out shots\r1\run-live.png [-Width 1280 -Height 800] [-Exe path]
param(
    [Parameter(Mandatory = $true)][string]$Scenario,
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Width = 1280,
    [int]$Height = 800,
    [string]$Exe = "",
    [int]$DelayMs = 2200,
    # Extra arguments for the app, e.g. "--server http://127.0.0.1:7878". With -Scenario none, no
    # scenario is passed and the app connects to real data.
    [string]$ExtraArgs = ""
)

$ErrorActionPreference = "Stop"
if (-not $Exe) {
    $target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $PSScriptRoot "../target" }
    $Exe = Join-Path $target "debug/folo-desktop.exe"
}
$Exe = (Resolve-Path $Exe).Path
Add-Type -AssemblyName System.Drawing
if (-not ("FoloWin" -as [type])) {
    Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class FoloWin {
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint flags);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
}
"@
}
[FoloWin]::SetProcessDPIAware() | Out-Null
# Park the cursor in the screen corner so no hover state leaks into the capture.
[FoloWin]::SetCursorPos(0, 0) | Out-Null

$argList = @("--width", $Width, "--height", $Height)
if ($Scenario -ne "none") { $argList = @("--scenario", $Scenario) + $argList }
if ($ExtraArgs) { $argList += $ExtraArgs.Split(" ") }
$p = Start-Process -FilePath $Exe -ArgumentList $argList -PassThru
try {
    $deadline = (Get-Date).AddSeconds(25)
    while ($p.MainWindowHandle -eq 0 -and (Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 100
        $p.Refresh()
    }
    if ($p.MainWindowHandle -eq 0) { throw "window did not appear for $Scenario" }
    Start-Sleep -Milliseconds $DelayMs
    $h = $p.MainWindowHandle
    $r = New-Object FoloWin+RECT
    for ($i = 0; $i -lt 50; $i++) {
        $p.Refresh(); $h = $p.MainWindowHandle
        [FoloWin]::GetClientRect($h, [ref]$r) | Out-Null
        $w = $r.Right - $r.Left; $hh = $r.Bottom - $r.Top
        if ($w -gt 100 -and $hh -gt 100) { break }
        Start-Sleep -Milliseconds 100
    }
    $bmp = New-Object System.Drawing.Bitmap $w, $hh
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $hdc = $g.GetHdc()
    [FoloWin]::PrintWindow($h, $hdc, 3) | Out-Null   # PW_CLIENTONLY | PW_RENDERFULLCONTENT
    $g.ReleaseHdc($hdc)
    $g.Dispose()
    $dir = Split-Path -Parent $Out
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }
    $bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    "$Out ${w}x${hh}"
}
finally {
    if (-not $p.HasExited) { $p.Kill() }
}
