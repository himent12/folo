# Checks FOLO Desktop's window chrome with window messages instead of real mouse input (so a prompt that
# swallows injected input cannot interfere): what WM_NCHITTEST answers over each caption area, and what a
# non-client press and release on Maximize, Minimize and Close does. Exit code 1 when a check fails.
#
#   tools\chrome-check.ps1 -Exe <exe> [-AppArgs "--scenario overview"]   # on your screen (it maximizes)
#   tools\chrome-check.ps1 -Exe <exe> -Offscreen   # FOLO_UI_OFFSCREEN=1: never on a display, never active
#   tools\chrome-check.ps1 -Exe <exe> -Sandbox     # on a private desktop nobody sees (sandbox-run.ps1)
#
# Off screen the app emulates maximize (the client takes the work area's size, at the off-screen origin)
# and Windows' maximize is never called, so "maximized" is checked by size there. Restoring a minimized
# window is WM_SYSCOMMAND SC_RESTORE, what the taskbar sends.
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [string]$AppArgs = "--scenario overview",
    [string]$Label = "",
    [switch]$Offscreen,
    [switch]$Sandbox,
    [switch]$InSandbox
)
$ErrorActionPreference = "Stop"
$Exe = (Resolve-Path $Exe).Path
if ($Sandbox -and -not $InSandbox) {
    $self = $MyInvocation.MyCommand.Path
    $off = if ($Offscreen) { " -Offscreen" } else { "" }
    & (Join-Path $PSScriptRoot "sandbox-run.ps1") -TimeoutS 300 -Command "& '$self' -Exe '$Exe' -AppArgs '$AppArgs' -Label '$Label' -InSandbox$off"
    exit $LASTEXITCODE
}
. (Join-Path $PSScriptRoot "drive.ps1")
$fails = 0
function Check($name, $ok, $detail) {
    if ($ok) { "PASS  $name  $detail" } else { $script:fails++; "FAIL  $name  $detail" }
}
$names = @{ 1 = "HTCLIENT"; 2 = "HTCAPTION"; 8 = "HTMINBUTTON"; 9 = "HTMAXBUTTON"; 12 = "HTTOP"; 20 = "HTCLOSE" }
$tmp = if ($env:FOLO_TEST_TMP) { $env:FOLO_TEST_TMP } else { [System.IO.Path]::GetTempPath() }
$errFile = Join-Path $tmp "chrome-check-$PID.stderr.txt"
$envs = @{ FOLO_DESKTOP_STATE = (Join-Path $tmp "chrome-check-$PID.json") }
if ($Offscreen) { $envs.FOLO_UI_OFFSCREEN = "1" }
$fg0 = [FoloDrive]::GetForegroundWindow()
$p = Start-Folo $Exe $AppArgs.Split(" ") $envs $errFile
try {
    $h = Wait-FoloWindow $p 25
    if ($h -eq [IntPtr]::Zero) { throw "no window" }
    Start-Sleep -Milliseconds 2500
    $s = [FoloDrive]::GetDpiForWindow($h) / 96.0
    $mode = if ($Offscreen) { "off screen" } else { "on screen" }
    if ($InSandbox) { $mode += ", private desktop" }
    "-- $Label ($AppArgs), $mode, scale $s"
    function Rect() { [FoloDrive]::Rect($h) }
    function Maximized() {
        if (-not $Offscreen) { return [FoloDrive]::IsZoomed($h) }
        # Emulated: the client area is the work area's size.
        $c = New-Object FoloDrive+RECT; [FoloDrive]::GetClientRect($h, [ref]$c) | Out-Null
        $m = [FoloDrive]::Monitors()[0]
        ($c.Right -ge ($m.Right - $m.Left) * 0.9) -and ($c.Bottom -ge ($m.Bottom - $m.Top) * 0.8)
    }
    function Press($x, $y, $code) {
        [FoloDrive]::NcClick($h, $x, $y, $code, $false)
        if ($Offscreen) {
            # Watch the rectangle while the state changes: it must never touch a display.
            $on = @()
            for ($i = 0; $i -lt 45; $i++) {
                $wr = Rect
                if (-not [FoloDrive]::OffEveryMonitor($wr)) { $on += [FoloDrive]::Str($wr) }
                Start-Sleep -Milliseconds 20
            }
            Check "  off every display throughout (code $code)" ($on.Count -eq 0) ($on -join " ")
        } else { Start-Sleep -Milliseconds 900 }
    }
    $r = Rect
    if ($Offscreen) { Check "window off every display" ([FoloDrive]::OffEveryMonitor($r)) ([FoloDrive]::Str($r)) }
    $y = $r.Top + [int](24 * $s)
    $btn = [int](46 * $s)
    # The window rect includes about 8 logical px of invisible frame on the left, right and bottom.
    $inset = [int](8 * $s)
    $closeX = $r.Right - $inset - [int]($btn / 2)
    $maxX = $closeX - $btn
    $minX = $maxX - $btn
    $dragX = $r.Left + [int](520 * $s)
    $brandX = $r.Left + $inset + [int](60 * $s)
    $pageY = $r.Top + [int](400 * $s)
    foreach ($c in @(@("Close", $closeX, $y, 20), @("Maximize", $maxX, $y, 9), @("Minimize", $minX, $y, 8), @("title bar", $dragX, $y, 2), @("sidebar brand row", $brandX, $y, 2), @("top edge", $dragX, ($r.Top + [int](2 * $s)), 12))) {
        $got = [FoloDrive]::HitTest($h, $c[1], $c[2])
        Check "hit test over $($c[0])" ($got -eq $c[3]) "$($names[$got]) ($got), want $($names[$c[3]])"
    }
    $got = [FoloDrive]::HitTest($h, $dragX, $pageY)
    Check "hit test over the page" ($got -eq 1) "$($names[$got]) ($got), want HTCLIENT"
    Press $maxX $y 9
    Check "Maximize press maximizes" (Maximized) ([FoloDrive]::Str((Rect)))
    $r2 = Rect
    $max2X = $r2.Right - [int](8 * $s) - $btn - [int]($btn / 2)
    $got = [FoloDrive]::HitTest($h, $max2X, $r2.Top + [int](24 * $s))
    Check "maximized: hit test over Restore" ($got -eq 9) "$($names[$got]) ($got)"
    Press $max2X ($r2.Top + [int](24 * $s)) 9
    Check "the same button restores" ((-not (Maximized)) -and ((Rect).Left -eq $r.Left) -and ((Rect).Right -eq $r.Right)) ([FoloDrive]::Str((Rect)))
    Press $minX $y 8
    Check "Minimize press minimizes" ([FoloDrive]::IsIconic($h)) ""
    [FoloDrive]::Post($h, 0x112, [IntPtr]0xF120, [IntPtr]::Zero)   # WM_SYSCOMMAND SC_RESTORE (the taskbar's)
    Start-Sleep -Milliseconds 900
    Check "back from minimized" ((-not [FoloDrive]::IsIconic($h)) -and ((Rect).Left -eq $r.Left)) ([FoloDrive]::Str((Rect)))
    if ($Offscreen) {
        Check "still off every display" ([FoloDrive]::OffEveryMonitor((Rect))) ([FoloDrive]::Str((Rect)))
        Check "foreground window unchanged" ([FoloDrive]::GetForegroundWindow() -eq $fg0) ""
    }
    Press $closeX $y 20
    Check "Close press closes the app" (Wait-FoloExit $p 5) "exit code $(if ($p.HasExited) { $p.ExitCode })"
}
finally {
    Stop-Folo $p
    if ((Test-Path $errFile) -and (Get-Item $errFile).Length) { "stderr: " + ((Get-Content $errFile) -join " | ") }
    Remove-Item -ErrorAction SilentlyContinue $errFile, "$errFile.out", $envs.FOLO_DESKTOP_STATE
}
if ($fails) { "$fails check(s) failed"; exit 1 } else { "all checks passed"; exit 0 }
