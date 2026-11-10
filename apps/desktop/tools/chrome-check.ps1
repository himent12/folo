# Checks FOLO Desktop's window chrome with window messages instead of real mouse input (so a prompt that
# swallows injected input cannot interfere): what WM_NCHITTEST answers over each caption area, and what a
# non-client press and release on Maximize, Minimize and Close does. Exit code 1 when a check fails.
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [string]$AppArgs = "--scenario overview",
    [string]$Label = ""
)
$ErrorActionPreference = "Stop"
if (-not ("M" -as [type])) {
    Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class M {
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern bool IsZoomed(IntPtr h);
    [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
    [DllImport("user32.dll")] public static extern uint GetDpiForWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint msg, IntPtr w, IntPtr l);
    [DllImport("user32.dll", SetLastError = true)] public static extern IntPtr SendMessageTimeout(IntPtr h, uint msg, IntPtr w, IntPtr l, uint flags, uint ms, out IntPtr result);
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
    public static IntPtr Pt(int x, int y) { return (IntPtr)((y << 16) | (x & 0xFFFF)); }
}
"@
}
[M]::SetProcessDPIAware() | Out-Null
$fails = 0
function Check($name, $ok, $detail) {
    if ($ok) { "PASS  $name  $detail" } else { $script:fails++; "FAIL  $name  $detail" }
}
$names = @{ 1 = "HTCLIENT"; 2 = "HTCAPTION"; 8 = "HTMINBUTTON"; 9 = "HTMAXBUTTON"; 12 = "HTTOP"; 20 = "HTCLOSE" }
$p = Start-Process -FilePath $Exe -ArgumentList $AppArgs.Split(" ") -PassThru
try {
    $deadline = (Get-Date).AddSeconds(25)
    while ($p.MainWindowHandle -eq 0 -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 100; $p.Refresh() }
    if ($p.MainWindowHandle -eq 0) { throw "no window" }
    Start-Sleep -Milliseconds 2500
    $h = $p.MainWindowHandle
    $s = [M]::GetDpiForWindow($h) / 96.0
    "-- $Label ($AppArgs), scale $s"
    function Rect() { $r = New-Object M+RECT; [M]::GetWindowRect($h, [ref]$r) | Out-Null; $r }
    # What the window answers to WM_NCHITTEST at a point, after telling it the mouse is there.
    function Hit($x, $y) {
        [M]::PostMessage($h, 0x00A0, [IntPtr]1, [M]::Pt($x, $y)) | Out-Null   # WM_NCMOUSEMOVE
        Start-Sleep -Milliseconds 250
        $res = [IntPtr]::Zero
        [M]::SendMessageTimeout($h, 0x0084, [IntPtr]::Zero, [M]::Pt($x, $y), 2, 3000, [ref]$res) | Out-Null
        [int]$res
    }
    function Press($x, $y, $code) {
        Hit $x $y | Out-Null
        [M]::PostMessage($h, 0x00A1, [IntPtr]$code, [M]::Pt($x, $y)) | Out-Null   # WM_NCLBUTTONDOWN
        Start-Sleep -Milliseconds 80
        [M]::PostMessage($h, 0x00A2, [IntPtr]$code, [M]::Pt($x, $y)) | Out-Null   # WM_NCLBUTTONUP
        Start-Sleep -Milliseconds 900
    }
    $r = Rect
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
        $got = Hit $c[1] $c[2]
        Check "hit test over $($c[0])" ($got -eq $c[3]) "$($names[$got]) ($got), want $($names[$c[3]])"
    }
    $got = Hit $dragX $pageY
    Check "hit test over the page" ($got -eq 1) "$($names[$got]) ($got), want HTCLIENT"
    Press $maxX $y 9
    Check "Maximize press maximizes" ([M]::IsZoomed($h)) ""
    $r2 = Rect
    Press ($r2.Right - [int](8 * $s) - $btn - [int]($btn / 2)) ($r2.Top + [int](24 * $s)) 9
    Check "the same button restores" (-not [M]::IsZoomed($h)) ""
    Press $minX $y 8
    Check "Minimize press minimizes" ([M]::IsIconic($h)) ""
    [M]::ShowWindow($h, 9) | Out-Null
    Start-Sleep -Milliseconds 900
    Check "back from minimized" (-not [M]::IsIconic($h)) ""
    Press $closeX $y 20
    Start-Sleep -Milliseconds 800
    $p.Refresh()
    Check "Close press closes the app" ($p.HasExited) ""
}
finally {
    $p.Refresh()
    if (-not $p.HasExited) { $p.Kill() }
}
if ($fails) { "$fails check(s) failed"; exit 1 } else { "all checks passed"; exit 0 }
