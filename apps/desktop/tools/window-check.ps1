# Drives FOLO Desktop's window chrome with real mouse and keyboard input (Windows) and checks what
# the window did: the caption buttons, double-click and drag on both drag areas, the top resize edge,
# F11 full screen and its caption buttons, the breadcrumbs and search field inside the title bar,
# and restoring the size, position and maximized / full-screen state on the next launch.
#
#   tools\window-check.ps1                       # $CARGO_TARGET_DIR\debug\folo-desktop.exe
#   tools\window-check.ps1 -Exe D:\b\release\folo-desktop.exe
#
# The app remembers its window in $FOLO_DESKTOP_STATE, pointed here at a fresh file under -StateDir
# ($FOLO_TEST_TMP\window-check by default), never the real profile. It moves the real pointer: leave
# the mouse alone while it runs (about a minute). Exits 1 if any check fails.
param(
    [string]$Exe = "",
    [string]$StateDir = "",
    [string]$ExtraArgs = "--mock --open run:run_a19f"
)
$ErrorActionPreference = "Stop"
if (-not $Exe) {
    $target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $PSScriptRoot "..\target" }
    $Exe = Join-Path $target "debug\folo-desktop.exe"
}
if (-not (Test-Path $Exe)) { throw "no app at $Exe (build it, or pass -Exe)" }
if (-not $StateDir) {
    $tmp = if ($env:FOLO_TEST_TMP) { $env:FOLO_TEST_TMP } else { [System.IO.Path]::GetTempPath() }
    $StateDir = Join-Path $tmp "window-check"
}
if (Test-Path $StateDir) { Remove-Item -Recurse -Force $StateDir }
New-Item -ItemType Directory -Force $StateDir | Out-Null
$stateFile = Join-Path $StateDir "desktop.json"
$env:FOLO_DESKTOP_STATE = $stateFile

# The real profile's file must not change.
$home_ = if ($env:FOLO_HOME) { $env:FOLO_HOME } else { Join-Path $env:USERPROFILE ".folo" }
$realFile = Join-Path $home_ "desktop.json"
$realBefore = if (Test-Path $realFile) { (Get-Item $realFile).LastWriteTimeUtc.Ticks } else { 0 }

Add-Type -ReferencedAssemblies System.Drawing @"
using System;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
public static class W {
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
    [DllImport("user32.dll")] public static extern bool IsZoomed(IntPtr h);
    [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
    [DllImport("user32.dll")] public static extern uint GetDpiForWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern int GetWindowLong(IntPtr h, int idx);
    [DllImport("user32.dll")] public static extern IntPtr MonitorFromWindow(IntPtr h, uint flags);
    [DllImport("user32.dll")] public static extern bool GetMonitorInfo(IntPtr m, ref MONITORINFO mi);
    [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc f, IntPtr l);
    [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint dx, uint dy, uint data, UIntPtr extra);
    [DllImport("user32.dll")] public static extern void keybd_event(byte vk, byte scan, uint flags, UIntPtr extra);
    [DllImport("user32.dll")] public static extern bool PostMessageW(IntPtr h, uint msg, IntPtr w, IntPtr l);
    [DllImport("user32.dll")] public static extern IntPtr GetClassLongPtrW(IntPtr h, int idx);
    public delegate bool EnumProc(IntPtr h, IntPtr l);
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
    [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
    [StructLayout(LayoutKind.Sequential)] public struct MONITORINFO { public int cbSize; public RECT rcMonitor, rcWork; public uint dwFlags; }
    // The app's main window: visible, titled "FOLO Desktop", owned by the process.
    public static IntPtr FindMain(uint pid) {
        IntPtr found = IntPtr.Zero;
        EnumWindows((h, l) => {
            uint p; GetWindowThreadProcessId(h, out p);
            if (p != pid || !IsWindowVisible(h)) return true;
            var sb = new StringBuilder(64); GetWindowText(h, sb, 64);
            if (sb.ToString() == "FOLO Desktop") { found = h; return false; }
            return true;
        }, IntPtr.Zero);
        return found;
    }
    public static RECT Client(IntPtr h) {
        RECT r; GetClientRect(h, out r);
        POINT p = new POINT(); ClientToScreen(h, ref p);
        r.Left += p.X; r.Right += p.X; r.Top += p.Y; r.Bottom += p.Y;
        return r;
    }
    public static RECT Monitor(IntPtr h) {
        var mi = new MONITORINFO(); mi.cbSize = Marshal.SizeOf(mi);
        GetMonitorInfo(MonitorFromWindow(h, 2), ref mi);
        return mi.rcMonitor;
    }
    // Point there with a small move first, so the app has seen the pointer before the press.
    public static void Hover(int x, int y) {
        SetCursorPos(x - 2, y); Thread.Sleep(120);
        SetCursorPos(x, y); Thread.Sleep(150);
    }
    public static void Click(int x, int y) {
        Hover(x, y);
        mouse_event(0x0002, 0, 0, 0, UIntPtr.Zero); Thread.Sleep(60);
        mouse_event(0x0004, 0, 0, 0, UIntPtr.Zero);
    }
    public static void DoubleClick(int x, int y) {
        Hover(x, y);
        for (int i = 0; i < 2; i++) {
            mouse_event(0x0002, 0, 0, 0, UIntPtr.Zero); Thread.Sleep(40);
            mouse_event(0x0004, 0, 0, 0, UIntPtr.Zero); Thread.Sleep(60);
        }
    }
    public static void Drag(int x, int y, int dx, int dy) {
        Hover(x, y);
        mouse_event(0x0002, 0, 0, 0, UIntPtr.Zero); Thread.Sleep(150);
        for (int i = 1; i <= 12; i++) { SetCursorPos(x + dx * i / 12, y + dy * i / 12); Thread.Sleep(30); }
        Thread.Sleep(100);
        mouse_event(0x0004, 0, 0, 0, UIntPtr.Zero);
    }
    public static void Key(byte vk, byte mod) {
        if (mod != 0) { keybd_event(mod, 0, 0, UIntPtr.Zero); Thread.Sleep(30); }
        keybd_event(vk, 0, 0, UIntPtr.Zero); Thread.Sleep(40);
        keybd_event(vk, 0, 2, UIntPtr.Zero);
        if (mod != 0) { Thread.Sleep(30); keybd_event(mod, 0, 2, UIntPtr.Zero); }
    }
    // A cheap fingerprint of what the screen shows in a rectangle.
    public static long Fingerprint(int x, int y, int w, int h) {
        using (var bmp = new Bitmap(w, h)) {
            using (var g = Graphics.FromImage(bmp)) g.CopyFromScreen(x, y, 0, 0, new Size(w, h));
            long sum = 0;
            for (int j = 0; j < h; j += 2) for (int i = 0; i < w; i += 2) sum = sum * 31 + bmp.GetPixel(i, j).ToArgb();
            return sum;
        }
    }
    public static void Shot(IntPtr h, string path) {
        RECT r = Monitor(h);
        using (var bmp = new Bitmap(r.Right - r.Left, r.Bottom - r.Top)) {
            using (var g = Graphics.FromImage(bmp)) g.CopyFromScreen(r.Left, r.Top, 0, 0, bmp.Size);
            bmp.Save(path);
        }
    }
}
"@
[W]::SetProcessDPIAware() | Out-Null

$script:failed = @()
$script:p = $null
$script:h = [IntPtr]::Zero
$script:s = 1.0

function Check([string]$name, [bool]$ok, [string]$detail = "") {
    $mark = if ($ok) { "PASS" } else { "FAIL" }
    "{0}  {1,-58} {2}" -f $mark, $name, $detail
    if (-not $ok) { $script:failed += $name }
}
function Rect {
    $r = New-Object W+RECT
    [W]::GetWindowRect($script:h, [ref]$r) | Out-Null
    $r
}
function Fmt($r) { "({0},{1}) {2}x{3}" -f $r.Left, $r.Top, ($r.Right - $r.Left), ($r.Bottom - $r.Top) }
function Same($a, $b, [int]$tol = 2) {
    [Math]::Abs($a.Left - $b.Left) -le $tol -and [Math]::Abs($a.Top - $b.Top) -le $tol -and
    [Math]::Abs($a.Right - $b.Right) -le $tol -and [Math]::Abs($a.Bottom - $b.Bottom) -le $tol
}
function Zoomed { [W]::IsZoomed($script:h) }
function Iconic { [W]::IsIconic($script:h) }
function Fullscreen { Same (Rect) ([W]::Monitor($script:h)) 0 }
function State { "zoomed={0} iconic={1} full={2} rect={3}" -f (Zoomed), (Iconic), (Fullscreen), (Fmt (Rect)) }
function Settle([int]$ms = 900) { Start-Sleep -Milliseconds $ms }

# Points in the title bar (physical px), from the client area: the bar is 40 px tall, the caption
# buttons 46 px wide at the right, the search field 280 px left of them, the sidebar 232 px wide.
function Bar { $c = [W]::Client($script:h); [int]($c.Top + 20 * $script:s) }
function BtnX([int]$i) { $c = [W]::Client($script:h); [int]($c.Right - (46 * $i + 23) * $script:s) }  # 0 close, 1 max, 2 min
function DragX { $c = [W]::Client($script:h); [int]($c.Right - 500 * $script:s) }
function BrandX { $c = [W]::Client($script:h); [int]($c.Left + 180 * $script:s) }

function Launch([string]$label) {
    # A Close that failed leaves the last one running: stop it (only ever our own process).
    if ($script:p) {
        $script:p.Refresh()
        if (-not $script:p.HasExited) { $script:p.Kill(); $script:p.WaitForExit(5000) | Out-Null }
    }
    $script:p = Start-Process -FilePath $Exe -ArgumentList $ExtraArgs.Split(" ") -PassThru
    $deadline = (Get-Date).AddSeconds(30)
    $script:h = [IntPtr]::Zero
    while ($script:h -eq [IntPtr]::Zero -and (Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 150
        $script:h = [W]::FindMain([uint32]$script:p.Id)
    }
    if ($script:h -eq [IntPtr]::Zero) { throw "$label`: no window within 30 s" }
    Start-Sleep -Milliseconds 2500
    [W]::SetForegroundWindow($script:h) | Out-Null
    Start-Sleep -Milliseconds 300
    $script:s = [W]::GetDpiForWindow($script:h) / 96.0
    "-- $label (pid $($script:p.Id), scale $($script:s)): $(State)"
}
function Exited([int]$seconds = 6) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while (-not $script:p.HasExited -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 150; $script:p.Refresh() }
    $script:p.HasExited
}
function F11 { [W]::SetForegroundWindow($script:h) | Out-Null; Start-Sleep -Milliseconds 150; [W]::Key(0x7A, 0); Settle 1200 }

try {
    # ---- First launch: nothing remembered.
    Launch "first launch"
    $r0 = Rect
    Check "opens windowed" (-not (Zoomed) -and -not (Iconic) -and -not (Fullscreen)) (Fmt $r0)

    # Title-bar content still takes its own clicks: a breadcrumb navigates (the crumbs go away
    # on the top-level Runs page) and the search field opens the palette; neither drags.
    $c = [W]::Client($h)
    $crumbs = @([int]($c.Left + 240 * $s), [int]($c.Top + 6 * $s), [int](200 * $s), [int](28 * $s))
    $before = [W]::Fingerprint($crumbs[0], $crumbs[1], $crumbs[2], $crumbs[3])
    Settle 300
    $stable = $before -eq [W]::Fingerprint($crumbs[0], $crumbs[1], $crumbs[2], $crumbs[3])
    Check "title bar is still between frames (the checks below can trust it)" $stable ""
    [W]::Click([int]($c.Left + 268 * $s), (Bar)); Settle
    $after = [W]::Fingerprint($crumbs[0], $crumbs[1], $crumbs[2], $crumbs[3])
    Check "breadcrumb click navigates" ($before -ne $after -and (Same (Rect) $r0) -and -not (Zoomed)) (State)
    # The palette's scrim dims the whole window, the (static) sidebar brand too.
    $brand = @([int]($c.Left + 8 * $s), [int]($c.Top + 6 * $s), [int](200 * $s), [int](28 * $s))
    $before = [W]::Fingerprint($brand[0], $brand[1], $brand[2], $brand[3])
    Settle 300
    $stable = $before -eq [W]::Fingerprint($brand[0], $brand[1], $brand[2], $brand[3])
    [W]::Click([int]($c.Right - (138 + 12 + 140) * $s), (Bar)); Settle
    $after = [W]::Fingerprint($brand[0], $brand[1], $brand[2], $brand[3])
    Check "search field click opens the palette" ($stable -and $before -ne $after -and (Same (Rect) $r0)) (State)
    [W]::Key(0x1B, 0); Settle 600
    $closed = [W]::Fingerprint($brand[0], $brand[1], $brand[2], $brand[3])
    Check "Esc closes it" ($closed -eq $before) ""

    # Windows 11 shows its snap layouts over Maximize: a screenshot to look at (not checked).
    [W]::Hover((BtnX 1), (Bar)); Start-Sleep -Milliseconds 1800
    $snap = Join-Path $StateDir "snap-flyout.png"
    [W]::Shot($h, $snap)
    [W]::Hover((DragX), (Bar)); Settle 600
    "      (hovering Maximize: $snap)"

    # ---- Caption buttons.
    [W]::Click((BtnX 1), (Bar)); Settle
    Check "Maximize maximizes" (Zoomed) (State)
    [W]::Click((BtnX 1), (Bar)); Settle
    Check "Restore (same button) restores the size and place" ((-not (Zoomed)) -and (Same (Rect) $r0)) (State)

    # ---- Double-click and drag on both drag areas.
    [W]::DoubleClick((DragX), (Bar)); Settle
    Check "double-click title bar maximizes" (Zoomed) (State)
    [W]::DoubleClick((DragX), (Bar)); Settle
    Check "double-click again restores" ((-not (Zoomed)) -and (Same (Rect) $r0)) (State)
    [W]::DoubleClick((BrandX), (Bar)); Settle
    Check "double-click sidebar brand maximizes" (Zoomed) (State)
    [W]::DoubleClick((BrandX), (Bar)); Settle
    Check "double-click sidebar brand again restores" ((-not (Zoomed)) -and (Same (Rect) $r0)) (State)

    [W]::Drag((DragX), (Bar), 120, 40); Settle 600
    $r = Rect
    Check "drag title bar moves the window" ((Same $r (New-Object W+RECT -Property @{ Left = $r0.Left + 120; Top = $r0.Top + 40; Right = $r0.Right + 120; Bottom = $r0.Bottom + 40 }) 3)) (Fmt $r)
    $r1 = Rect
    [W]::Drag((BrandX), (Bar), -60, -20); Settle 600
    $r = Rect
    Check "drag sidebar brand moves the window" ((Same $r (New-Object W+RECT -Property @{ Left = $r1.Left - 60; Top = $r1.Top - 20; Right = $r1.Right - 60; Bottom = $r1.Bottom - 20 }) 3)) (Fmt $r)

    # ---- Top edge: the first pixels of the title bar resize, over the bar, a button and the brand.
    foreach ($spot in @(@("title bar", (DragX)), @("Close button", (BtnX 0)), @("sidebar brand", (BrandX)))) {
        $before = Rect
        $c = [W]::Client($h)
        [W]::Drag($spot[1], $c.Top + 2, 0, -50); Settle 600
        $r = Rect
        $ok = ($r.Top -le $before.Top - 40) -and ([Math]::Abs($r.Bottom - $before.Bottom) -le 2) -and (-not $p.HasExited)
        Check "top edge over the $($spot[0]) resizes" $ok ("{0} -> {1}" -f (Fmt $before), (Fmt $r))
    }
    # And just below the edge it is the title bar again: a drag moves, doesn't resize.
    $before = Rect
    $c = [W]::Client($h)
    [W]::Drag((DragX), [int]($c.Top + 9 * $s), 30, 30); Settle 600
    $r = Rect
    Check "just below the top edge drags" (($r.Left - $before.Left) -ge 27 -and (($r.Right - $r.Left) -eq ($before.Right - $before.Left)) -and (($r.Bottom - $r.Top) -eq ($before.Bottom - $before.Top))) ("{0} -> {1}" -f (Fmt $before), (Fmt $r))

    # ---- Minimize.
    $rw = Rect
    [W]::Click((BtnX 2), (Bar)); Settle
    Check "Minimize minimizes" (Iconic) (State)
    [W]::ShowWindow($h, 9) | Out-Null; Settle 700
    [W]::SetForegroundWindow($h) | Out-Null
    Check "restores from the taskbar as it was" ((-not (Iconic)) -and (Same (Rect) $rw)) (State)

    # ---- F11 and the caption buttons in full screen.
    F11
    Check "F11 enters full screen" (Fullscreen) (State)
    F11
    Check "F11 again leaves it, as it was" ((-not (Fullscreen)) -and (Same (Rect) $rw)) (State)
    F11
    [W]::Click((BtnX 1), (Bar)); Settle 1200
    Check "full screen: Restore button leaves full screen" ((-not (Fullscreen)) -and (Same (Rect) $rw) -and -not (Zoomed)) (State)
    F11
    [W]::Click((BtnX 2), (Bar)); Settle
    Check "full screen: Minimize minimizes" (Iconic) (State)
    [W]::ShowWindow($h, 9) | Out-Null; Settle 900
    [W]::SetForegroundWindow($h) | Out-Null
    Check "full screen: back from the taskbar still full screen" ((-not (Iconic)) -and (Fullscreen)) (State)
    F11
    Check "F11 leaves full screen" ((-not (Fullscreen)) -and (Same (Rect) $rw)) (State)

    # ---- Close while maximized; the next launch is maximized and restores to $rw.
    [W]::Click((BtnX 1), (Bar)); Settle
    Check "Maximize before closing" (Zoomed) (State)
    [W]::Click((BtnX 0), (Bar))
    Check "Close closes the app" (Exited) ""
    Check "state saved where FOLO_DESKTOP_STATE says" (Test-Path $stateFile) $stateFile

    Launch "second launch"
    Check "relaunch: maximized again" (Zoomed) (State)
    [W]::Click((BtnX 1), (Bar)); Settle
    Check "relaunch: restores to the last size and place" ((-not (Zoomed)) -and (Same (Rect) $rw)) ("{0}, wanted {1}" -f (Fmt (Rect)), (Fmt $rw))
    [W]::Click((BtnX 0), (Bar))
    Check "Close (windowed) closes the app" (Exited) ""

    Launch "third launch"
    Check "relaunch: windowed at the last size and place" ((-not (Zoomed)) -and (-not (Fullscreen)) -and (Same (Rect) $rw)) ("{0}, wanted {1}" -f (Fmt (Rect)), (Fmt $rw))
    F11
    [W]::Click((BtnX 0), (Bar))
    Check "full screen: Close closes the app" (Exited) ""

    Launch "fourth launch"
    Check "relaunch: full screen again" (Fullscreen) (State)
    F11
    Check "relaunch: leaving full screen gives the last size and place" ((-not (Fullscreen)) -and (Same (Rect) $rw)) ("{0}, wanted {1}" -f (Fmt (Rect)), (Fmt $rw))
    [W]::Click((BtnX 0), (Bar))
    Check "Close closes the app" (Exited) ""

    # ---- Closed while minimized (the taskbar's Close): it comes back as it was before minimizing.
    Launch "launch to minimize maximized"
    [W]::Click((BtnX 1), (Bar)); Settle
    [W]::Click((BtnX 2), (Bar)); Settle
    Check "maximized, then minimized" (Iconic) (State)
    [W]::PostMessageW($h, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null   # WM_CLOSE
    Check "closes while minimized" (Exited) ""
    Launch "relaunch after closing minimized from maximized"
    Check "relaunch: maximized, as before minimizing" (Zoomed) (State)
    [W]::Click((BtnX 1), (Bar)); Settle
    Check "relaunch: restores to the last size and place" ((-not (Zoomed)) -and (Same (Rect) $rw)) ("{0}, wanted {1}" -f (Fmt (Rect)), (Fmt $rw))
    F11
    [W]::Click((BtnX 2), (Bar)); Settle
    Check "full screen, then minimized" (Iconic) (State)
    [W]::PostMessageW($h, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null
    Check "closes while minimized" (Exited) ""
    Launch "relaunch after closing minimized from full screen"
    Check "relaunch: full screen, as before minimizing" (Fullscreen) (State)
    F11
    Check "relaunch: leaving full screen gives the last size and place" ((-not (Fullscreen)) -and (Same (Rect) $rw)) ("{0}, wanted {1}" -f (Fmt (Rect)), (Fmt $rw))
    [W]::Click((BtnX 0), (Bar))
    Check "Close closes the app" (Exited) ""

    # ---- A saved place on no display opens centred on the primary display.
    $saved = Get-Content -Raw $stateFile | ConvertFrom-Json
    $saved.window.x = 40000
    $saved.window.y = 40000
    $saved.window.mode = "windowed"
    [System.IO.File]::WriteAllText($stateFile, ($saved | ConvertTo-Json -Depth 4))
    Launch "fifth launch (saved place off screen)"
    $m = [W]::Monitor($h); $r = Rect
    $cx = ($r.Left + $r.Right) / 2 - ($m.Left + $m.Right) / 2
    $cy = ($r.Top + $r.Bottom) / 2 - ($m.Top + $m.Bottom) / 2
    Check "off-screen place: centred, last size" (([Math]::Abs($cx) -le 20) -and ([Math]::Abs($cy) -le 20) -and (($r.Right - $r.Left) -eq ($rw.Right - $rw.Left))) (Fmt $r)
    [W]::Click((BtnX 0), (Bar))
    Check "Close closes the app" (Exited) ""

    # ---- A corrupt file is a first launch.
    [System.IO.File]::WriteAllText($stateFile, "{ not json")
    Launch "sixth launch (corrupt state file)"
    Check "corrupt state: opens windowed" ((-not (Zoomed)) -and (-not (Fullscreen))) (State)
    [W]::Click((BtnX 0), (Bar))
    Check "Close closes the app" (Exited) ""
    Check "corrupt state: replaced on close" ((Get-Content -Raw $stateFile) -match '"window"') ""
}
catch {
    Check "script ran to the end" $false "$_"
}
finally {
    if ($script:p) {
        $script:p.Refresh()
        if (-not $script:p.HasExited) { $script:p.Kill() }
    }
}
$realAfter = if (Test-Path $realFile) { (Get-Item $realFile).LastWriteTimeUtc.Ticks } else { 0 }
Check "real profile's desktop.json untouched" ($realBefore -eq $realAfter) $realFile

if ($script:failed.Count) {
    "$($script:failed.Count) check(s) failed"
    exit 1
}
"all checks passed"
exit 0
