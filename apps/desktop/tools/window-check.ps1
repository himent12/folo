# Drives FOLO Desktop's window chrome and checks what the window did: the caption buttons, double-click
# and drag on both drag areas, the top resize edge, F11 full screen and its caption buttons, the
# breadcrumbs and search field inside the title bar, the same with the palette and a dialog open (over
# their scrim), and restoring the size, position and maximized / full-screen state on the next launch.
#
#   tools\window-check.ps1                       # $CARGO_TARGET_DIR\debug\folo-desktop.exe, real input
#   tools\window-check.ps1 -Exe D:\b\release\folo-desktop.exe
#   tools\window-check.ps1 -Exe D:\b\release\folo-desktop.exe -Sandbox
#
# Two ways to drive it:
# - By default, with real mouse and keyboard input on your screen: it moves the real pointer, so leave
#   the mouse alone while it runs (about a minute).
# - -Sandbox: on a private desktop nobody sees (sandbox-run.ps1), with window messages only (drive.ps1),
#   since real input cannot reach that desktop. Each real action becomes the messages Windows would
#   produce for it: a press is hit tested (WM_NCHITTEST) and posted as WM_(NC)LBUTTONDOWN / UP, a double
#   click on a caption adds WM_NCLBUTTONDBLCLK, keys are WM_KEYDOWN / UP with the modifiers in the UI
#   thread's key state, "restore from the taskbar" is WM_SYSCOMMAND SC_RESTORE, and what the screen shows
#   is read with PrintWindow. Two things only Windows does with the real mouse are replaced: a drag on a
#   caption area or the top edge is the press (which the app must leave to Windows) followed by the move
#   or resize Windows' loop would make (SetWindowPos), so the app's part, the hit test, the press and
#   following the new place, is what is checked; and the snap-layouts flyout over Maximize is not shown.
#
# The app remembers its window in $FOLO_DESKTOP_STATE, pointed here at a fresh file under -StateDir
# ($FOLO_TEST_TMP\window-check by default), never the real profile. Exits 1 if any check fails.
param(
    [string]$Exe = "",
    [string]$StateDir = "",
    [string]$ExtraArgs = "--mock --open run:run_a19f",
    [switch]$Sandbox,
    [switch]$InSandbox
)
$ErrorActionPreference = "Stop"
if (-not $Exe) {
    $target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $PSScriptRoot "..\target" }
    $Exe = Join-Path $target "debug\folo-desktop.exe"
}
if (-not (Test-Path $Exe)) { throw "no app at $Exe (build it, or pass -Exe)" }
$Exe = (Resolve-Path $Exe).Path
if (-not $StateDir) {
    $tmp = if ($env:FOLO_TEST_TMP) { $env:FOLO_TEST_TMP } else { [System.IO.Path]::GetTempPath() }
    $StateDir = Join-Path $tmp "window-check"
}
if ($Sandbox -and -not $InSandbox) {
    $self = $MyInvocation.MyCommand.Path
    & (Join-Path $PSScriptRoot "sandbox-run.ps1") -TimeoutS 600 -Command "& '$self' -Exe '$Exe' -StateDir '$StateDir' -ExtraArgs '$ExtraArgs' -InSandbox"
    exit $LASTEXITCODE
}
$msg = [bool]$InSandbox
if (Test-Path $StateDir) { Remove-Item -Recurse -Force $StateDir }
New-Item -ItemType Directory -Force $StateDir | Out-Null
$stateFile = Join-Path $StateDir "desktop.json"
# The app's launches inherit it; put back when the script ends (the `finally` below).
$state0 = $env:FOLO_DESKTOP_STATE

# The real profile's file must not change.
$home_ = if ($env:FOLO_HOME) { $env:FOLO_HOME } else { Join-Path $env:USERPROFILE ".folo" }
$realFile = Join-Path $home_ "desktop.json"
$realBefore = if (Test-Path $realFile) { (Get-Item $realFile).LastWriteTimeUtc.Ticks } else { 0 }

. (Join-Path $PSScriptRoot "drive.ps1")
if (-not $msg -and -not ("W" -as [type])) {
    Add-Type -ReferencedAssemblies System.Drawing @"
using System;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Threading;
public static class W {
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
    [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint dx, uint dy, uint data, UIntPtr extra);
    [DllImport("user32.dll")] public static extern void keybd_event(byte vk, byte scan, uint flags, UIntPtr extra);
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
    public static void Shot(int l, int t, int r, int b, string path) {
        using (var bmp = new Bitmap(r - l, b - t)) {
            using (var g = Graphics.FromImage(bmp)) g.CopyFromScreen(l, t, 0, 0, bmp.Size);
            bmp.Save(path);
        }
    }
}
"@
}

$script:failed = @()
$script:p = $null
$script:h = [IntPtr]::Zero
$script:s = 1.0

function Check([string]$name, [bool]$ok, [string]$detail = "") {
    $mark = if ($ok) { "PASS" } else { "FAIL" }
    "{0}  {1,-58} {2}" -f $mark, $name, $detail
    if (-not $ok) { $script:failed += $name }
}
function Rect { [FoloDrive]::Rect($script:h) }
function Fmt($r) { "({0},{1}) {2}x{3}" -f $r.Left, $r.Top, ($r.Right - $r.Left), ($r.Bottom - $r.Top) }
function Same($a, $b, [int]$tol = 2) {
    [Math]::Abs($a.Left - $b.Left) -le $tol -and [Math]::Abs($a.Top - $b.Top) -le $tol -and
    [Math]::Abs($a.Right - $b.Right) -le $tol -and [Math]::Abs($a.Bottom - $b.Bottom) -le $tol
}
function Monitor {
    $r = Rect; $cx = ($r.Left + $r.Right) / 2; $cy = ($r.Top + $r.Bottom) / 2
    $ms = [FoloDrive]::Monitors()
    foreach ($m in $ms) { if ($cx -ge $m.Left -and $cx -lt $m.Right -and $cy -ge $m.Top -and $cy -lt $m.Bottom) { return $m } }
    $ms[0]
}
function Zoomed { [FoloDrive]::IsZoomed($script:h) }
function Iconic { [FoloDrive]::IsIconic($script:h) }
function Fullscreen { Same (Rect) (Monitor) 0 }
function State { "zoomed={0} iconic={1} full={2} rect={3}" -f (Zoomed), (Iconic), (Fullscreen), (Fmt (Rect)) }
function Settle([int]$ms = 900) { Start-Sleep -Milliseconds $ms }
function Client { [FoloDrive]::Client($script:h) }

# ---- input: real (default) or the window messages it produces (-Sandbox)
function ToClient([int]$x, [int]$y) { $c = Client; @(($x - $c.Left), ($y - $c.Top)) }
function Hover([int]$x, [int]$y) {
    if (-not $msg) { [W]::Hover($x, $y); return }
    $hit = [FoloDrive]::HitTest($script:h, $x, $y)
    if ($hit -eq 1) { $pt = ToClient $x $y; [FoloDrive]::Move($script:h, $pt[0], $pt[1], 0) }
}
function Click([int]$x, [int]$y) {
    if (-not $msg) { [W]::Click($x, $y); return }
    $hit = [FoloDrive]::HitTest($script:h, $x, $y)
    if ($hit -eq 1) { $pt = ToClient $x $y; [FoloDrive]::Click($script:h, $pt[0], $pt[1]) }
    else { [FoloDrive]::NcClick($script:h, $x, $y, $hit, $false) }
}
function DoubleClick([int]$x, [int]$y) {
    if (-not $msg) { [W]::DoubleClick($x, $y); return }
    $hit = [FoloDrive]::HitTest($script:h, $x, $y)
    if ($hit -eq 1) { $pt = ToClient $x $y; [FoloDrive]::Click($script:h, $pt[0], $pt[1]); [FoloDrive]::Click($script:h, $pt[0], $pt[1]) }
    else { [FoloDrive]::NcClick($script:h, $x, $y, $hit, $true) }
}
# A drag from (x, y) by (dx, dy). With messages, on a caption area or a resize edge: the press goes to
# the app (which must leave it to Windows), then the window moves or resizes as Windows' loop would.
# Returns the hit test answer at the press.
function Drag([int]$x, [int]$y, [int]$dx, [int]$dy) {
    if (-not $msg) { [W]::Drag($x, $y, $dx, $dy); return 0 }
    $hit = [FoloDrive]::HitTest($script:h, $x, $y)
    if ($hit -eq 1) { $pt = ToClient $x $y; [FoloDrive]::Drag($script:h, $pt[0], $pt[1], $dx, $dy, 12); return $hit }
    [FoloDrive]::Post($script:h, [FoloDrive]::WM_NCLBUTTONDOWN, [IntPtr]$hit, [FoloDrive]::Pt($x, $y)); [FoloDrive]::Sync($script:h)
    # A resize-edge press the app left to Windows starts its size loop (a caption press starts the move
    # loop only once the real button moves, so there is nothing to see for it); end it, it waits for the
    # real mouse.
    $script:loop = [FoloDrive]::InMoveSize($script:h)
    if ($script:loop) { [FoloDrive]::EndMoveSize($script:h) }
    $r = Rect
    switch ($hit) {
        2 { [FoloDrive]::MoveWindow($script:h, $r.Left + $dx, $r.Top + $dy, $r.Right - $r.Left, $r.Bottom - $r.Top) }   # HTCAPTION
        12 { [FoloDrive]::MoveWindow($script:h, $r.Left, $r.Top + $dy, $r.Right - $r.Left, $r.Bottom - $r.Top - $dy) }  # HTTOP
    }
    [FoloDrive]::Sync($script:h)
    $hit
}
# vk with an optional modifier vk (0x11 Ctrl).
function Key([byte]$vk, [byte]$mod) {
    if (-not $msg) { [W]::Key($vk, $mod); return }
    $mods = switch ($mod) { 0x11 { 1 } 0x10 { 2 } 0x12 { 4 } default { 0 } }
    [FoloDrive]::Key($script:h, $vk, $mods)
}
function RestoreFromTaskbar {
    if (-not $msg) { [W]::ShowWindow($script:h, 9) | Out-Null; Settle 900; [W]::SetForegroundWindow($script:h) | Out-Null; Settle 300; return }
    [FoloDrive]::Post($script:h, 0x112, [IntPtr]0xF120, [IntPtr]::Zero); Settle 900   # WM_SYSCOMMAND SC_RESTORE
}
function Foreground { if (-not $msg) { [W]::SetForegroundWindow($script:h) | Out-Null; Start-Sleep -Milliseconds 150 } }
# A fingerprint of what a screen rectangle of the window shows.
function Fingerprint([int]$x, [int]$y, [int]$w, [int]$hh) {
    if (-not $msg) { return [W]::Fingerprint($x, $y, $w, $hh) }
    $c = Client
    $b = [FoloDrive]::Capture($script:h)
    try { [FoloDrive]::Hash($b, $x - $c.Left, $y - $c.Top, $w, $hh) } finally { $b.Dispose() }
}

# Points in the title bar (physical px), from the client area: the bar is 40 px tall, the caption
# buttons 46 px wide at the right, the search field 280 px left of them, the sidebar 232 px wide.
function Bar { $c = Client; [int]($c.Top + 20 * $script:s) }
function BtnX([int]$i) { $c = Client; [int]($c.Right - (46 * $i + 23) * $script:s) }  # 0 close, 1 max, 2 min
function DragX { $c = Client; [int]($c.Right - 500 * $script:s) }
function BrandX { $c = Client; [int]($c.Left + 180 * $script:s) }

function Launch([string]$label) {
    # A Close that failed leaves the last one running: stop it (only ever our own process).
    Stop-Folo $script:p
    $script:p = Start-Process -FilePath $Exe -ArgumentList $ExtraArgs.Split(" ") -PassThru
    $null = $script:p.Handle
    $script:h = Wait-FoloWindow $script:p 30
    if ($script:h -eq [IntPtr]::Zero) { throw "$label`: no window within 30 s" }
    Start-Sleep -Milliseconds 2500
    Foreground
    $script:s = [FoloDrive]::GetDpiForWindow($script:h) / 96.0
    "-- $label (pid $($script:p.Id), scale $($script:s)): $(State)"
}
function Exited([int]$seconds = 6) { Wait-FoloExit $script:p $seconds }
function F11 { Foreground; Key 0x7A 0; Settle 1200 }
# The sidebar's (static) brand row, which an overlay's scrim dims: whether one is open.
function BrandFp {
    $c = Client
    Fingerprint ([int]($c.Left + 8 * $script:s)) ([int]($c.Top + 6 * $script:s)) ([int](200 * $script:s)) ([int](28 * $script:s))
}
function Moved($r, [int]$dx, [int]$dy) {
    New-Object FoloDrive+RECT -Property @{ Left = $r.Left + $dx; Top = $r.Top + $dy; Right = $r.Right + $dx; Bottom = $r.Bottom + $dy }
}
# A drag on a caption area moved the window by (dx, dy); with messages the press was a caption press too.
function DragCheck([string]$name, $hit, $r, [int]$dx, [int]$dy, [string]$extra = "", [bool]$alsoOk = $true) {
    $ok = (Same (Rect) (Moved $r $dx $dy) 3) -and $alsoOk
    $how = ""
    if ($msg) { $ok = $ok -and ($hit -eq 2); $how = " (hit $hit; moved by SetWindowPos, as Windows' loop)" }
    Check $name $ok "$(State)$how $extra"
}

# Launches, opens an overlay with Ctrl + $key, and drives the caption areas over its scrim.
function OverlayChecks([string]$what, [byte]$key) {
    Launch "launch with the $what"
    $r0 = Rect
    $plain = BrandFp
    Key $key 0x11; Settle
    $dim = BrandFp
    Check "${what}: opens and dims the title bar" ($dim -ne $plain) ""
    $c = Client
    # The dimmed search field is part of the drag band: a click neither reaches it nor dismisses.
    Click ([int]($c.Right - (138 + 12 + 140) * $script:s)) (Bar); Settle
    Check "${what}: a click on the dimmed search field does nothing" (((BrandFp) -eq $dim) -and (Same (Rect) $r0)) (State)
    Click (BtnX 1) (Bar); Settle
    Check "${what}: Maximize maximizes, it stays open" ((Zoomed) -and ((BrandFp) -eq $dim)) (State)
    Click (BtnX 1) (Bar); Settle
    Check "${what}: Restore restores, it stays open" ((-not (Zoomed)) -and (Same (Rect) $r0) -and ((BrandFp) -eq $dim)) (State)
    $hit = Drag (DragX) (Bar) 120 40; Settle 600
    DragCheck "${what}: drag title bar moves the window, it stays open" $hit $r0 120 40 "" ((BrandFp) -eq $dim)
    $r1 = Rect
    $hit = Drag (BrandX) (Bar) -60 -20; Settle 600
    DragCheck "${what}: drag sidebar brand moves the window, it stays open" $hit $r1 -60 -20 "" ((BrandFp) -eq $dim)
    $rw = Rect
    Click (BtnX 2) (Bar); Settle
    Check "${what}: Minimize minimizes" (Iconic) (State)
    RestoreFromTaskbar
    Check "${what}: back from the taskbar, still open" ((-not (Iconic)) -and (Same (Rect) $rw) -and ((BrandFp) -eq $dim)) (State)
    # The dimmed page still dismisses it (left of the 720 px New run dialog in a 1280 px window).
    $c = Client
    Click ([int]($c.Left + 256 * $script:s)) ([int]($c.Top + ($c.Bottom - $c.Top) * 0.6)); Settle
    Check "${what}: a click on the dimmed page dismisses it" ((BrandFp) -eq $plain) ""
    Key $key 0x11; Settle
    Check "${what}: opens again" ((BrandFp) -eq $dim) ""
    Click (BtnX 0) (Bar)
    Check "${what}: Close closes the app" (Exited) ""
}

$env:FOLO_DESKTOP_STATE = $stateFile
try {
    "mode: $(if ($msg) { 'window messages on a private desktop' } else { 'real input on this screen' })"
    # ---- First launch: nothing remembered.
    Launch "first launch"
    $r0 = Rect
    Check "opens windowed" (-not (Zoomed) -and -not (Iconic) -and -not (Fullscreen)) (Fmt $r0)

    # Title-bar content still takes its own clicks: a breadcrumb navigates (the crumbs go away
    # on the top-level Runs page) and the search field opens the palette; neither drags.
    $c = Client
    $crumbs = @([int]($c.Left + 240 * $s), [int]($c.Top + 6 * $s), [int](200 * $s), [int](28 * $s))
    $before = Fingerprint $crumbs[0] $crumbs[1] $crumbs[2] $crumbs[3]
    Settle 300
    $stable = $before -eq (Fingerprint $crumbs[0] $crumbs[1] $crumbs[2] $crumbs[3])
    Check "title bar is still between frames (the checks below can trust it)" $stable ""
    Click ([int]($c.Left + 268 * $s)) (Bar); Settle
    $after = Fingerprint $crumbs[0] $crumbs[1] $crumbs[2] $crumbs[3]
    Check "breadcrumb click navigates" ($before -ne $after -and (Same (Rect) $r0) -and -not (Zoomed)) (State)
    # The palette's scrim dims the whole window, the (static) sidebar brand too.
    $brand = @([int]($c.Left + 8 * $s), [int]($c.Top + 6 * $s), [int](200 * $s), [int](28 * $s))
    $before = Fingerprint $brand[0] $brand[1] $brand[2] $brand[3]
    Settle 300
    $stable = $before -eq (Fingerprint $brand[0] $brand[1] $brand[2] $brand[3])
    Click ([int]($c.Right - (138 + 12 + 140) * $s)) (Bar); Settle
    $after = Fingerprint $brand[0] $brand[1] $brand[2] $brand[3]
    Check "search field click opens the palette" ($stable -and $before -ne $after -and (Same (Rect) $r0)) (State)
    Key 0x1B 0; Settle 600
    $closed = Fingerprint $brand[0] $brand[1] $brand[2] $brand[3]
    Check "Esc closes it" ($closed -eq $before) ""

    # Keyboard mode: focus went back to the search field. An arrow key turns keyboard mode on (its
    # focus ring shows); a press on the title bar's drag area turns it off, as any press does.
    Hover (DragX) (Bar); Settle 400
    $field = @([int]($c.Right - (138 + 12 + 280 + 6) * $s), [int]($c.Top + 2 * $s), [int](292 * $s), [int](36 * $s))
    $plain = Fingerprint $field[0] $field[1] $field[2] $field[3]
    Key 0x26 0; Settle 500   # Up
    $ring = Fingerprint $field[0] $field[1] $field[2] $field[3]
    Click (DragX) (Bar); Settle 600
    $off = Fingerprint $field[0] $field[1] $field[2] $field[3]
    Check "a key shows the focus ring" ($ring -ne $plain) ""
    Check "a press on the title bar leaves keyboard mode" (($off -eq $plain) -and (Same (Rect) $r0) -and -not (Zoomed)) (State)
    Check "the window has FOLO's icon" ([FoloDrive]::GetClassLongPtrW($h, -14) -ne [IntPtr]::Zero) ""

    if (-not $msg) {
        # Windows 11 shows its snap layouts over Maximize: a screenshot to look at (not checked).
        Hover (BtnX 1) (Bar); Start-Sleep -Milliseconds 1800
        $snap = Join-Path $StateDir "snap-flyout.png"
        $m = Monitor
        [W]::Shot($m.Left, $m.Top, $m.Right, $m.Bottom, $snap)
        Hover (DragX) (Bar); Settle 600
        "      (hovering Maximize: $snap)"
    }

    # ---- Caption buttons.
    Click (BtnX 1) (Bar); Settle
    Check "Maximize maximizes" (Zoomed) (State)
    Click (BtnX 1) (Bar); Settle
    Check "Restore (same button) restores the size and place" ((-not (Zoomed)) -and (Same (Rect) $r0)) (State)

    # ---- Double-click and drag on both drag areas.
    DoubleClick (DragX) (Bar); Settle
    Check "double-click title bar maximizes" (Zoomed) (State)
    DoubleClick (DragX) (Bar); Settle
    Check "double-click again restores" ((-not (Zoomed)) -and (Same (Rect) $r0)) (State)
    DoubleClick (BrandX) (Bar); Settle
    Check "double-click sidebar brand maximizes" (Zoomed) (State)
    DoubleClick (BrandX) (Bar); Settle
    Check "double-click sidebar brand again restores" ((-not (Zoomed)) -and (Same (Rect) $r0)) (State)

    $hit = Drag (DragX) (Bar) 120 40; Settle 600
    DragCheck "drag title bar moves the window" $hit $r0 120 40
    $r1 = Rect
    $hit = Drag (BrandX) (Bar) -60 -20; Settle 600
    DragCheck "drag sidebar brand moves the window" $hit $r1 -60 -20

    # ---- Top edge: the first pixels of the title bar resize, over the bar, a button and the brand.
    foreach ($spot in @(@("title bar", (DragX)), @("Close button", (BtnX 0)), @("sidebar brand", (BrandX)))) {
        $before = Rect
        $c = Client
        $hit = Drag $spot[1] ($c.Top + 2) 0 -50; Settle 600
        $r = Rect
        $ok = ($r.Top -le $before.Top - 40) -and ([Math]::Abs($r.Bottom - $before.Bottom) -le 2) -and (-not $p.HasExited)
        $how = ""
        if ($msg) {
            $ok = $ok -and ($hit -eq 12) -and $script:loop
            # The app followed: it drew the new size (the capture has it, and is not blank).
            $b = [FoloDrive]::Capture($h); $cr = Client
            $ok = $ok -and ($b.Height -eq ($cr.Bottom - $cr.Top)) -and ([FoloDrive]::Hash($b, 0, 0, 64, 64) -ne 0)
            $b.Dispose()
            $how = " (hit $hit, Windows' size loop started: $($script:loop); resized by SetWindowPos)"
        }
        Check "top edge over the $($spot[0]) resizes" $ok ("{0} -> {1}{2}" -f (Fmt $before), (Fmt $r), $how)
    }
    # And just below the edge it is the title bar again: a drag moves, doesn't resize.
    $before = Rect
    $c = Client
    $hit = Drag (DragX) ([int]($c.Top + 9 * $s)) 30 30; Settle 600
    $r = Rect
    $ok = ($r.Left - $before.Left) -ge 27 -and (($r.Right - $r.Left) -eq ($before.Right - $before.Left)) -and (($r.Bottom - $r.Top) -eq ($before.Bottom - $before.Top))
    if ($msg) { $ok = $ok -and ($hit -eq 2) }
    Check "just below the top edge drags" $ok ("{0} -> {1}{2}" -f (Fmt $before), (Fmt $r), $(if ($msg) { " (hit $hit)" } else { "" }))

    # ---- Minimize.
    $rw = Rect
    Click (BtnX 2) (Bar); Settle
    Check "Minimize minimizes" (Iconic) (State)
    RestoreFromTaskbar
    Check "restores from the taskbar as it was" ((-not (Iconic)) -and (Same (Rect) $rw)) (State)

    # ---- F11 and the caption buttons in full screen.
    F11
    Check "F11 enters full screen" (Fullscreen) (State)
    F11
    Check "F11 again leaves it, as it was" ((-not (Fullscreen)) -and (Same (Rect) $rw)) (State)
    F11
    Click (BtnX 1) (Bar); Settle 1200
    Check "full screen: Restore button leaves full screen" ((-not (Fullscreen)) -and (Same (Rect) $rw) -and -not (Zoomed)) (State)
    F11
    Click (BtnX 2) (Bar); Settle
    Check "full screen: Minimize minimizes" (Iconic) (State)
    RestoreFromTaskbar
    Check "full screen: back from the taskbar still full screen" ((-not (Iconic)) -and (Fullscreen)) (State)
    F11
    Check "F11 leaves full screen" ((-not (Fullscreen)) -and (Same (Rect) $rw)) (State)

    # ---- Close while maximized; the next launch is maximized and restores to $rw.
    Click (BtnX 1) (Bar); Settle
    Check "Maximize before closing" (Zoomed) (State)
    Click (BtnX 0) (Bar)
    Check "Close closes the app" (Exited) ""
    Check "state saved where FOLO_DESKTOP_STATE says" (Test-Path $stateFile) $stateFile

    Launch "second launch"
    Check "relaunch: maximized again" (Zoomed) (State)
    Click (BtnX 1) (Bar); Settle
    Check "relaunch: restores to the last size and place" ((-not (Zoomed)) -and (Same (Rect) $rw)) ("{0}, wanted {1}" -f (Fmt (Rect)), (Fmt $rw))
    Click (BtnX 0) (Bar)
    Check "Close (windowed) closes the app" (Exited) ""

    Launch "third launch"
    Check "relaunch: windowed at the last size and place" ((-not (Zoomed)) -and (-not (Fullscreen)) -and (Same (Rect) $rw)) ("{0}, wanted {1}" -f (Fmt (Rect)), (Fmt $rw))
    F11
    Click (BtnX 0) (Bar)
    Check "full screen: Close closes the app" (Exited) ""

    Launch "fourth launch"
    Check "relaunch: full screen again" (Fullscreen) (State)
    F11
    Check "relaunch: leaving full screen gives the last size and place" ((-not (Fullscreen)) -and (Same (Rect) $rw)) ("{0}, wanted {1}" -f (Fmt (Rect)), (Fmt $rw))
    Click (BtnX 0) (Bar)
    Check "Close closes the app" (Exited) ""

    # ---- Closed while minimized (the taskbar's Close): it comes back as it was before minimizing.
    Launch "launch to minimize maximized"
    Click (BtnX 1) (Bar); Settle
    Click (BtnX 2) (Bar); Settle
    Check "maximized, then minimized" (Iconic) (State)
    [FoloDrive]::Post($h, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero)   # WM_CLOSE
    Check "closes while minimized" (Exited) ""
    Launch "relaunch after closing minimized from maximized"
    Check "relaunch: maximized, as before minimizing" (Zoomed) (State)
    Click (BtnX 1) (Bar); Settle
    Check "relaunch: restores to the last size and place" ((-not (Zoomed)) -and (Same (Rect) $rw)) ("{0}, wanted {1}" -f (Fmt (Rect)), (Fmt $rw))
    F11
    Click (BtnX 2) (Bar); Settle
    Check "full screen, then minimized" (Iconic) (State)
    [FoloDrive]::Post($h, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero)
    Check "closes while minimized" (Exited) ""
    Launch "relaunch after closing minimized from full screen"
    Check "relaunch: full screen, as before minimizing" (Fullscreen) (State)
    F11
    Check "relaunch: leaving full screen gives the last size and place" ((-not (Fullscreen)) -and (Same (Rect) $rw)) ("{0}, wanted {1}" -f (Fmt (Rect)), (Fmt $rw))
    Click (BtnX 0) (Bar)
    Check "Close closes the app" (Exited) ""

    # ---- A saved place on no display opens centred on the primary display.
    $saved = Get-Content -Raw $stateFile | ConvertFrom-Json
    $saved.window.x = 40000
    $saved.window.y = 40000
    $saved.window.mode = "windowed"
    [System.IO.File]::WriteAllText($stateFile, ($saved | ConvertTo-Json -Depth 4))
    Launch "fifth launch (saved place off screen)"
    $m = Monitor; $r = Rect
    $cx = ($r.Left + $r.Right) / 2 - ($m.Left + $m.Right) / 2
    $cy = ($r.Top + $r.Bottom) / 2 - ($m.Top + $m.Bottom) / 2
    Check "off-screen place: centred, last size" (([Math]::Abs($cx) -le 20) -and ([Math]::Abs($cy) -le 20) -and (($r.Right - $r.Left) -eq ($rw.Right - $rw.Left))) (Fmt $r)
    Click (BtnX 0) (Bar)
    Check "Close closes the app" (Exited) ""

    # ---- A corrupt file is a first launch.
    [System.IO.File]::WriteAllText($stateFile, "{ not json")
    Launch "sixth launch (corrupt state file)"
    Check "corrupt state: opens windowed" ((-not (Zoomed)) -and (-not (Fullscreen))) (State)
    Click (BtnX 0) (Bar)
    Check "Close closes the app" (Exited) ""
    Check "corrupt state: replaced on close" ((Get-Content -Raw $stateFile) -match '"window"') ""

    # ---- With an overlay open (the palette, a dialog), its scrim dims the title bar but the window
    # still moves, maximizes, minimizes and closes, and none of that dismisses the overlay.
    OverlayChecks "palette" 0x4B   # Ctrl K
    OverlayChecks "New run dialog" 0x4E   # Ctrl N
}
catch {
    Check "script ran to the end" $false "$_ $($_.InvocationInfo.PositionMessage)"
}
finally {
    Stop-Folo $script:p
    $env:FOLO_DESKTOP_STATE = $state0
}
$realAfter = if (Test-Path $realFile) { (Get-Item $realFile).LastWriteTimeUtc.Ticks } else { 0 }
Check "real profile's desktop.json untouched" ($realBefore -eq $realAfter) $realFile

if ($script:failed.Count) {
    "$($script:failed.Count) check(s) failed"
    exit 1
}
"all checks passed"
exit 0
