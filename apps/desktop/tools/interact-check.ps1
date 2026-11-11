# Drives the real FOLO Desktop window with window messages only (drive.ps1) and checks what it shows:
# its captures equal the headless `--shot` of the same scenario, and hover, clicks, the palette and its
# text field, focus (Tab, F6), wheel scrolling, a slider, a select, tooltips, the New run dialog, a
# toast, the theme toggle and resizing behave. Prints PASS / FAIL per step; exit code 1 if one fails.
#
#   tools\interact-check.ps1 -Exe D:\b\release\folo-desktop.exe                       # off screen, never active
#   tools\interact-check.ps1 -Exe D:\b\release\folo-desktop.exe -Sandbox              # on a private desktop
#   tools\interact-check.ps1 -Exe D:\b\release\folo-desktop.exe -Sandbox -Clipboard   # and copy and paste
#
# Off screen (FOLO_UI_OFFSCREEN=1) the window is outside every display and never activated; nothing
# touches the cursor, the keyboard or the clipboard of the person at the PC. -Sandbox runs the same on a
# private desktop (sandbox-run.ps1). The file dialog step runs there; the clipboard steps (copy and
# paste) run only with -Clipboard as well, because the clipboard is not private: the private desktop
# shares it with you.
#
# -Clipboard REPLACES YOUR CLIPBOARD for a few seconds. Before the run this script saves every format
# byte for byte (clip-save.ps1; it skips the steps when a format cannot be restored exactly), and after
# the run, however the run ended (passed, failed, threw, timed out and was killed, Ctrl+C), it puts it
# back, unless someone copied something meanwhile (theirs stays). What you had copied is lost if this
# PowerShell itself dies during those seconds (its window closed, the process killed, the PC going
# down): the saved copy lives only in its memory. Off by default.
#
# "Equal" means no pixel differs by more than 8 levels (the breathing status dot's pulse; see the GPUI
# baseline's noise floor). Scratch files go under -Out ($FOLO_TEST_TMP\interact-check by default).
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [string]$Out = "",
    [string[]]$Only = @(),
    [switch]$Sandbox,
    [switch]$Clipboard,
    [switch]$InSandbox,
    # Set by the outer run for the sandboxed one: where the clipboard steps record their own changes, and
    # the clipboard's sequence number when it was saved.
    [string]$ClipLedger = "",
    [uint32]$ClipSeq = 0
)
$ErrorActionPreference = "Stop"
$Exe = (Resolve-Path $Exe).Path
$tmp = if ($env:FOLO_TEST_TMP) { $env:FOLO_TEST_TMP } else { [System.IO.Path]::GetTempPath() }
if (-not $Out) { $Out = Join-Path $tmp "interact-check" }
if ($Clipboard -and -not $Sandbox -and -not $InSandbox) { "NOTE  -Clipboard needs -Sandbox: no clipboard steps" }
if ($Sandbox -and -not $InSandbox) {
    $self = $MyInvocation.MyCommand.Path
    $only = if ($Only.Count) { " -Only " + ($Only -join ",") } else { "" }
    # The clipboard is saved and put back here, outside the sandbox, so a sandboxed run that is killed
    # (a timeout, Ctrl+C here, which closes its job) still gets it put back. The sandboxed steps write a
    # ledger of their own changes; [FoloClip]::Decide reads it.
    $clip = ""
    if ($Clipboard -and ($Only.Count -eq 0 -or $Only -contains "clipboard")) {
        . (Join-Path $PSScriptRoot "clip-save.ps1")
        if (-not (Test-Path $tmp)) { New-Item -ItemType Directory -Force $tmp | Out-Null }
        $ledger = Join-Path $tmp "interact-check-clip-$PID.txt"
        Remove-Item -ErrorAction SilentlyContinue $ledger
        $why = ""
        $seq0 = [FoloClip]::GetClipboardSequenceNumber()
        $list0 = [FoloClip]::List()
        $saved = [FoloClip]::Save([ref]$why)
        if ($null -eq $saved) { "SKIP  clipboard steps: your clipboard holds a format that could not be put back exactly ($why)" }
        else { $clip = " -Clipboard -ClipLedger '$ledger' -ClipSeq $seq0" }
    }
    $code = 1
    try {
        & (Join-Path $PSScriptRoot "sandbox-run.ps1") -TimeoutS 900 -Command "& '$self' -Exe '$Exe' -Out '$Out' -InSandbox$only$clip"
        $code = $LASTEXITCODE
    } finally {
        if ($clip) {
            $lines = if (Test-Path $ledger) { @(Get-Content -Encoding ascii $ledger) } else { @() }
            $now = [FoloClip]::GetClipboardSequenceNumber()
            $verdict = [FoloClip]::Decide($seq0, [string[]]$lines, $now, [FoloClip]::Text())
            $did = switch ($verdict) {
                "unchanged" { "untouched" }
                "restore" { [FoloClip]::Restore($saved, $now) }
                default { "changed by someone else meanwhile: left as it is" }
            }
            $same = [FoloClip]::Same($saved) -and ([FoloClip]::List() -eq $list0)
            $line = "your clipboard is back byte for byte  $did; formats before '$list0', after '$([FoloClip]::List())'"
            if ($same) { "PASS  $line" } else { "FAIL  $line"; $code = 1 }
            Remove-Item -ErrorAction SilentlyContinue $ledger
        }
    }
    exit $code
}
. (Join-Path $PSScriptRoot "drive.ps1")
Add-Type -AssemblyName System.Windows.Forms
if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force "$Out\shots", "$Out\fail" | Out-Null

$script:fails = 0
$script:passes = 0
function Pass([string]$name, [bool]$ok, [string]$detail = "") {
    if ($ok) { $script:passes++; "PASS  $name  $detail" } else { $script:fails++; "FAIL  $name  $detail" }
}
function Wants([string]$group) { $Only.Count -eq 0 -or $Only -contains $group }

# ---- the app
$script:p = $null
$script:h = [IntPtr]::Zero
$script:s = 1.5
$script:launches = 0
function Launch([string[]]$appArgs) {
    Stop-Folo $script:p
    $envs = @{ FOLO_DESKTOP_STATE = "$Out\desktop.json" }
    if (-not $InSandbox) { $envs.FOLO_UI_OFFSCREEN = "1" }
    $script:launches++
    $script:mask = $null
    $script:p = Start-Folo $Exe $appArgs $envs "$Out\app-$($script:launches).stderr.txt"
    $script:h = Wait-FoloWindow $script:p 25
    if ($script:h -eq [IntPtr]::Zero) { throw "no window for $($appArgs -join ' ')" }
    $script:s = [FoloDrive]::GetDpiForWindow($script:h) / 96.0
    Settle 1500
}
function Settle([int]$ms = 350) { Start-Sleep -Milliseconds $ms }
function Alive { -not $script:p.HasExited }
# Logical px to client device px.
function D([double]$v) { [int][Math]::Round($v * $script:s) }
function Point([double]$x, [double]$y) { [FoloDrive]::Move($script:h, (D $x), (D $y), 0) }
function Click([double]$x, [double]$y) { [FoloDrive]::Click($script:h, (D $x), (D $y)); Settle }
function Leave { [FoloDrive]::Leave($script:h); Settle }
# keys: Key 0x4B "ctrl"
function Key([int]$vk, [string]$mods = "") {
    $m = 0
    if ($mods -match "ctrl") { $m += 1 }
    if ($mods -match "shift") { $m += 2 }
    if ($mods -match "alt") { $m += 4 }
    [FoloDrive]::Key($script:h, [uint16]$vk, $m); Settle 150
}
function Keys([string]$text) { [FoloDrive]::Type($script:h, $text); Settle 200 }
function Wheel([double]$x, [double]$y, [int]$notches) { [FoloDrive]::Wheel($script:h, (D $x), (D $y), 120 * $notches); Settle }
function Cap { [FoloDrive]::Capture($script:h) }
# Pixels that animate by themselves (the breathing status dot, a progress shimmer) are learned from two
# captures of a still state and left out of every comparison; a launch or a resize starts over.
$script:mask = $null
function Learn {
    $a = Cap; Settle 450; $b = Cap
    $m = [FoloDrive]::LiveMask($a, $b, 2)
    $a.Dispose(); $b.Dispose()
    $script:mask = if ($script:mask -and $script:mask.Length -eq $m.Length) { [FoloDrive]::Or($script:mask, $m) } else { $m }
}
function Snap { Cap }
# Whether two captures show the same thing (no pixel off by more than 8, live pixels skipped), in a
# logical rectangle @(x, y, w, h) or everywhere.
function Same($a, $b, $rect = $null) {
    $r = if ($rect) { @((D $rect[0]), (D $rect[1]), (D $rect[2]), (D $rect[3])) } else { @(0, 0, 0, 0) }
    $d = [FoloDrive]::DiffMasked($a, $b, 8, $script:mask, $r[0], $r[1], $r[2], $r[3])
    $script:why = "$($d[1]) px over 8 (max $($d[2])) in ($($d[3]),$($d[4]))-($($d[5]),$($d[6]))"
    $script:lastB = $b
    $d[0] -eq 1 -and $d[1] -eq 0
}
# What the last comparison found (for a FAIL line), saving its second capture.
function Why([string]$label) {
    if ($script:lastB) { [FoloDrive]::Save($script:lastB, (Join-Path "$Out\fail" "$label.png")) }
    "$($script:why); capture in fail\$label.png"
}
# Parks the pointer where nothing reacts to it, the title bar's empty drag band (as the real pointer:
# WM_NCMOUSEMOVE), outside every scroll area (a thumb shows while the pointer is over its area), then
# lets it leave the window.
function Away {
    $c = [FoloDrive]::Client($script:h)
    [FoloDrive]::HitTest($script:h, $c.Left + (D 480), $c.Top + (D 20)) | Out-Null
    [FoloDrive]::Leave($script:h); Settle 1100
}

# The headless shot of a scenario (cached).
function Shot([string]$scenario, [int]$w = 1280, [int]$hh = 800) {
    $file = Join-Path "$Out\shots" ("{0}-{1}x{2}.png" -f ($scenario -replace '[+]', '_'), $w, $hh)
    if (-not (Test-Path $file)) {
        & $Exe --scenario $scenario --width $w --height $hh --scale 1.5 --shot $file 2>&1 | Out-Null
        if ($LASTEXITCODE) { throw "--shot $scenario failed ($LASTEXITCODE)" }
    }
    $file
}
# Whether the window now shows what the headless shot of $scenario shows.
function MatchStep([string]$name, [string]$scenario, [string]$label, [int]$w = 1280, [int]$hh = 800) {
    Learn
    $want = [System.Drawing.Bitmap]::FromFile((Shot $scenario $w $hh))
    $got = Cap
    try {
        $d = [FoloDrive]::DiffMasked($want, $got, 8, $script:mask, 0, 0, 0, 0)
        $ok = $d[0] -eq 1 -and $d[1] -eq 0
        if (-not $ok) { [FoloDrive]::Save($got, (Join-Path "$Out\fail" "$label.png")) }
        $live = if ($script:mask) { [FoloDrive]::Count($script:mask) } else { 0 }
        $detail = if ($d[0] -eq 0) { "size $($got.Width)x$($got.Height) vs $($want.Width)x$($want.Height)" }
                  elseif ($ok) { "max diff $($d[2]), $live live px skipped" }
                  else { "$($d[1]) px over 8 (max $($d[2])) in ($($d[3]),$($d[4]))-($($d[5]),$($d[6])), $live live px skipped; capture in fail\$label.png" }
        Pass $name $ok "= --shot $scenario ${w}x${hh}: $detail"
    } finally { $want.Dispose(); $got.Dispose() }
}
function Guard {
    $r = [FoloDrive]::Rect($script:h)
    if (-not $InSandbox -and -not [FoloDrive]::OffEveryMonitor($r)) { Stop-Folo $script:p; throw "the window is on a display: $([FoloDrive]::Str($r))" }
    if (-not $InSandbox -and [FoloDrive]::GetForegroundWindow() -eq $script:h) { Stop-Folo $script:p; throw "the window took the foreground" }
}

# Sidebar rows (logical y) at 1280x800.
$nav = [ordered]@{ overview = 62.7; datasets = 94.7; runs = 126.7; models = 158.7; benchmarks = 190.7; playground = 222.7 }
# The palette's panel (logical) at 1280x800.
$panel = @(320, 88, 640, 490)

try {
    "mode: $(if ($InSandbox) { 'private desktop' } else { 'off screen' }), exe $Exe"

    # ---------- 1. the window shows what --shot renders
    if (Wants "captures") {
        $cases = @(
            @("overview", 1280, 800), @("overview+light", 1280, 800), @("palette", 1280, 800), @("datasets", 1280, 800),
            @("dataset", 1280, 800), @("inspector", 1280, 800), @("run-live", 1280, 800), @("run-done", 1280, 800),
            @("new-run", 1280, 800), @("models", 1280, 800), @("playground", 1280, 800), @("settings+cloud", 1280, 800),
            @("bench-entry", 1280, 800), @("toast", 1280, 800), @("kb-focus", 1280, 800),
            @("overview", 960, 620), @("run-live", 960, 620), @("new-run", 960, 620), @("overview", 1680, 1050), @("model", 1680, 1050))
        foreach ($c in $cases) {
            Launch @("--scenario", $c[0], "--width", $c[1], "--height", $c[2])
            Settle 700
            Guard
            MatchStep "capture equals the headless shot" $c[0] "cap-$($c[0])-$($c[1])" $c[1] $c[2]
        }
    }

    # ---------- 2. overview: hover, tooltip, navigation, palette, keys, wheel, theme, resize
    if (Wants "overview") {
        Launch @("--scenario", "overview", "--width", "1280", "--height", "800")
        Learn
        $before = Snap
        if ($InSandbox) {
            # Off screen the backend lets only a posted WM_MOUSELEAVE end a hover; on a desktop the
            # real cursor is not on (this one), TrackMouseEvent answers a posted move with an
            # immediate WM_MOUSELEAVE, so hover cannot be held there. The off-screen run covers it.
            "SKIP  hover, button hover and tooltip steps: no cursor on a private desktop (see the off-screen run)"
        } else {
        Point 66 94.7; Settle
        Pass "hover a sidebar item: the hover style appears" (-not (Same $before (Snap))) ""
        Away
        Pass "the mouse leaves: it clears" (Same $before (Snap)) ""
        Point 1209 73; Settle
        Pass "hover a button (New run): the hover style appears" (-not (Same $before (Snap))) ""
        Point 120 560; Settle
        Pass "moving off the button clears it" (Same $before (Snap)) ""
        Away

        # Tooltip: the Runs count badge has one; it shows after the 500 ms delay, goes when the mouse leaves.
        Point 205 127; Settle 150
        $early = Snap
        Settle 800
        $late = Snap
        Pass "tooltip: not before its delay, shown after it" (-not (Same $early $late)) ""
        Away
        Pass "tooltip: hidden when the mouse leaves its element" (Same $before (Snap)) ""
        }

        # Every sidebar section, by click: the page is that scenario's shot.
        foreach ($k in @("datasets", "runs", "models", "benchmarks")) {
            Click 66 $nav[$k]; Away
            MatchStep "click sidebar $k" $k "nav-$k"
        }
        # The playground scenario has a model and an image picked; by click it is the empty page.
        $pre = Snap
        Click 66 $nav.playground; Away
        Pass "click sidebar playground: the page changes" (-not (Same $pre (Snap))) ""
        Click 66 744.7; Away
        MatchStep "click sidebar settings" "settings" "nav-settings"
        Click 66 $nav.overview; Away
        MatchStep "click sidebar overview" "overview" "nav-overview"

        # The palette: Ctrl+K opens it (the palette scenario), Esc closes it.
        Key 0x4B "ctrl"; Settle 500
        MatchStep "Ctrl+K opens the palette" "palette" "palette-open"
        # Its field: characters, backspace, arrows, selection, Ctrl+A. Each sequence must show what
        # typing the result directly shows (the panel: field and filtered results).
        function PanelAfter([scriptblock]$keys) {
            Key 0x41 "ctrl"; Key 0x2E   # select all, Delete: an empty field
            & $keys
            Settle 300
            Snap
        }
        $empty = PanelAfter {}
        $runs = PanelAfter { Keys "runs" }
        Pass "palette field: typing filters the list" (-not (Same $empty $runs $panel)) ""
        Pass "palette field: backspace" (Same $runs (PanelAfter { Keys "runsx"; Key 0x08 }) $panel) "runsx + Backspace = runs"
        Pass "palette field: arrows insert in place" (Same $runs (PanelAfter { Keys "rns"; Key 0x25; Key 0x25; Keys "u"; Key 0x23 }) $panel) "rns, Left, Left, u, End = runs"
        Pass "palette field: shift+arrows select, typing replaces" (Same $runs (PanelAfter { Keys "rxyz"; Key 0x25 "shift"; Key 0x25 "shift"; Key 0x25 "shift"; Keys "uns" }) $panel) "rxyz, Shift+Left x3, uns = runs"
        Pass "palette field: Ctrl+A selects all, typing replaces" (Same $runs (PanelAfter { Keys "models"; Key 0x41 "ctrl"; Keys "runs" }) $panel) "models, Ctrl+A, runs = runs"
        Pass "palette field: Home and Delete" (Same $runs (PanelAfter { Keys "xruns"; Key 0x24; Key 0x2E; Key 0x23 }) $panel) "xruns, Home, Delete, End = runs"
        # Arrows move the highlighted result.
        $first = PanelAfter {}
        Key 0x28; Settle 200
        $second = Snap
        Key 0x26; Settle 200
        Pass "palette: Down / Up move the highlight" ((-not (Same $first $second $panel)) -and (Same $first (Snap) $panel)) ""
        # IME: a composition (WM_IME_STARTCOMPOSITION, the composed characters, WM_IME_ENDCOMPOSITION)
        # goes into the field like typed text.
        $null = PanelAfter {}
        [FoloDrive]::Send($script:h, 0x10D, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null
        foreach ($ch in "runs".ToCharArray()) { [FoloDrive]::Post($script:h, 0x286, [IntPtr][int]$ch, [IntPtr]1) }
        [FoloDrive]::Sync($script:h)
        [FoloDrive]::Send($script:h, 0x10E, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null
        Settle 300
        Pass "IME: composed characters reach the field" (Same $runs (Snap) $panel) ""
        Key 0x1B; Away
        MatchStep "Esc closes the palette" "overview" "palette-closed"

        # Tab / Shift+Tab focus rings, F6 regions.
        $s0 = Snap
        Key 0x09; Settle 300; $t1 = Snap
        Key 0x09; Settle 300; $t2 = Snap
        Key 0x09 "shift"; Settle 300; $t3 = Snap
        Pass "Tab shows a focus ring and moves it" ((-not (Same $s0 $t1)) -and (-not (Same $t1 $t2))) ""
        Pass "Shift+Tab moves it back" (Same $t1 $t3) ""
        Key 0x75; Settle 300; $f1 = Snap
        Key 0x75; Settle 300; $f2 = Snap
        Key 0x75 "shift"; Settle 300; $f3 = Snap
        Pass "F6 moves between regions, Shift+F6 back" ((-not (Same $t3 $f1)) -and (-not (Same $f1 $f2)) -and (Same $f1 $f3)) ""
        Key 0x1B; Click 120 560; Away
        MatchStep "a click leaves keyboard mode" "overview" "kb-left"

        # Wheel: the long overview page scrolls down and back.
        $top = Snap
        Wheel 900 500 -3
        $down = Snap
        Wheel 900 500 3
        Away
        Pass "wheel scrolls the page down and back" ((-not (Same $top $down)) -and (Same $top (Snap))) ""

        # Theme: Ctrl+Shift+L is the light theme's shot, and back.
        Key 0x4C "ctrl,shift"; Settle 500
        MatchStep "Ctrl+Shift+L switches to the light theme" "overview+light" "theme-light"
        Key 0x4C "ctrl,shift"; Settle 500
        MatchStep "and back to dark" "overview" "theme-dark"

        # Resize: below the minimum is clamped to 960x620, a large size lays out as that size's shot.
        $r = [FoloDrive]::Rect($script:h)
        $c = New-Object FoloDrive+RECT; [FoloDrive]::GetClientRect($script:h, [ref]$c) | Out-Null
        $fw = ($r.Right - $r.Left) - $c.Right; $fh = ($r.Bottom - $r.Top) - $c.Bottom
        [FoloDrive]::MoveWindow($script:h, $r.Left, $r.Top, 400, 300); Settle 800
        Guard
        MatchStep "resize below the minimum: 960x620" "overview" "resize-min" 960 620
        [FoloDrive]::MoveWindow($script:h, $r.Left, $r.Top, 2520 + $fw, 1575 + $fh); Settle 800
        Guard
        MatchStep "resize to 1680x1050" "overview" "resize-large" 1680 1050
        [FoloDrive]::MoveWindow($script:h, $r.Left, $r.Top, $r.Right - $r.Left, $r.Bottom - $r.Top); Settle 800
        MatchStep "and back to 1280x800" "overview" "resize-back"
        Pass "scale stays 1.5" ([FoloDrive]::GetDpiForWindow($script:h) -eq 144) "dpi $([FoloDrive]::GetDpiForWindow($script:h))"
    }

    # ---------- 3. toast: Generate synthetic on the demo explains itself in a toast, which goes away
    # (the live demo: a scenario's frozen clock keeps toasts up). Compared where the toast shows.
    if (Wants "toast") {
        Launch @("--mock", "--open", "datasets", "--width", "1280", "--height", "800")
        Learn
        $corner = @(860, 640, 420, 160)
        $plain = Snap
        Click 1049 73; Away
        Pass "a toast appears" (-not (Same $plain (Snap) $corner)) ""
        Settle 4600
        $ok = Same $plain (Snap) $corner
        Pass "and goes after its time" $ok $(if (-not $ok) { Why "toast-gone" })
    }

    # ---------- 4. a long grid: the dataset page scrolls by wheel
    if (Wants "grid") {
        Launch @("--scenario", "dataset", "--width", "1280", "--height", "800")
        Learn
        $top = Snap
        Wheel 760 600 -4
        $down = Snap
        Wheel 760 600 4
        Away
        Pass "wheel scrolls the dataset grid and back" ((-not (Same $top $down)) -and (Same $top (Snap))) ""
        Guard
    }

    # ---------- 5. playground: the confidence slider drags; the model select picks with the keyboard
    if (Wants "playground") {
        Launch @("--scenario", "playground", "--width", "1280", "--height", "800")
        Learn
        $value = @(1195, 330, 50, 26)   # the "0.35" label
        $before = Snap
        [FoloDrive]::Drag($script:h, (D 1104.7), (D 343.3), (D 50), 0, 10); Away
        $dragged = Snap
        [FoloDrive]::Drag($script:h, (D 1154.7), (D 343.3), (D -50), 0, 10); Away
        Pass "drag the slider: the value follows, and back" ((-not (Same $before $dragged $value)) -and (Same $before (Snap) $value)) ""
        $sel = @(953, 173, 287, 32)
        $model = Snap
        Click 1096 189; Settle 400
        $open = Snap
        Key 0x28; Key 0x0D; Settle 400; Away
        $picked = Snap
        Pass "select: opens, Down + Enter picks another model and closes it" ((-not (Same $model $open)) -and (-not (Same $model $picked $sel)) -and (-not (Same $open $picked))) ""
        Guard
    }

    # ---------- 6. the New run dialog: Ctrl+N opens it, fill the name, a select with keys, Esc closes
    if (Wants "dialog") {
        Launch @("--scenario", "runs", "--width", "1280", "--height", "800")
        Learn
        $runs = Snap
        $brand = @(8, 6, 200, 28)
        Key 0x4E "ctrl"; Settle 600
        $open = Snap
        Pass "Ctrl+N opens the New run dialog (its scrim dims the sidebar)" (-not (Same $runs $open $brand)) ""
        $name = @(300, 138, 334, 34)
        Keys "my-run"; Settle 200
        Pass "fill: the name field takes typing" (-not (Same $open (Snap) $name)) ""
        $ds = @(648, 138, 334, 34)
        $d0 = Snap
        Click 814 154; Settle 400
        Key 0x28; Key 0x0D; Settle 400; Away
        Pass "the Dataset select picks with the keyboard" (-not (Same $d0 (Snap) $ds)) ""
        $epochs = @(548, 648, 84, 28)
        $e0 = Snap
        # A press on the track sets the value there. (Dragging a slider inside the dialog does not move
        # it, on GPUI too: the root view tracks slider drags with on_mouse_move, which the dialog's
        # occluding scrim hides. The playground step above drags one on a page.)
        Click 506.7 662; Away
        $moved = -not (Same $e0 (Snap) $epochs)
        Pass "a click on the Epochs track sets it" $moved $(if (-not $moved) { Why "epochs" })
        Key 0x1B; Away
        Pass "Esc closes it" (Same $runs (Snap)) ""
        Guard
    }

    # ---------- 7. clipboard (-Sandbox -Clipboard only): copy from and paste into a text field
    # The outer run saved the clipboard and puts it back after this run ends, however it ends (see the
    # header). Each change these steps make is written to the ledger before it is made ("pending <text>")
    # and once it is confirmed ("owned <sequence number>"), so the outer run knows which change is the
    # test's own; a change made by anyone else stops the steps and is never overwritten.
    if ((Wants "clipboard") -and $InSandbox -and $Clipboard) {
        if (-not $ClipLedger) { throw "-Clipboard in the sandbox needs -ClipLedger (the outer run passes it)" }
        . (Join-Path $PSScriptRoot "clip-save.ps1")
        $script:clipSeq = $ClipSeq
        # Makes one clipboard change: refuses if the clipboard is not as the steps left it, records the
        # change as pending, makes it, waits for the clipboard to hold $text and records it as the test's.
        # Returns whether it got there; if not, the change is left unconfirmed (the outer run puts the
        # saved data back only if the clipboard holds exactly $text).
        function ClipChange([string]$text, [scriptblock]$change) {
            if ([FoloClip]::GetClipboardSequenceNumber() -ne $script:clipSeq) { throw "someone else changed the clipboard: stopping the clipboard steps" }
            Add-Content -Encoding ascii $ClipLedger "pending $text"
            & $change
            for ($i = 0; $i -lt 50; $i++) {
                $s1 = [FoloClip]::GetClipboardSequenceNumber()
                $t = [FoloClip]::Text()
                if ($s1 -ne $script:clipSeq -and $t -eq $text -and [FoloClip]::GetClipboardSequenceNumber() -eq $s1) {
                    $script:clipSeq = $s1
                    Add-Content -Encoding ascii $ClipLedger "owned $s1"
                    return $true
                }
                Start-Sleep -Milliseconds 20
            }
            $false
        }
        Launch @("--scenario", "overview", "--width", "1280", "--height", "800")
        Learn
        Key 0x4B "ctrl"; Settle 400
        Key 0x41 "ctrl"; Key 0x2E; Keys "runs"; Settle 300
        $runsPanel = Snap
        $copied = ClipChange "runs" { Key 0x41 "ctrl"; Key 0x43 "ctrl" }
        Pass "Ctrl+C copies the field's selection" $copied "clipboard '$([FoloClip]::Text())'"
        if (-not (ClipChange "models" { [System.Windows.Forms.Clipboard]::SetText("models") })) { throw "could not put 'models' on the clipboard" }
        Key 0x41 "ctrl"; Key 0x2E; Keys "runs"; Key 0x41 "ctrl"; Key 0x56 "ctrl"; Settle 300
        $pasted = Snap
        Key 0x41 "ctrl"; Key 0x2E; Keys "models"; Settle 300
        Pass "Ctrl+V pastes over the selection" ((Same $pasted (Snap) $panel) -and (-not (Same $pasted $runsPanel $panel))) ""
    }

    # ---------- 8. the file dialog (private desktop only: it is a real, visible window)
    if ((Wants "filedialog") -and $InSandbox) {
        Launch @("--scenario", "overview", "--width", "1280", "--height", "800")
        Learn
        $plain = Snap
        Key 0x4F "ctrl"
        $dlg = [IntPtr]::Zero
        for ($i = 0; $i -lt 100 -and $dlg -eq [IntPtr]::Zero; $i++) {
            Start-Sleep -Milliseconds 50
            $line = [FoloDrive]::List([uint32]$script:p.Id) | Where-Object { $_ -match "^\d+ #32770 " } | Select-Object -First 1
            if ($line) { $dlg = [IntPtr][long]($line.Split(" ")[0]) }
        }
        Pass "Ctrl+O shows the open dialog" ($dlg -ne [IntPtr]::Zero) "$line"
        if ($dlg -ne [IntPtr]::Zero) {
            # The app's loop keeps running under the dialog's: the live demo's dot keeps animating.
            $a = Cap; Settle 600; $b = Cap
            $live = [FoloDrive]::Count([FoloDrive]::LiveMask($a, $b, 0))
            Pass "frames keep coming while it is up" ($live -gt 0) "$live px changed in 600 ms"
            [FoloDrive]::Post($dlg, [FoloDrive]::WM_CLOSE, [IntPtr]::Zero, [IntPtr]::Zero)
            $gone = $false
            for ($i = 0; $i -lt 60 -and -not $gone; $i++) { Start-Sleep -Milliseconds 50; $gone = -not [FoloDrive]::IsWindow($dlg) }
            Settle 400
            Pass "Cancel closes it; the app is as before" ($gone -and (Alive) -and (Same $plain (Snap))) ""
            Key 0x4B "ctrl"; Settle 500
            Pass "and still takes keys (Ctrl+K opens the palette)" (-not (Same $plain (Snap))) ""
        }
    }

    # ---------- close: WM_CLOSE ends the app cleanly
    if (Wants "close") {
        Launch @("--mock", "--width", "1280", "--height", "800")
        Settle 1000
        Guard
        Pass "the live demo stays alive" (Alive) ""
        [FoloDrive]::Post($script:h, [FoloDrive]::WM_CLOSE, [IntPtr]::Zero, [IntPtr]::Zero)
        Pass "WM_CLOSE exits it cleanly" ((Wait-FoloExit $script:p 10) -and $script:p.ExitCode -eq 0) "exit code $($script:p.ExitCode)"
    }
}
catch {
    Pass "script ran to the end" $false "$_ $($_.InvocationInfo.PositionMessage)"
}
finally {
    Stop-Folo $script:p
    $panics = @(Get-ChildItem "$Out\app-*.stderr.txt" -ErrorAction SilentlyContinue | Select-String -Pattern "panicked")
    Pass "no panic in $($script:launches) launches" ($panics.Count -eq 0) (($panics | ForEach-Object { $_.Line }) -join " | ")
}
"$($script:passes) passed, $($script:fails) failed"
if ($script:fails) { exit 1 } else { exit 0 }
