# Draws FOLO's logo (design/logo.svg: four corner brackets and the accent dot) on a dark rounded tile
# and writes assets/folo.ico, the Windows app icon that build.rs links into folo-desktop.exe (the
# taskbar, Alt-Tab, Explorer and the window's own icon). Colours come from design/tokens.json (dark
# theme: bg.panel, line.strong, text.primary, accent.solid). Run it again after the logo or those
# tokens change, and commit the .ico:
#
#   tools\make-icon.ps1
param([string]$Out = (Join-Path $PSScriptRoot "..\assets\folo.ico"))
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

$tokens = Get-Content -Raw (Join-Path $PSScriptRoot "..\..\..\design\tokens.json") | ConvertFrom-Json
$dark = $tokens.color.dark
function Color([string]$hex) {
    $h = $hex.TrimStart("#")
    $a = if ($h.Length -eq 8) { [Convert]::ToInt32($h.Substring(6, 2), 16) } else { 255 }
    [System.Drawing.Color]::FromArgb($a, [Convert]::ToInt32($h.Substring(0, 2), 16),
        [Convert]::ToInt32($h.Substring(2, 2), 16), [Convert]::ToInt32($h.Substring(4, 2), 16))
}
$tile = Color $dark.bg.panel
$edge = Color $dark.line.strong
$ink = Color $dark.text.primary
$dot = Color $dark.accent.solid

function Draw([int]$n) {
    $bmp = New-Object System.Drawing.Bitmap $n, $n, ([System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
    $g.Clear([System.Drawing.Color]::Transparent)

    # The tile: a rounded square, its radius about a fifth of the side, with a hairline edge.
    $r = [Math]::Max(2.0, $n * 0.2)
    $box = New-Object System.Drawing.Drawing2D.GraphicsPath
    $w = $n - 1.0
    $box.AddArc(0.5, 0.5, 2 * $r, 2 * $r, 180, 90)
    $box.AddArc($w - 2 * $r + 0.5, 0.5, 2 * $r, 2 * $r, 270, 90)
    $box.AddArc($w - 2 * $r + 0.5, $w - 2 * $r + 0.5, 2 * $r, 2 * $r, 0, 90)
    $box.AddArc(0.5, $w - 2 * $r + 0.5, 2 * $r, 2 * $r, 90, 90)
    $box.CloseFigure()
    $g.FillPath((New-Object System.Drawing.SolidBrush $tile), $box)
    if ($n -ge 32) { $g.DrawPath((New-Object System.Drawing.Pen $edge, 1.0), $box) }

    # The logo's 24-unit box, inset so the brackets clear the tile's corners.
    $s = $n * 0.78 / 24.0
    $o = ($n - 24 * $s) / 2.0
    $g.TranslateTransform([float]$o, [float]$o)
    $g.ScaleTransform([float]$s, [float]$s)
    $pen = New-Object System.Drawing.Pen $ink, ([float][Math]::Max(2.0, 1.4 / $s))
    $pen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
    $pen.EndCap = [System.Drawing.Drawing2D.LineCap]::Round
    $pen.LineJoin = [System.Drawing.Drawing2D.LineJoin]::Round
    $p = New-Object System.Drawing.Drawing2D.GraphicsPath
    # M3 8.5V5a2 2 0 0 1 2-2h3.5, and the same at the other three corners.
    $p.StartFigure(); $p.AddLine(3, 8.5, 3, 5); $p.AddArc(3, 3, 4, 4, 180, 90); $p.AddLine(5, 3, 8.5, 3)
    $p.StartFigure(); $p.AddLine(15.5, 3, 19, 3); $p.AddArc(17, 3, 4, 4, 270, 90); $p.AddLine(21, 5, 21, 8.5)
    $p.StartFigure(); $p.AddLine(21, 15.5, 21, 19); $p.AddArc(17, 17, 4, 4, 0, 90); $p.AddLine(19, 21, 15.5, 21)
    $p.StartFigure(); $p.AddLine(8.5, 21, 5, 21); $p.AddArc(3, 17, 4, 4, 90, 90); $p.AddLine(3, 19, 3, 15.5)
    $g.DrawPath($pen, $p)
    $g.FillEllipse((New-Object System.Drawing.SolidBrush $dot), 8.75, 8.75, 6.5, 6.5)
    $g.Dispose()
    $ms = New-Object System.IO.MemoryStream
    $bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    , $ms.ToArray()
}

# An .ico of PNG images (Windows Vista and later read those at every size).
$sizes = @(16, 20, 24, 32, 40, 48, 64, 256)
$images = @($sizes | ForEach-Object { , (Draw $_) })
$fs = [System.IO.File]::Create($Out)
$bw = New-Object System.IO.BinaryWriter $fs
$bw.Write([uint16]0); $bw.Write([uint16]1); $bw.Write([uint16]$sizes.Count)
$offset = 6 + 16 * $sizes.Count
for ($i = 0; $i -lt $sizes.Count; $i++) {
    $d = if ($sizes[$i] -ge 256) { 0 } else { $sizes[$i] }
    $bw.Write([byte]$d); $bw.Write([byte]$d); $bw.Write([byte]0); $bw.Write([byte]0)
    $bw.Write([uint16]1); $bw.Write([uint16]32)
    $bw.Write([uint32]$images[$i].Length); $bw.Write([uint32]$offset)
    $offset += $images[$i].Length
}
foreach ($img in $images) { $bw.Write($img) }
$bw.Close()
"wrote $Out ($offset bytes, $($sizes -join ', ') px)"
