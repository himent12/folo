# Tests clip-save.ps1's logic that touches no clipboard: which formats a save keeps (Plan), the memory
# blocks a restore hands over (Blocks, Bytes, Free), and when a restore may put the saved data back over
# the clipboard (Decide). Never opens, reads or writes the clipboard. Exit code 1 if a check fails.
#
#   tools\clip-save-test.ps1
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "clip-save.ps1")

$script:fails = 0
function Check([string]$name, $got, $want) {
    if ("$got" -eq "$want") { "PASS  $name" } else { $script:fails++; "FAIL  ${name}: got '$got', want '$want'" }
}
function Plan([uint32[]]$all) {
    $why = ""
    $l = New-Object 'System.Collections.Generic.List[uint32]'
    foreach ($f in $all) { $l.Add($f) }
    $p = [FoloClip]::Plan($l, [ref]$why)
    if ($null -eq $p) { "refused: $why" } else { ($p | ForEach-Object { $_ }) -join " " }
}

# ---- Plan: the formats a save keeps, in the clipboard's order
Check "text: CF_TEXT and CF_OEMTEXT come from CF_UNICODETEXT" (Plan 13, 16, 1, 7) "13 16"
Check "a registered format and text" (Plan 49687, 13, 16, 1, 7) "49687 13 16"
Check "text put as CF_TEXT: Unicode kept, ANSI synthesized" (Plan 1, 16, 13, 7) "16 13"
Check "CF_TEXT alone: CF_OEMTEXT synthesized" (Plan 1, 7) "1"
Check "a DIB: the bitmap and the other DIB are synthesized" (Plan 8, 17, 2) "17"
Check "a bitmap with its DIBs: the DIBV5 is kept" (Plan 2, 8, 17) "17"
Check "a bitmap alone is a GDI handle" (Plan 2) "refused: format 2 is a GDI handle"
Check "an enhanced metafile is a GDI handle" (Plan 15, 14, 3) "refused: format 14 is a GDI handle"
Check "a palette is a GDI handle" (Plan 8, 9) "refused: format 9 is a GDI handle"
Check "owner display is a handle" (Plan 13, 0x80) "refused: format 128 is a GDI handle"
Check "a GDI object format is a handle" (Plan 0x300) "refused: format 768 is a GDI handle"
Check "a private format is kept (Save refuses it if it is not global memory)" (Plan 0x200) "512"
Check "files (CF_HDROP) are memory" (Plan 15, 49158) "15 49158"
Check "an empty clipboard: nothing to keep" (Plan) ""

# ---- Blocks: one movable block per format with its bytes; Free releases what is still ours
$saved = New-Object 'System.Collections.Generic.List[System.Collections.Generic.KeyValuePair[uint32,byte[]]]'
$big = New-Object byte[] 1000003
(New-Object System.Random 7).NextBytes($big)
foreach ($kv in @(@(13, [byte[]](0x72, 0, 0x75, 0, 0, 0)), @(49687, $big), @(16, [byte[]]@()))) {
    $saved.Add((New-Object 'System.Collections.Generic.KeyValuePair[uint32,byte[]]' ([uint32]$kv[0]), ([byte[]]$kv[1])))
}
$why = ""
$blocks = [FoloClip]::Blocks($saved, [ref]$why)
Check "every block is made" ($null -ne $blocks -and $blocks.Length -eq 3 -and @($blocks | Where-Object { $_ -eq [IntPtr]::Zero }).Count -eq 0) "True"
for ($i = 0; $i -lt $saved.Count; $i++) {
    $want = $saved[$i].Value
    $got = [FoloClip]::Bytes($blocks[$i])
    $n = [Math]::Max(1, $want.Length)
    $prefix = if ($want.Length) { [System.Linq.Enumerable]::SequenceEqual([byte[]]$got[0..($want.Length - 1)], $want) } else { $true }
    Check "block $i holds format $($saved[$i].Key)'s $($want.Length) bytes" ($got.Length -ge $n -and $prefix) "True"
}
$blocks[1] = [IntPtr]::Zero   # as if handed to the clipboard: not ours to free
$freed = [IntPtr[]]$blocks.Clone()
[FoloClip]::Free($blocks)
Check "Free releases the blocks still ours" (@($blocks | Where-Object { $_ -ne [IntPtr]::Zero }).Count) 0
Check "a null handle is not memory" ($null -eq [FoloClip]::Bytes([IntPtr]::Zero)) "True"

# ---- Decide: put the saved data back only over the steps' own change
$seq0 = [uint32]100
function Decide([string[]]$ledger, [uint32]$now, $text) { [FoloClip]::Decide($seq0, [string[]]$ledger, $now, $text) }
Check "nothing changed" (Decide @() 100 "x") "unchanged"
Check "nothing changed, although a change was pending (the run stopped before it)" (Decide @("pending runs") 100 "x") "unchanged"
Check "the copy happened and was confirmed" (Decide @("pending runs", "owned 103") 103 "runs") "restore"
Check "the second change was confirmed" (Decide @("pending runs", "owned 103", "pending models", "owned 105") 105 "models") "restore"
Check "the copy happened, the run stopped before confirming it" (Decide @("pending runs") 102 "runs") "restore"
Check "the second change happened, not confirmed" (Decide @("pending runs", "owned 103", "pending models") 106 "models") "restore"
Check "the second change did not happen: the confirmed one is there" (Decide @("pending runs", "owned 103", "pending models") 103 "runs") "restore"
Check "someone copied after the confirmed change (the same text: a confirmed change is not pending)" (Decide @("pending runs", "owned 103") 104 "runs") "leave"
Check "someone copied after a pending change" (Decide @("pending runs", "owned 103", "pending models") 107 "theirs") "leave"
Check "someone copied before the steps changed anything" (Decide @() 101 "theirs") "leave"
Check "a pending change whose text is not there" (Decide @("pending runs") 102 "theirs") "leave"
Check "a pending change, no text on the clipboard" (Decide @("pending runs") 102 $null) "leave"
Check "an unreadable ledger line is ignored" (Decide @("owned x", "garbage") 103 "runs") "leave"

if ($script:fails) { "$($script:fails) failed"; exit 1 } else { "all passed"; exit 0 }
