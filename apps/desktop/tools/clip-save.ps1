# Saves the clipboard byte for byte and puts it back, for interact-check.ps1's copy and paste steps (the
# clipboard belongs to the window station, which a private desktop shares with the person at the PC).
# Dot-source it. [FoloClip]::Save() returns null when some format cannot be restored exactly, so the
# steps must not run: a GDI handle (CF_BITMAP with no DIB, metafiles, palettes, owner display, GDI object
# formats) is not plain memory. Every other format is a global memory block whose bytes are copied.
# Formats Windows synthesizes are left to it (CF_TEXT and CF_OEMTEXT from CF_UNICODETEXT and CF_LOCALE,
# CF_BITMAP and the other DIB from a DIB), so the restored clipboard lists the same formats with the same
# bytes. What cannot be kept: the clipboard's owner (a program that renders on request has rendered) and
# its sequence number. [FoloClip]::Restore puts the saved formats back only if nobody else changed the
# clipboard since the sequence number it is given; [FoloClip]::Same compares the clipboard with a save.
if (-not ("FoloClip" -as [type])) {
    Add-Type @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
public static class FoloClip {
    [DllImport("user32.dll")] static extern bool OpenClipboard(IntPtr h);
    [DllImport("user32.dll")] static extern bool CloseClipboard();
    [DllImport("user32.dll")] static extern bool EmptyClipboard();
    [DllImport("user32.dll")] static extern uint EnumClipboardFormats(uint f);
    [DllImport("user32.dll")] static extern IntPtr GetClipboardData(uint f);
    [DllImport("user32.dll")] static extern IntPtr SetClipboardData(uint f, IntPtr h);
    [DllImport("user32.dll")] public static extern uint GetClipboardSequenceNumber();
    [DllImport("user32.dll")] static extern IntPtr CreateWindowExW(int ex, string cls, string name, int style, int x, int y, int w, int h, IntPtr parent, IntPtr menu, IntPtr inst, IntPtr param);
    [DllImport("user32.dll")] static extern bool DestroyWindow(IntPtr h);
    [DllImport("kernel32.dll")] static extern IntPtr GlobalLock(IntPtr h);
    [DllImport("kernel32.dll")] static extern bool GlobalUnlock(IntPtr h);
    [DllImport("kernel32.dll")] static extern UIntPtr GlobalSize(IntPtr h);
    [DllImport("kernel32.dll")] static extern IntPtr GlobalAlloc(uint flags, UIntPtr size);
    [DllImport("kernel32.dll")] static extern IntPtr GlobalFree(IntPtr h);

    // Handles, not memory: not restorable from bytes.
    static bool IsHandle(uint f) {
        return f == 2 || f == 3 || f == 9 || f == 14 || f == 0x80 || f == 0x82 || f == 0x83 || f == 0x8E
            || (f >= 0x300 && f <= 0x3FF);
    }
    // Opening can fail while another program holds the clipboard for a moment: retry briefly.
    static bool Open(IntPtr h) {
        for (int i = 0; i < 50; i++) { if (OpenClipboard(h)) return true; Thread.Sleep(20); }
        return false;
    }
    static List<uint> Formats() {
        var l = new List<uint>();
        uint f = 0;
        while ((f = EnumClipboardFormats(f)) != 0) l.Add(f);
        return l;
    }
    // The formats to restore, in the clipboard's order: synthesized ones left out.
    static List<uint> Sources(List<uint> all) {
        var keep = new List<uint>();
        foreach (var f in all) {
            if ((f == 1 || f == 7) && all.Contains(13)) continue;
            if (f == 7 && all.Contains(1)) continue;
            if (f == 2 && (all.Contains(8) || all.Contains(17))) continue;
            if (f == 8 && all.Contains(17)) continue;
            keep.Add(f);
        }
        return keep;
    }
    static byte[] Bytes(IntPtr h) {
        int n = (int)GlobalSize(h).ToUInt64();
        IntPtr p = GlobalLock(h);
        if (p == IntPtr.Zero) return null;
        try { var b = new byte[n]; Marshal.Copy(p, b, 0, n); return b; } finally { GlobalUnlock(h); }
    }

    // Every format's bytes (keys in the clipboard's order), or null when one cannot be restored.
    // Throws when the clipboard cannot be opened.
    public static List<KeyValuePair<uint, byte[]>> Save(out string why) {
        why = "";
        if (!Open(IntPtr.Zero)) throw new Exception("the clipboard stayed open in another program");
        try {
            var all = Formats();
            var saved = new List<KeyValuePair<uint, byte[]>>();
            foreach (var f in Sources(all)) {
                if (IsHandle(f)) { why = "format " + f + " is a GDI handle"; return null; }
                IntPtr h = GetClipboardData(f);
                if (h == IntPtr.Zero) { why = "format " + f + " gave no data"; return null; }
                var b = Bytes(h);
                if (b == null) { why = "format " + f + " is not global memory"; return null; }
                saved.Add(new KeyValuePair<uint, byte[]>(f, b));
            }
            return saved;
        } finally { CloseClipboard(); }
    }

    // Every format the clipboard lists now, synthesized ones included ("13 49687 16 1 7").
    public static string List() {
        if (!Open(IntPtr.Zero)) return "?";
        try { return string.Join(" ", Formats()); } finally { CloseClipboard(); }
    }

    // Whether the clipboard holds exactly `saved` (the same restorable formats, the same bytes).
    public static bool Same(List<KeyValuePair<uint, byte[]>> saved) {
        string why;
        var now = Save(out why);
        if (now == null || now.Count != saved.Count) return false;
        for (int i = 0; i < now.Count; i++) {
            if (now[i].Key != saved[i].Key) return false;
            var a = now[i].Value; var b = saved[i].Value;
            if (a.Length != b.Length) return false;
            for (int j = 0; j < a.Length; j++) if (a[j] != b[j]) return false;
        }
        return true;
    }

    // Puts `saved` back, unless the clipboard changed since sequence number `expected` (someone else
    // copied: theirs stays). Returns what it did.
    public static string Restore(List<KeyValuePair<uint, byte[]>> saved, uint expected) {
        if (GetClipboardSequenceNumber() != expected) return "changed by someone else meanwhile: left as it is";
        // EmptyClipboard needs an owner window for SetClipboardData to work; the data outlives it.
        IntPtr owner = CreateWindowExW(0, "STATIC", "folo-clip", 0, 0, 0, 0, 0, new IntPtr(-3), IntPtr.Zero, IntPtr.Zero, IntPtr.Zero);
        try {
            if (!Open(owner)) return "could not open the clipboard";
            try {
                if (GetClipboardSequenceNumber() != expected) return "changed by someone else meanwhile: left as it is";
                EmptyClipboard();
                foreach (var kv in saved) {
                    IntPtr h = GlobalAlloc(0x0002, new UIntPtr((ulong)Math.Max(1, kv.Value.Length)));   // GMEM_MOVEABLE
                    IntPtr p = GlobalLock(h);
                    Marshal.Copy(kv.Value, 0, p, kv.Value.Length);
                    GlobalUnlock(h);
                    if (SetClipboardData(kv.Key, h) == IntPtr.Zero) { GlobalFree(h); return "format " + kv.Key + " was refused"; }
                }
                return "restored";
            } finally { CloseClipboard(); }
        } finally { DestroyWindow(owner); }
    }
}
"@
}
