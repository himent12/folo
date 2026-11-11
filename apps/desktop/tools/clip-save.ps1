# Saves the clipboard byte for byte and puts it back, for interact-check.ps1's copy and paste steps (the
# clipboard belongs to the window station, which a private desktop shares with the person at the PC).
# Dot-source it. [FoloClip]::Save() returns null when some format cannot be restored exactly, so the
# steps must not run: a GDI handle (CF_BITMAP with no DIB, metafiles, palettes, owner display, GDI object
# formats) is not plain memory. Every other format is a global memory block whose bytes are copied.
# Formats Windows synthesizes are left to it (CF_TEXT and CF_OEMTEXT from CF_UNICODETEXT and CF_LOCALE,
# CF_BITMAP and the other DIB from a DIB), so the restored clipboard lists the same formats with the same
# bytes. What cannot be kept: the clipboard's owner (a program that renders on request has rendered) and
# its sequence number. [FoloClip]::Restore puts the saved formats back only if nobody else changed the
# clipboard since the sequence number it is given, and touches nothing unless it has an owner window and
# every block of memory ready; [FoloClip]::Decide says whether a change is the test's own (from the ledger
# the steps keep); [FoloClip]::Same compares the clipboard with a save. Plan, Blocks and Decide touch no
# clipboard: clip-save-test.ps1 tests them.
if (-not ("FoloClip" -as [type])) {
    Add-Type @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
public static class FoloClip {
    [DllImport("user32.dll", SetLastError = true)] static extern bool OpenClipboard(IntPtr h);
    [DllImport("user32.dll")] static extern bool CloseClipboard();
    [DllImport("user32.dll", SetLastError = true)] static extern bool EmptyClipboard();
    [DllImport("user32.dll")] static extern uint EnumClipboardFormats(uint f);
    [DllImport("user32.dll")] static extern IntPtr GetClipboardData(uint f);
    [DllImport("user32.dll")] static extern IntPtr SetClipboardData(uint f, IntPtr h);
    [DllImport("user32.dll")] public static extern uint GetClipboardSequenceNumber();
    // The wide API with wide strings (the default CharSet is Ansi, which hands it a garbled class name).
    [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)] static extern IntPtr CreateWindowExW(int ex, string cls, string name, int style, int x, int y, int w, int h, IntPtr parent, IntPtr menu, IntPtr inst, IntPtr param);
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
    // The formats to save and put back, from every format the clipboard lists (in its order): the ones
    // Windows synthesizes left out. Null when one cannot be restored from bytes (`why` says which).
    public static List<uint> Plan(IList<uint> all, out string why) {
        why = "";
        var keep = new List<uint>();
        foreach (var f in all) {
            if ((f == 1 || f == 7) && all.Contains(13)) continue;
            if (f == 7 && all.Contains(1)) continue;
            if (f == 2 && (all.Contains(8) || all.Contains(17))) continue;
            if (f == 8 && all.Contains(17)) continue;
            if (IsHandle(f)) { why = "format " + f + " is a GDI handle"; return null; }
            keep.Add(f);
        }
        return keep;
    }
    // A global memory block's bytes, or null when it is not one.
    public static byte[] Bytes(IntPtr h) {
        ulong n = GlobalSize(h).ToUInt64();
        if (n > int.MaxValue) return null;
        IntPtr p = GlobalLock(h);
        if (p == IntPtr.Zero) return null;
        try { var b = new byte[n]; Marshal.Copy(p, b, 0, (int)n); return b; } finally { GlobalUnlock(h); }
    }
    // One movable global memory block per saved format, holding its bytes, ready for SetClipboardData.
    // Null (with nothing left allocated) when one cannot be made.
    public static IntPtr[] Blocks(List<KeyValuePair<uint, byte[]>> saved, out string why) {
        why = "";
        var hs = new IntPtr[saved.Count];
        for (int i = 0; i < saved.Count; i++) {
            byte[] b = saved[i].Value;
            IntPtr h = GlobalAlloc(0x0002, new UIntPtr((ulong)Math.Max(1, b.Length)));   // GMEM_MOVEABLE
            if (h == IntPtr.Zero) { Free(hs); why = "no memory for format " + saved[i].Key; return null; }
            hs[i] = h;
            IntPtr p = GlobalLock(h);
            if (p == IntPtr.Zero) { Free(hs); why = "could not lock the block for format " + saved[i].Key; return null; }
            Marshal.Copy(b, 0, p, b.Length);
            GlobalUnlock(h);
        }
        return hs;
    }
    // Frees the blocks still ours (an entry handed to the clipboard is zero: the system owns it).
    public static void Free(IntPtr[] hs) {
        for (int i = 0; i < hs.Length; i++) { if (hs[i] != IntPtr.Zero) { GlobalFree(hs[i]); hs[i] = IntPtr.Zero; } }
    }

    // Every format's bytes (keys in the clipboard's order), or null when one cannot be restored.
    // Throws when the clipboard cannot be opened.
    public static List<KeyValuePair<uint, byte[]>> Save(out string why) {
        if (!Open(IntPtr.Zero)) throw new Exception("the clipboard stayed open in another program");
        try {
            var plan = Plan(Formats(), out why);
            if (plan == null) return null;
            var saved = new List<KeyValuePair<uint, byte[]>>();
            foreach (var f in plan) {
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

    // The clipboard's text (CF_UNICODETEXT, up to its terminator), or null when it has none or stays busy.
    public static string Text() {
        if (!Open(IntPtr.Zero)) return null;
        try {
            IntPtr h = GetClipboardData(13);
            if (h == IntPtr.Zero) return null;
            var b = Bytes(h);
            if (b == null) return null;
            string s = System.Text.Encoding.Unicode.GetString(b, 0, b.Length & ~1);
            int end = s.IndexOf('\0');
            return end < 0 ? s : s.Substring(0, end);
        } finally { CloseClipboard(); }
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

    // What to do with the clipboard after the steps, from the sequence number it had when it was saved,
    // the ledger the steps wrote as they went ("pending <text>" before each change they make, "owned
    // <sequence number>" once that change is confirmed), and the clipboard now. "unchanged": it still
    // holds what was saved. "restore": it holds the steps' last change (the confirmed one, or a pending
    // one that happened but was not confirmed because the run stopped: the clipboard holds exactly the
    // text it was putting there). "leave": someone else changed it, and theirs stays.
    public static string Decide(uint seq0, IList<string> ledger, uint nowSeq, string nowText) {
        if (nowSeq == seq0) return "unchanged";
        bool hasOwned = false; uint owned = 0; string pending = null;
        foreach (var line in ledger) {
            if (line.StartsWith("owned ")) { hasOwned = uint.TryParse(line.Substring(6), out owned); pending = null; }
            else if (line.StartsWith("pending ")) pending = line.Substring(8);
        }
        if (hasOwned && nowSeq == owned) return "restore";
        if (pending != null && nowText == pending) return "restore";
        return "leave";
    }

    // Puts `saved` back, unless the clipboard changed since sequence number `expected` (someone else
    // copied: theirs stays). Everything that can fail is done before the clipboard is emptied (the memory
    // blocks, the owner window, opening it), and a failure there leaves the clipboard as it is. Returns
    // what it did.
    public static string Restore(List<KeyValuePair<uint, byte[]>> saved, uint expected) {
        if (GetClipboardSequenceNumber() != expected) return "changed by someone else meanwhile: left as it is";
        string why;
        IntPtr[] blocks = Blocks(saved, out why);
        if (blocks == null) return why + ": left as it is";
        try {
            // SetClipboardData needs the clipboard opened by a window (the owner EmptyClipboard sets);
            // with no owner it is documented to fail, after the clear. The data outlives the window.
            IntPtr owner = CreateWindowExW(0, "STATIC", "folo-clip", 0, 0, 0, 0, 0, new IntPtr(-3), IntPtr.Zero, IntPtr.Zero, IntPtr.Zero);
            if (owner == IntPtr.Zero) return "no owner window (error " + Marshal.GetLastWin32Error() + "): left as it is";
            try {
                if (!Open(owner)) return "the clipboard stayed open in another program: left as it is";
                try {
                    if (GetClipboardSequenceNumber() != expected) return "changed by someone else meanwhile: left as it is";
                    if (!EmptyClipboard()) return "could not empty it (error " + Marshal.GetLastWin32Error() + "): left as it is";
                    var refused = new List<string>();
                    for (int i = 0; i < saved.Count; i++) {
                        if (SetClipboardData(saved[i].Key, blocks[i]) != IntPtr.Zero) blocks[i] = IntPtr.Zero;
                        else refused.Add(saved[i].Key.ToString());
                    }
                    return refused.Count == 0 ? "restored" : "format " + string.Join(", ", refused) + " refused";
                } finally { CloseClipboard(); }
            } finally { DestroyWindow(owner); }
        } finally { Free(blocks); }
    }
}
"@
}
