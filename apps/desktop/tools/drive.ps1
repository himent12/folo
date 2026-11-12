# Drives a FOLO Desktop window with window messages only, for the window checks (dot-source it):
# interact-check.ps1, chrome-check.ps1 and window-check.ps1. Nothing here moves the cursor, activates a
# window or injects real input, so it works on an off-screen window (FOLO_UI_OFFSCREEN=1) next to someone
# using the PC, and on a private desktop (sandbox-run.ps1), where real input cannot reach.
#
# - Mouse: posted WM_MOUSEMOVE / WM_*BUTTON* / WM_MOUSEWHEEL / non-client messages, as the real mouse
#   would produce them (client or screen coordinates in device pixels).
# - Keys: the app reads modifiers with GetKeyState, which posted messages do not change. [FoloDrive]::Key
#   attaches this thread's input state to the app's UI thread (AttachThreadInput), sets the modifiers
#   with SetKeyboardState (that thread state only, never the system's), sends WM_KEYDOWN / WM_KEYUP, and
#   detaches. The app turns an unhandled key into WM_CHAR itself (TranslateMessage), as with a real key.
# - Sync: two WM_NULL sends in a row return only after everything posted before them was dispatched.
# - Captures: PrintWindow of the client area (PW_CLIENTONLY | PW_RENDERFULLCONTENT; on a desktop the
#   compositor does not show, PW_CLIENTONLY alone, which the app answers with WM_PRINTCLIENT).
if (-not ("FoloDrive" -as [type])) {
    Add-Type -ReferencedAssemblies System.Drawing @"
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
public static class FoloDrive {
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
    [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
    public delegate bool EnumProc(IntPtr h, IntPtr l);
    public delegate bool MonEnum(IntPtr h, IntPtr hdc, ref RECT r, IntPtr d);
    [DllImport("user32.dll")] public static extern bool SetProcessDpiAwarenessContext(IntPtr v);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc f, IntPtr l);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll")] public static extern bool IsWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool IsZoomed(IntPtr h);
    [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
    [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern IntPtr GetActiveWindow();
    [DllImport("user32.dll")] public static extern uint GetDpiForWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern int GetWindowLong(IntPtr h, int i);
    [DllImport("user32.dll")] public static extern IntPtr GetClassLongPtrW(IntPtr h, int i);
    [DllImport("user32.dll")] public static extern IntPtr SendMessageW(IntPtr h, uint m, IntPtr w, IntPtr l);
    [DllImport("user32.dll")] public static extern bool PostMessageW(IntPtr h, uint m, IntPtr w, IntPtr l);
    [DllImport("user32.dll", SetLastError = true)] public static extern IntPtr SendMessageTimeoutW(IntPtr h, uint m, IntPtr w, IntPtr l, uint flags, uint ms, out IntPtr result);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint flags);
    [DllImport("user32.dll")] public static extern bool EnumDisplayMonitors(IntPtr hdc, IntPtr clip, MonEnum f, IntPtr d);
    [DllImport("user32.dll")] public static extern IntPtr MonitorFromWindow(IntPtr h, uint flags);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowTextW(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetClassNameW(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint a, uint b, bool attach);
    [DllImport("user32.dll")] public static extern bool SetKeyboardState(byte[] s);
    [DllImport("user32.dll")] public static extern uint MapVirtualKeyW(uint code, uint type);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern short VkKeyScanW(char c);
    [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();

    public const uint WM_NULL = 0, WM_CLOSE = 0x10, WM_KEYDOWN = 0x100, WM_KEYUP = 0x101, WM_CHAR = 0x102,
        WM_SYSKEYDOWN = 0x104, WM_SYSKEYUP = 0x105, WM_MOUSEMOVE = 0x200, WM_LBUTTONDOWN = 0x201,
        WM_LBUTTONUP = 0x202, WM_RBUTTONDOWN = 0x204, WM_RBUTTONUP = 0x205, WM_MOUSEWHEEL = 0x20A,
        WM_MOUSELEAVE = 0x2A3, WM_NCHITTEST = 0x84, WM_NCMOUSEMOVE = 0xA0, WM_NCLBUTTONDOWN = 0xA1,
        WM_NCLBUTTONUP = 0xA2, WM_NCLBUTTONDBLCLK = 0xA3, WM_SYSCOMMAND = 0x112, WM_DROPFILES = 0x233;

    public static IntPtr Pt(int x, int y) { return (IntPtr)(((y & 0xFFFF) << 16) | (x & 0xFFFF)); }

    // The app's window: the process's visible top-level FOLO::Window with the largest client area
    // (IntPtr.Zero if none). By class, so a dialog or any other window of the process is never taken
    // for it (a 1200 x 720 file dialog once was, once the app's window was smaller).
    public static IntPtr Find(uint pid) {
        IntPtr best = IntPtr.Zero; long bestArea = -1;
        EnumWindows((h, l) => {
            uint p; GetWindowThreadProcessId(h, out p);
            if (p == pid && IsWindowVisible(h) && ClassOf(h) == "FOLO::Window") {
                RECT r; GetClientRect(h, out r);
                long a = (long)(r.Right - r.Left) * (r.Bottom - r.Top);
                if (a > bestArea) { bestArea = a; best = h; }
            }
            return true;
        }, IntPtr.Zero);
        return best;
    }
    // Every visible top-level window of a process: "hwnd class 'title' rect".
    public static List<string> List(uint pid) {
        var l = new List<string>();
        EnumWindows((h, x) => {
            uint p; GetWindowThreadProcessId(h, out p);
            if (p == pid && IsWindowVisible(h)) {
                var c = new StringBuilder(256); GetClassNameW(h, c, 256);
                var t = new StringBuilder(256); GetWindowTextW(h, t, 256);
                RECT r; GetWindowRect(h, out r);
                l.Add(h + " " + c + " '" + t + "' " + Str(r));
            }
            return true;
        }, IntPtr.Zero);
        return l;
    }
    public static string ClassOf(IntPtr h) { var c = new StringBuilder(256); GetClassNameW(h, c, 256); return c.ToString(); }
    // Every visible top-level window of a process (the app's, a dialog, anything) that a display shows a
    // part of, as List prints them: empty when nothing of the process can be seen. The guards check
    // this, never only the app's window.
    public static List<string> OnDisplay(uint pid) {
        var l = new List<string>();
        foreach (var line in List(pid)) {
            IntPtr h = (IntPtr)long.Parse(line.Split(' ')[0]);
            RECT r; GetWindowRect(h, out r);
            if (r.Right > r.Left && r.Bottom > r.Top && !OffEveryMonitor(r)) l.Add(line);
        }
        return l;
    }
    public static List<RECT> Monitors() {
        var l = new List<RECT>();
        EnumDisplayMonitors(IntPtr.Zero, IntPtr.Zero, (IntPtr h, IntPtr hdc, ref RECT r, IntPtr d) => { l.Add(r); return true; }, IntPtr.Zero);
        return l;
    }
    public static bool OffEveryMonitor(RECT w) {
        foreach (var m in Monitors())
            if (w.Left < m.Right && w.Right > m.Left && w.Top < m.Bottom && w.Bottom > m.Top) return false;
        return true;
    }
    public static string Str(RECT r) { return "(" + r.Left + "," + r.Top + ")-(" + r.Right + "," + r.Bottom + ")"; }
    public static RECT Rect(IntPtr h) { RECT r; GetWindowRect(h, out r); return r; }
    public static RECT Client(IntPtr h) {
        RECT r; GetClientRect(h, out r);
        POINT p = new POINT(); ClientToScreen(h, ref p);
        r.Left += p.X; r.Right += p.X; r.Top += p.Y; r.Bottom += p.Y;
        return r;
    }

    public static IntPtr Send(IntPtr h, uint msg, IntPtr w, IntPtr l) {
        IntPtr res;
        SendMessageTimeoutW(h, msg, w, l, 2 /* SMTO_ABORTIFHUNG */, 5000, out res);
        return res;
    }
    public static void Post(IntPtr h, uint msg, IntPtr w, IntPtr l) { PostMessageW(h, msg, w, l); }
    // Returns once everything posted to the window before the call has been dispatched.
    public static void Sync(IntPtr h) { Send(h, WM_NULL, IntPtr.Zero, IntPtr.Zero); Send(h, WM_NULL, IntPtr.Zero, IntPtr.Zero); }

    // ---- mouse, in client device pixels (non-client: screen device pixels)
    public static void Move(IntPtr h, int x, int y, int mk) { Post(h, WM_MOUSEMOVE, (IntPtr)mk, Pt(x, y)); Sync(h); }
    public static void Click(IntPtr h, int x, int y) {
        Move(h, x - 2, y, 0); Move(h, x, y, 0);
        Post(h, WM_LBUTTONDOWN, (IntPtr)1, Pt(x, y)); Sync(h);
        Post(h, WM_LBUTTONUP, IntPtr.Zero, Pt(x, y)); Sync(h);
    }
    public static void Drag(IntPtr h, int x, int y, int dx, int dy, int steps) {
        Move(h, x, y, 0);
        Post(h, WM_LBUTTONDOWN, (IntPtr)1, Pt(x, y)); Sync(h);
        for (int i = 1; i <= steps; i++) Move(h, x + dx * i / steps, y + dy * i / steps, 1);
        Post(h, WM_LBUTTONUP, IntPtr.Zero, Pt(x + dx, y + dy)); Sync(h);
    }
    // One wheel notch is 120; positive scrolls up. The position is in screen pixels.
    public static void Wheel(IntPtr h, int x, int y, int delta) {
        POINT p = new POINT(); p.X = x; p.Y = y; ClientToScreen(h, ref p);
        Post(h, WM_MOUSEWHEEL, (IntPtr)((delta & 0xFFFF) << 16), Pt(p.X, p.Y)); Sync(h);
    }
    [StructLayout(LayoutKind.Sequential)] struct GUITHREADINFO { public int cbSize; public uint flags; public IntPtr hwndActive, hwndFocus, hwndCapture, hwndMenuOwner, hwndMoveSize, hwndCaret; public RECT rcCaret; }
    [DllImport("user32.dll")] static extern bool GetGUIThreadInfo(uint tid, ref GUITHREADINFO g);
    // Whether the window's thread is in Windows' modal move / size loop (GUI_INMOVESIZE): a caption or
    // edge press the app left to Windows starts it. Posted messages cannot end it (it follows the real
    // mouse): EndMoveSize does, with WM_CANCELMODE.
    public static bool InMoveSize(IntPtr h) {
        uint pid; uint t = GetWindowThreadProcessId(h, out pid);
        var g = new GUITHREADINFO(); g.cbSize = Marshal.SizeOf(g);
        return GetGUIThreadInfo(t, ref g) && (g.flags & 2) != 0;
    }
    public static void EndMoveSize(IntPtr h) { Send(h, 0x1F /* WM_CANCELMODE */, IntPtr.Zero, IntPtr.Zero); }
    [DllImport("user32.dll")] static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int cx, int cy, uint flags);
    // Moves / resizes the window as Windows' own move and size loop would (no activation, no z-order).
    public static void MoveWindow(IntPtr h, int x, int y, int w, int hh) { SetWindowPos(h, IntPtr.Zero, x, y, w, hh, 0x0004 | 0x0010); }
    public static void Leave(IntPtr h) { Post(h, WM_MOUSELEAVE, IntPtr.Zero, IntPtr.Zero); Sync(h); }
    public static int HitTest(IntPtr h, int sx, int sy) {
        Post(h, WM_NCMOUSEMOVE, (IntPtr)1, Pt(sx, sy)); Sync(h);
        return (int)Send(h, WM_NCHITTEST, IntPtr.Zero, Pt(sx, sy));
    }
    // A non-client press and release (or a double click: the second press is WM_NCLBUTTONDBLCLK). On a
    // caption (HTCAPTION) or a resize edge the press goes to Windows' move / size loop, which takes the
    // release itself, so the window never gets WM_NCLBUTTONUP there; here nothing is posted for it, and
    // a loop the press started is ended (EndMoveSize).
    public static void NcClick(IntPtr h, int sx, int sy, int code, bool dbl) {
        HitTest(h, sx, sy);
        Post(h, WM_NCLBUTTONDOWN, (IntPtr)code, Pt(sx, sy)); Sync(h);
        bool loopArea = code == 2 || (code >= 10 && code <= 17);
        if (InMoveSize(h)) EndMoveSize(h);
        if (!loopArea) { Post(h, WM_NCLBUTTONUP, (IntPtr)code, Pt(sx, sy)); Sync(h); }
        if (dbl) {
            Post(h, WM_NCLBUTTONDBLCLK, (IntPtr)code, Pt(sx, sy)); Sync(h);
            Post(h, WM_NCLBUTTONUP, (IntPtr)code, Pt(sx, sy)); Sync(h);
        }
    }

    // ---- keys. mods: 1 ctrl, 2 shift, 4 alt.
    static IntPtr KeyL(ushort vk, bool up, bool alt) {
        long scan = MapVirtualKeyW(vk, 0);
        long l = 1 | (scan << 16);
        if (alt) l |= 1L << 29;
        if (up) l |= (1L << 30) | (1L << 31);
        return (IntPtr)l;
    }
    static void SetMods(int mods) {
        var s = new byte[256];
        if ((mods & 1) != 0) { s[0x11] = 0x80; s[0xA2] = 0x80; }
        if ((mods & 2) != 0) { s[0x10] = 0x80; s[0xA0] = 0x80; }
        if ((mods & 4) != 0) { s[0x12] = 0x80; s[0xA4] = 0x80; }
        SetKeyboardState(s);
    }
    public static void Key(IntPtr h, ushort vk, int mods) {
        uint pid; uint tid = GetWindowThreadProcessId(h, out pid);
        uint me = GetCurrentThreadId();
        bool attached = AttachThreadInput(me, tid, true);
        try {
            SetMods(mods);
            bool alt = (mods & 4) != 0;
            Send(h, alt ? WM_SYSKEYDOWN : WM_KEYDOWN, (IntPtr)vk, KeyL(vk, false, alt));
            Sync(h);   // the WM_CHAR the app's TranslateMessage posted, under the same key state
            Send(h, alt ? WM_SYSKEYUP : WM_KEYUP, (IntPtr)vk, KeyL(vk, true, alt));
            Sync(h);
        } finally {
            SetMods(0);
            if (attached) AttachThreadInput(me, tid, false);
        }
    }
    // Types text as key presses (letters, digits, space and what VkKeyScan maps on this layout).
    public static void Type(IntPtr h, string text) {
        foreach (char c in text) {
            short k = VkKeyScanW(c);
            if (k == -1) { Post(h, WM_CHAR, (IntPtr)c, (IntPtr)1); Sync(h); continue; }
            Key(h, (ushort)(k & 0xFF), ((k >> 8) & 1) != 0 ? 2 : 0);
        }
    }

    // ---- captures
    [StructLayout(LayoutKind.Sequential)] struct BIH { public int biSize, biWidth, biHeight; public short biPlanes, biBitCount; public int biCompression, biSizeImage, x, y, c, i; }
    [DllImport("gdi32.dll")] static extern IntPtr CreateCompatibleDC(IntPtr h);
    [DllImport("gdi32.dll")] static extern bool DeleteDC(IntPtr h);
    [DllImport("gdi32.dll")] static extern IntPtr CreateDIBSection(IntPtr dc, ref BIH bi, uint usage, out IntPtr bits, IntPtr sec, uint off);
    [DllImport("gdi32.dll")] static extern IntPtr SelectObject(IntPtr dc, IntPtr o);
    [DllImport("gdi32.dll")] static extern bool DeleteObject(IntPtr o);
    [DllImport("gdi32.dll")] static extern bool GdiFlush();
    // PrintWindow into a plain GDI DIB section, copied into a Bitmap. Not through GDI+'s
    // Graphics.GetHdc: that pre-fills its DC with RGB(13, 11, 12) and on ReleaseHdc keeps the old
    // bitmap wherever a pixel still has that colour, so every drawn pixel of exactly RGB(13, 11, 12)
    // came back black.
    static Bitmap Print(IntPtr h, int w, int hh, uint flags) {
        var bi = new BIH(); bi.biSize = 40; bi.biWidth = w; bi.biHeight = -hh; bi.biPlanes = 1; bi.biBitCount = 32;
        IntPtr dc = CreateCompatibleDC(IntPtr.Zero), bits;
        IntPtr dib = CreateDIBSection(dc, ref bi, 0, out bits, IntPtr.Zero, 0);
        IntPtr old = SelectObject(dc, dib);
        PrintWindow(h, dc, flags);
        GdiFlush();
        var bmp = new Bitmap(w, hh, PixelFormat.Format32bppRgb);
        var d = bmp.LockBits(new Rectangle(0, 0, w, hh), ImageLockMode.WriteOnly, PixelFormat.Format32bppRgb);
        var row = new byte[w * 4];
        for (int y = 0; y < hh; y++) {
            Marshal.Copy(bits + y * w * 4, row, 0, row.Length);
            Marshal.Copy(row, 0, d.Scan0 + y * d.Stride, row.Length);
        }
        bmp.UnlockBits(d);
        SelectObject(dc, old); DeleteObject(dib); DeleteDC(dc);
        return bmp;
    }
    public static Bitmap Capture(IntPtr h) {
        RECT r; GetClientRect(h, out r);
        int w = Math.Max(1, r.Right - r.Left), hh = Math.Max(1, r.Bottom - r.Top);
        foreach (uint flags in new uint[] { 3, 1 }) {
            var bmp = Print(h, w, hh, flags);
            if (!Blank(bmp)) return bmp;
            bmp.Dispose();
        }
        return new Bitmap(w, hh, PixelFormat.Format32bppRgb);
    }
    // Saves a capture as a PNG, alpha dropped.
    public static void Save(Bitmap b, string path) { b.Save(path, ImageFormat.Png); }
    static bool Blank(Bitmap b) {
        for (int y = 4; y < b.Height; y += Math.Max(1, b.Height / 16))
            for (int x = 4; x < b.Width; x += Math.Max(1, b.Width / 16))
                if ((b.GetPixel(x, y).ToArgb() & 0xFFFFFF) != 0) return false;
        return true;
    }
    static int[] Pixels(Bitmap b) {
        var d = b.LockBits(new Rectangle(0, 0, b.Width, b.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppRgb);
        var px = new int[b.Width * b.Height];
        for (int y = 0; y < b.Height; y++) Marshal.Copy(d.Scan0 + y * d.Stride, px, y * b.Width, b.Width);
        b.UnlockBits(d);
        return px;
    }
    // "same size?", pixels over tol, largest channel difference, and the box around the differences:
    // { sizeOk, over, max, x0, y0, x1, y1 }. Regions in `ignore` (x, y, w, h quadruples) are skipped.
    public static int[] Diff(Bitmap a, Bitmap b, int tol, int[] ignore) {
        if (a.Width != b.Width || a.Height != b.Height) return new int[] { 0, -1, -1, 0, 0, 0, 0 };
        int[] pa = Pixels(a), pb = Pixels(b);
        int over = 0, max = 0, x0 = int.MaxValue, y0 = int.MaxValue, x1 = -1, y1 = -1, w = a.Width;
        for (int i = 0; i < pa.Length; i++) {
            if (pa[i] == pb[i]) continue;
            int x = i % w, y = i / w;
            bool skip = false;
            if (ignore != null)
                for (int k = 0; k + 3 < ignore.Length; k += 4)
                    if (x >= ignore[k] && y >= ignore[k + 1] && x < ignore[k] + ignore[k + 2] && y < ignore[k + 1] + ignore[k + 3]) { skip = true; break; }
            if (skip) continue;
            int d = 0;
            for (int s = 0; s < 24; s += 8) d = Math.Max(d, Math.Abs(((pa[i] >> s) & 255) - ((pb[i] >> s) & 255)));
            max = Math.Max(max, d);
            if (d > tol) { over++; x0 = Math.Min(x0, x); y0 = Math.Min(y0, y); x1 = Math.Max(x1, x); y1 = Math.Max(y1, y); }
        }
        return new int[] { 1, over, max, x0, y0, x1, y1 };
    }
    // Pixels that differ between two captures of the same still state (what animates by itself, such as
    // the breathing status dot), grown by `grow` pixels: comparisons skip them.
    public static bool[] LiveMask(Bitmap a, Bitmap b, int grow) {
        int w = a.Width, h = a.Height;
        var mask = new bool[w * h];
        if (b.Width != w || b.Height != h) return mask;
        int[] pa = Pixels(a), pb = Pixels(b);
        for (int i = 0; i < pa.Length; i++) {
            if (((pa[i] ^ pb[i]) & 0xFFFFFF) == 0) continue;
            int x = i % w, y = i / w;
            for (int j = Math.Max(0, y - grow); j <= Math.Min(h - 1, y + grow); j++)
                for (int k = Math.Max(0, x - grow); k <= Math.Min(w - 1, x + grow); k++) mask[j * w + k] = true;
        }
        return mask;
    }
    public static bool[] Or(bool[] a, bool[] b) {
        var r = new bool[a.Length];
        for (int i = 0; i < a.Length; i++) r[i] = a[i] || (i < b.Length && b[i]);
        return r;
    }
    public static int Count(bool[] m) { int n = 0; foreach (bool v in m) if (v) n++; return n; }
    // Like Diff, inside the rectangle (x, y, rw, rh) only (rw <= 0: everywhere) and skipping masked pixels.
    public static int[] DiffMasked(Bitmap a, Bitmap b, int tol, bool[] mask, int rx, int ry, int rw, int rh) {
        if (a.Width != b.Width || a.Height != b.Height) return new int[] { 0, -1, -1, 0, 0, 0, 0 };
        int[] pa = Pixels(a), pb = Pixels(b);
        int w = a.Width, h = a.Height;
        if (rw <= 0) { rx = 0; ry = 0; rw = w; rh = h; }
        int over = 0, max = 0, x0 = int.MaxValue, y0 = int.MaxValue, x1 = -1, y1 = -1;
        for (int y = Math.Max(0, ry); y < Math.Min(h, ry + rh); y++)
            for (int x = Math.Max(0, rx); x < Math.Min(w, rx + rw); x++) {
                int i = y * w + x;
                if (pa[i] == pb[i] || (mask != null && mask.Length == pa.Length && mask[i])) continue;
                int d = 0;
                for (int s = 0; s < 24; s += 8) d = Math.Max(d, Math.Abs(((pa[i] >> s) & 255) - ((pb[i] >> s) & 255)));
                max = Math.Max(max, d);
                if (d > tol) { over++; x0 = Math.Min(x0, x); y0 = Math.Min(y0, y); x1 = Math.Max(x1, x); y1 = Math.Max(y1, y); }
            }
        return new int[] { 1, over, max, x0, y0, x1, y1 };
    }
    // A hash of a rectangle of a capture (device pixels).
    public static long Hash(Bitmap b, int x, int y, int w, int h) {
        x = Math.Max(0, x); y = Math.Max(0, y);
        w = Math.Min(w, b.Width - x); h = Math.Min(h, b.Height - y);
        if (w <= 0 || h <= 0) return 0;
        var d = b.LockBits(new Rectangle(x, y, w, h), ImageLockMode.ReadOnly, PixelFormat.Format32bppRgb);
        var row = new int[w];
        long hash = 17;
        for (int j = 0; j < h; j++) {
            Marshal.Copy(d.Scan0 + j * d.Stride, row, 0, w);
            for (int i = 0; i < w; i++) hash = hash * 31 + (row[i] & 0xFFFFFF);
        }
        b.UnlockBits(d);
        return hash;
    }
    public static int Pixel(Bitmap b, int x, int y) { return b.GetPixel(x, y).ToArgb() & 0xFFFFFF; }
}
"@
}
# Per-monitor DPI aware v2: rectangles, messages and captures in device pixels.
[FoloDrive]::SetProcessDpiAwarenessContext([IntPtr]-4) | Out-Null

# Starts an exe with extra environment variables and stdout / stderr in files; returns the Process.
function Start-Folo([string]$Exe, [string[]]$ArgList, [hashtable]$Env, [string]$ErrFile) {
    $old = @{}
    foreach ($k in $Env.Keys) { $old[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $Env[$k]) }
    try {
        $p = Start-Process -FilePath $Exe -ArgumentList $ArgList -PassThru -NoNewWindow -RedirectStandardError $ErrFile -RedirectStandardOutput "$ErrFile.out"
        $null = $p.Handle   # keeps the handle, so ExitCode is readable after the exit
        $p
    }
    finally {
        foreach ($k in $Env.Keys) { [Environment]::SetEnvironmentVariable($k, $old[$k]) }
    }
}

# Waits for the process's window (client larger than 100x100); returns its handle or 0.
function Wait-FoloWindow($p, [int]$TimeoutS = 25) {
    $deadline = (Get-Date).AddSeconds($TimeoutS)
    while ((Get-Date) -lt $deadline) {
        if ($p.HasExited) { return [IntPtr]::Zero }
        $h = [FoloDrive]::Find([uint32]$p.Id)
        if ($h -ne [IntPtr]::Zero) {
            $r = New-Object FoloDrive+RECT
            [FoloDrive]::GetClientRect($h, [ref]$r) | Out-Null
            if (($r.Right - $r.Left) -gt 100 -and ($r.Bottom - $r.Top) -gt 100) { return $h }
        }
        Start-Sleep -Milliseconds 50
    }
    [IntPtr]::Zero
}

# Off screen, nothing of the process may show: kills it and throws, naming every visible window of
# it that a display shows a part of (not only the app's window: a dialog is a window of its own), or
# that took the foreground. Call it after every step that could open something.
function Assert-FoloOffscreen($p) {
    $on = [FoloDrive]::OnDisplay([uint32]$p.Id)
    $fg = [FoloDrive]::GetForegroundWindow()
    $fgPid = [uint32]0
    [FoloDrive]::GetWindowThreadProcessId($fg, [ref]$fgPid) | Out-Null
    if ($on.Count -gt 0 -or $fgPid -eq $p.Id) {
        Stop-Folo $p
        if ($on.Count -gt 0) { throw "ON A DISPLAY, killed: $($on -join ' | ')" }
        throw "TOOK THE FOREGROUND, killed: $fg"
    }
}

# Waits up to $TimeoutS for a process we started to exit; true if it did.
function Wait-FoloExit($p, [double]$TimeoutS = 6) {
    $p.WaitForExit([int]($TimeoutS * 1000)) | Out-Null
    $p.HasExited
}

# Kills a process we started (by PID, never by image name).
function Stop-Folo($p) {
    if ($p -and -not $p.HasExited) {
        try { $p.Kill() } catch {}
        try { $p.WaitForExit(5000) | Out-Null } catch {}
    }
}
