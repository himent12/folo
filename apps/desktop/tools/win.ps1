# Shared Win32 helpers for the baseline tools (dot-source it). Nothing here moves the cursor, activates a
# window or injects input: it only finds a process's window, reads rectangles and prints a window's client
# area into a bitmap.
if (-not ("DfwWin" -as [type])) {
    Add-Type -ReferencedAssemblies System.Drawing @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
public static class DfwWin {
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
    [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
    public delegate bool EnumProc(IntPtr h, IntPtr l);
    public delegate bool MonEnum(IntPtr h, IntPtr hdc, ref RECT r, IntPtr d);
    [DllImport("user32.dll")] public static extern bool SetProcessDpiAwarenessContext(IntPtr v);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc f, IntPtr l);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern bool GetCursorPos(out POINT p);
    [DllImport("user32.dll")] public static extern uint GetDpiForWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint flags);
    [DllImport("user32.dll")] public static extern bool EnumDisplayMonitors(IntPtr hdc, IntPtr clip, MonEnum f, IntPtr d);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetClassName(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);

    // The visible top-level window of a process with the largest client area (IntPtr.Zero if none).
    public static IntPtr FindWindow(uint pid) {
        IntPtr best = IntPtr.Zero; long bestArea = -1;
        EnumWindows((h, l) => {
            uint p; GetWindowThreadProcessId(h, out p);
            if (p == pid && IsWindowVisible(h)) {
                RECT r; GetClientRect(h, out r);
                long a = (long)(r.Right - r.Left) * (r.Bottom - r.Top);
                if (a > bestArea) { bestArea = a; best = h; }
            }
            return true;
        }, IntPtr.Zero);
        return best;
    }
    // Every visible top-level window of a process: "hwnd class rect".
    public static List<string> ListWindows(uint pid) {
        var l = new List<string>();
        EnumWindows((h, x) => {
            uint p; GetWindowThreadProcessId(h, out p);
            if (p == pid && IsWindowVisible(h)) {
                var c = new StringBuilder(256); GetClassName(h, c, 256);
                RECT r; GetWindowRect(h, out r);
                l.Add(h + " " + c + " (" + r.Left + "," + r.Top + ")-(" + r.Right + "," + r.Bottom + ")");
            }
            return true;
        }, IntPtr.Zero);
        return l;
    }
    public static List<RECT> Monitors() {
        var l = new List<RECT>();
        EnumDisplayMonitors(IntPtr.Zero, IntPtr.Zero, (IntPtr h, IntPtr hdc, ref RECT r, IntPtr d) => { l.Add(r); return true; }, IntPtr.Zero);
        return l;
    }
    // True when the rectangle overlaps no monitor.
    public static bool OffEveryMonitor(RECT w) {
        foreach (var m in Monitors()) {
            if (w.Left < m.Right && w.Right > m.Left && w.Top < m.Bottom && w.Bottom > m.Top) return false;
        }
        return true;
    }
    public static string Str(RECT r) { return "(" + r.Left + "," + r.Top + ")-(" + r.Right + "," + r.Bottom + ")"; }
}
"@
}
if (-not ("DfwPrint" -as [type])) {
    Add-Type -ReferencedAssemblies System.Drawing @"
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
public static class DfwPrint {
    [StructLayout(LayoutKind.Sequential)] struct BIH { public int biSize, biWidth, biHeight; public short biPlanes, biBitCount; public int biCompression, biSizeImage, x, y, c, i; }
    [DllImport("user32.dll")] static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint flags);
    [DllImport("gdi32.dll")] static extern IntPtr CreateCompatibleDC(IntPtr h);
    [DllImport("gdi32.dll")] static extern bool DeleteDC(IntPtr h);
    [DllImport("gdi32.dll")] static extern IntPtr CreateDIBSection(IntPtr dc, ref BIH bi, uint usage, out IntPtr bits, IntPtr sec, uint off);
    [DllImport("gdi32.dll")] static extern IntPtr SelectObject(IntPtr dc, IntPtr o);
    [DllImport("gdi32.dll")] static extern bool DeleteObject(IntPtr o);
    [DllImport("gdi32.dll")] static extern bool GdiFlush();
    // PrintWindow of a w x h client area into a plain GDI DIB section, copied into a Bitmap (as drive.ps1
    // does). Not through GDI+'s Graphics.GetHdc: that pre-fills its DC with RGB(13, 11, 12) and on
    // ReleaseHdc keeps the old bitmap wherever a pixel still has that colour, so every drawn pixel of
    // exactly RGB(13, 11, 12) came back black.
    public static Bitmap Print(IntPtr h, int w, int hh, uint flags) {
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
}
"@
}
# Per-monitor DPI aware v2, so rectangles and captures are in physical pixels.
[DfwWin]::SetProcessDpiAwarenessContext([IntPtr]-4) | Out-Null
Add-Type -AssemblyName System.Drawing

# Starts an exe with extra environment variables and stderr redirected to a file, returns the Process.
# The variables are set on this PowerShell process for the launch and then put back.
function Start-Dfw([string]$Exe, [string[]]$ArgList, [hashtable]$Env, [string]$ErrFile) {
    $old = @{}
    foreach ($k in $Env.Keys) { $old[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $Env[$k]) }
    try {
        $outFile = "$ErrFile.out"
        Start-Process -FilePath $Exe -ArgumentList $ArgList -PassThru -NoNewWindow -RedirectStandardError $ErrFile -RedirectStandardOutput $outFile
    }
    finally {
        foreach ($k in $Env.Keys) { [Environment]::SetEnvironmentVariable($k, $old[$k]) }
    }
}

# Waits for the process's window; returns its handle or 0.
function Wait-DfwWindow($p, [int]$TimeoutS = 25) {
    $deadline = (Get-Date).AddSeconds($TimeoutS)
    while ((Get-Date) -lt $deadline) {
        if ($p.HasExited) { return [IntPtr]::Zero }
        $h = [DfwWin]::FindWindow([uint32]$p.Id)
        if ($h -ne [IntPtr]::Zero) {
            $r = New-Object DfwWin+RECT
            [DfwWin]::GetClientRect($h, [ref]$r) | Out-Null
            if (($r.Right - $r.Left) -gt 100 -and ($r.Bottom - $r.Top) -gt 100) { return $h }
        }
        Start-Sleep -Milliseconds 50
    }
    [IntPtr]::Zero
}

# Kills a process we started (by PID, never by image name) and its window with it.
function Stop-Dfw($p) {
    if ($p -and -not $p.HasExited) {
        try { $p.Kill() } catch {}
        try { $p.WaitForExit(5000) | Out-Null } catch {}
    }
}
