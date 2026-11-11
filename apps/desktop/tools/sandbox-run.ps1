# Runs a command on a private Win32 desktop that is never shown: whatever it opens (maximized windows,
# full screen, dialogs, activation) happens there, never on the desktop you are looking at. For the
# window checks that cannot run off screen (window-check.ps1 -Sandbox, chrome-check.ps1 -Sandbox,
# interact-check.ps1 -Sandbox, the win_smoke example's FOLO_SMOKE_SANDBOX=1 checks).
#
#   tools\sandbox-run.ps1 -Command "& tools\window-check.ps1 -Sandbox -Exe D:\b\release\folo-desktop.exe"
#   tools\sandbox-run.ps1 -Exe D:\b\release\examples\win_smoke.exe -ExeArgs "--dialog" -Env @{ FOLO_SMOKE_SANDBOX = "1" }
#
# How: CreateDesktopW makes a desktop in this window station (never SwitchDesktop'd to, and the handle
# has no DESKTOP_SWITCHDESKTOP right), CreateProcessW starts the command with STARTUPINFO.lpDesktop
# naming it, inside a job object that kills every process the command started when it ends or times out.
# Children inherit the desktop. Output goes to a log file that is printed at the end. Real input
# (SendInput, SetCursorPos) goes to the input desktop, which is yours: a check run here must drive its
# windows with posted messages only. The script also checks that your foreground window is the same
# before and after. Exit code: the command's (or 124 on a timeout).
param(
    [string]$Command = "",
    [string]$Exe = "",
    [string]$ExeArgs = "",
    [hashtable]$Env = @{},
    [int]$TimeoutS = 900,
    [string]$Log = ""
)
$ErrorActionPreference = "Stop"
if (-not $Command -and -not $Exe) { throw "pass -Command (PowerShell) or -Exe" }
if (-not ("FoloSandbox" -as [type])) {
    Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class FoloSandbox {
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    public struct STARTUPINFO {
        public int cb; public string lpReserved; public string lpDesktop; public string lpTitle;
        public int dwX, dwY, dwXSize, dwYSize, dwXCountChars, dwYCountChars, dwFillAttribute, dwFlags;
        public short wShowWindow, cbReserved2; public IntPtr lpReserved2, hStdInput, hStdOutput, hStdError;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct PROCESS_INFORMATION { public IntPtr hProcess, hThread; public int dwProcessId, dwThreadId; }
    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
        public long PerProcessUserTimeLimit, PerJobUserTimeLimit; public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize; public uint ActiveProcessLimit;
        public UIntPtr Affinity; public uint PriorityClass, SchedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct IO_COUNTERS { public ulong a, b, c, d, e, f; }
    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation; public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed;
    }
    [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern IntPtr CreateDesktopW(string name, IntPtr device, IntPtr devmode, uint flags, uint access, IntPtr sa);
    [DllImport("user32.dll", SetLastError = true)] public static extern bool CloseDesktop(IntPtr h);
    [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern IntPtr CreateWindowStationW(string name, uint flags, uint access, IntPtr sa);
    [DllImport("user32.dll", SetLastError = true)] public static extern bool CloseWindowStation(IntPtr h);
    [DllImport("user32.dll", SetLastError = true)] public static extern IntPtr GetProcessWindowStation();
    [DllImport("user32.dll", SetLastError = true)] public static extern bool SetProcessWindowStation(IntPtr h);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern bool CreateProcessW(string app, System.Text.StringBuilder cmd, IntPtr pa, IntPtr ta, bool inherit,
        uint flags, IntPtr env, string cwd, ref STARTUPINFO si, out PROCESS_INFORMATION pi);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern uint ResumeThread(IntPtr h);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern uint WaitForSingleObject(IntPtr h, uint ms);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool GetExitCodeProcess(IntPtr h, out uint code);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool CloseHandle(IntPtr h);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] public static extern IntPtr CreateJobObjectW(IntPtr sa, string name);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool SetInformationJobObject(IntPtr job, int cls, ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION info, int size);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool TerminateJobObject(IntPtr job, uint code);

    // Every desktop right except DESKTOP_SWITCHDESKTOP (0x0100): nobody can show it through this handle.
    public const uint Access = 0x00FF | 0x000F0000;

    public static IntPtr NewDesktop(string name) {
        IntPtr d = CreateDesktopW(name, IntPtr.Zero, IntPtr.Zero, 0, Access, IntPtr.Zero);
        if (d == IntPtr.Zero) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "CreateDesktopW");
        return d;
    }
    // A job that kills everything in it when its handle closes.
    public static IntPtr NewJob() {
        IntPtr job = CreateJobObjectW(IntPtr.Zero, null);
        var info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
        info.BasicLimitInformation.LimitFlags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        SetInformationJobObject(job, 9, ref info, Marshal.SizeOf(info));
        return job;
    }
    // Starts cmd suspended on WinSta0\desktop, puts it in the job, resumes it; returns the process handle.
    public static IntPtr Start(string cmd, string desktop, string cwd, IntPtr job, out int pid) {
        var si = new STARTUPINFO();
        si.cb = Marshal.SizeOf(si);
        si.lpDesktop = "WinSta0\\" + desktop;
        PROCESS_INFORMATION pi;
        // CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT
        if (!CreateProcessW(null, new System.Text.StringBuilder(cmd), IntPtr.Zero, IntPtr.Zero, false,
                0x4 | 0x08000000 | 0x400, IntPtr.Zero, cwd, ref si, out pi))
            throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "CreateProcessW");
        AssignProcessToJobObject(job, pi.hProcess);
        ResumeThread(pi.hThread);
        CloseHandle(pi.hThread);
        pid = pi.dwProcessId;
        return pi.hProcess;
    }
}
"@
}

$tmp = if ($env:FOLO_TEST_TMP) { $env:FOLO_TEST_TMP } else { [System.IO.Path]::GetTempPath() }
if (-not (Test-Path $tmp)) { New-Item -ItemType Directory -Force $tmp | Out-Null }
if (-not $Log) { $Log = Join-Path $tmp "sandbox-run-$PID.log" }
$Log = [System.IO.Path]::GetFullPath($Log)
Remove-Item -ErrorAction SilentlyContinue $Log

# The command line: PowerShell with its whole output (all streams) in the log, keeping the exit code.
$q = { param($s) "'" + ($s -replace "'", "''") + "'" }
if ($Exe) {
    $Command = "& $(& $q ([System.IO.Path]::GetFullPath($Exe))) $ExeArgs"
}
$inner = "`$ErrorActionPreference = 'Continue'; & { $Command } *>&1 | Out-File -Encoding utf8 -Width 400 $(& $q $Log); exit `$LASTEXITCODE"
$encoded = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($inner))
$ps = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$cmdLine = "`"$ps`" -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand $encoded"

$name = "folo-sandbox-$PID-$([Environment]::TickCount)"
$fg0 = [FoloSandbox]::GetForegroundWindow()
$old = @{}
foreach ($k in $Env.Keys) { $old[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, [string]$Env[$k]) }
$desk = [FoloSandbox]::NewDesktop($name)
$job = [FoloSandbox]::NewJob()
$code = 1
try {
    $childPid = 0
    $hp = [FoloSandbox]::Start($cmdLine, $name, (Get-Location).Path, $job, [ref]$childPid)
    foreach ($k in $Env.Keys) { [Environment]::SetEnvironmentVariable($k, $old[$k]) }
    "sandbox: desktop WinSta0\$name, pid $childPid, log $Log"
    $wait = [FoloSandbox]::WaitForSingleObject($hp, [uint32]($TimeoutS * 1000))
    if ($wait -ne 0) {
        "sandbox: timed out after $TimeoutS s; killing everything it started"
        [FoloSandbox]::TerminateJobObject($job, 124) | Out-Null
        $code = 124
    } else {
        $c = [uint32]0
        [FoloSandbox]::GetExitCodeProcess($hp, [ref]$c) | Out-Null
        $code = [int]$c
    }
    [FoloSandbox]::CloseHandle($hp) | Out-Null
}
finally {
    foreach ($k in $Env.Keys) { [Environment]::SetEnvironmentVariable($k, $old[$k]) }
    # Closing the job kills whatever the command left running; the desktop goes when nothing uses it.
    [FoloSandbox]::CloseHandle($job) | Out-Null
    [FoloSandbox]::CloseDesktop($desk) | Out-Null
}
if (Test-Path $Log) { Get-Content -Encoding utf8 $Log }
$fg1 = [FoloSandbox]::GetForegroundWindow()
if ($fg0 -ne $fg1) { "sandbox: NOTE your foreground window changed during the run ($fg0 -> $fg1); a sandboxed process cannot set it, so this was probably you" }
"sandbox: exit code $code"
exit $code
