//! A child process together with everything it starts, so that all of it can be ended at once.
//!
//! An agent CLI is often a shim around the process that does the work (a batch file or a node script when it was
//! installed through npm), and killing the shim alone would leave that process running, and spending. On Unix
//! the child leads a process group and the group is killed. On Windows the child is created suspended, put in a
//! job object and only then let run, so nothing it starts can be outside the job, and the job is terminated.
//! Both still reach the descendants after the child itself has exited.

use std::process::{Child, Command, Stdio};

/// The process tree of one child.
pub struct Tree {
    #[cfg(windows)]
    job: Option<windows::JobObject>,
}

impl Tree {
    /// Sets `cmd` up so that the process it starts can be tracked as a tree ([`Tree::start`] must follow the
    /// spawn: on Windows the process waits for it), and has no console window of its own (the server may have
    /// none: a desktop app started it).
    pub fn prepare(cmd: &mut Command) {
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(windows::CREATE_NO_WINDOW | windows::CREATE_SUSPENDED);
        }
        #[cfg(not(any(unix, windows)))]
        let _ = cmd;
    }

    /// Takes the just-spawned `child` of a [`Tree::prepare`]d command into a tree and lets it run. An error
    /// means the child cannot run (it is still suspended): the caller kills it.
    pub fn start(child: &Child) -> std::io::Result<Tree> {
        #[cfg(windows)]
        {
            // Without a job (Windows refused one) the call still runs; its tree is then found through the child.
            let job = windows::JobObject::holding(child);
            if !windows::resume(child.id()) {
                return Err(std::io::Error::other("the process was created but could not be started"));
            }
            Ok(Tree { job })
        }
        #[cfg(not(windows))]
        {
            let _ = child;
            Ok(Tree {})
        }
    }

    /// Kills everything that is left of the tree, the child included, and reaps the child.
    pub fn kill(&self, child: &mut Child) {
        #[cfg(windows)]
        match &self.job {
            Some(job) => job.terminate(),
            None => quiet(Command::new("taskkill").args(["/T", "/F", "/PID", &child.id().to_string()])),
        }
        // The child leads a process group of its own; a negative pid names the group.
        #[cfg(unix)]
        quiet(Command::new("kill").args(["-KILL", "--", &format!("-{}", child.id())]));
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// Runs a helper command for its effect.
#[cfg(any(unix, windows))]
fn quiet(cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(windows::CREATE_NO_WINDOW);
    }
    let _ = cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status();
}

#[cfg(windows)]
mod windows {
    use std::os::windows::io::AsRawHandle;
    use std::process::Child;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
    };
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, TerminateJobObject,
    };
    use windows_sys::Win32::System::Threading::{OpenThread, ResumeThread, THREAD_SUSPEND_RESUME};

    pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    pub const CREATE_SUSPENDED: u32 = 0x0000_0004;

    /// A Windows job object: the processes assigned to it, and every process they start, stay members.
    pub struct JobObject(HANDLE);

    impl JobObject {
        /// A new job with the (suspended) `child` in it, or `None` when Windows refuses either step.
        pub fn holding(child: &Child) -> Option<JobObject> {
            // SAFETY: both arguments may be null (default security, no name); the result is checked.
            let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if handle.is_null() {
                return None;
            }
            let job = JobObject(handle);
            // SAFETY: `job.0` is the job handle just created and `child` owns a valid process handle.
            let assigned = unsafe { AssignProcessToJobObject(job.0, child.as_raw_handle() as HANDLE) };
            (assigned != 0).then_some(job)
        }

        /// Ends every process in the job.
        pub fn terminate(&self) {
            // SAFETY: `self.0` is a job handle this value owns.
            unsafe { TerminateJobObject(self.0, 1) };
        }
    }

    impl Drop for JobObject {
        fn drop(&mut self) {
            // SAFETY: `self.0` is a handle this value owns, closed exactly once. Closing the job does not end
            // its processes: a call that finished normally leaves what it started alone.
            unsafe { CloseHandle(self.0) };
        }
    }

    /// Lets the process `pid`, created suspended, run: resumes its threads (it has the one it was created
    /// with). False when none could be resumed.
    pub fn resume(pid: u32) -> bool {
        // SAFETY: the snapshot handle is checked and closed; `entry` is a plain struct with its size set, as
        // the ToolHelp calls require; each thread handle is checked, used and closed here.
        unsafe {
            let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            if snapshot == INVALID_HANDLE_VALUE {
                return false;
            }
            let mut entry: THREADENTRY32 = std::mem::zeroed();
            entry.dwSize = size_of::<THREADENTRY32>() as u32;
            let mut resumed = false;
            let mut more = Thread32First(snapshot, &mut entry);
            while more != 0 {
                if entry.th32OwnerProcessID == pid {
                    let thread = OpenThread(THREAD_SUSPEND_RESUME, 0, entry.th32ThreadID);
                    if !thread.is_null() {
                        resumed |= ResumeThread(thread) != u32::MAX;
                        CloseHandle(thread);
                    }
                }
                more = Thread32Next(snapshot, &mut entry);
            }
            CloseHandle(snapshot);
            resumed
        }
    }
}
