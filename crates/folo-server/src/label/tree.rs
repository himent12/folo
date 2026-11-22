//! A child process together with everything it starts, so that all of it can be ended at once.
//!
//! An agent CLI is often a shim around the process that does the work (a batch file or a node script when it was
//! installed through npm), and killing the shim alone would leave that process running, and spending. On Unix
//! the child leads a process group and the group is killed. On Windows the child is created suspended, put in a
//! job object and only then let run, so nothing it starts can be outside the job, and the job is terminated.
//! Both still reach the descendants after the child itself has exited. On Windows the job also ends its processes
//! when its last handle closes: when the call is over, and when the server dies (killed, crashed), so no agent CLI
//! outlives the server that started it, holding its pipes open.

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
        Self::start_inner(child, false)
    }

    /// Like [`Tree::start`], but on Windows fails closed if a job object cannot be assigned so descendant
    /// termination never depends on finding children through an already-exited parent.
    pub fn start_strict(child: &Child) -> std::io::Result<Tree> {
        Self::start_inner(child, true)
    }

    fn start_inner(child: &Child, require_job: bool) -> std::io::Result<Tree> {
        #[cfg(windows)]
        {
            // Without a job (Windows refused one) normal calls still run; strict calls fail closed before resuming.
            let job = windows::JobObject::holding(child);
            if require_job && job.is_none() {
                return Err(std::io::Error::other("could not assign child process to a Windows job object"));
            }
            if !windows::resume(child.id()) {
                return Err(std::io::Error::other("the process was created but could not be started"));
            }
            Ok(Tree { job })
        }
        #[cfg(not(windows))]
        {
            let _ = (child, require_job);
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
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation, SetInformationJobObject,
        TerminateJobObject,
    };
    use windows_sys::Win32::System::Threading::{OpenThread, ResumeThread, THREAD_SUSPEND_RESUME};

    pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    pub const CREATE_SUSPENDED: u32 = 0x0000_0004;

    /// A Windows job object: the processes assigned to it, and every process they start, stay members, and they
    /// end when its handle closes (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`).
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
            // SAFETY: `limits` is a plain struct of numbers, zero apart from the flag, passed with its size to the
            // job handle just created. Should Windows refuse the limit, the job still ends its processes on
            // `terminate`; only a server that dies leaves them.
            unsafe {
                let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                SetInformationJobObject(
                    job.0,
                    JobObjectExtendedLimitInformation,
                    (&raw const limits).cast(),
                    size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );
            }
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
            // SAFETY: `self.0` is a handle this value owns, closed exactly once. Closing the job ends what is
            // left in it (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`): nothing a call started outlives the call.
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

#[cfg(all(test, windows))]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    #[test]
    fn a_tree_whose_job_closes_ends_with_it() {
        // What happens to a call's tree when the server dies: its job handle closes without `kill`.
        let mut cmd = Command::new("cmd");
        cmd.args(["/C", "ping -n 30 127.0.0.1 > NUL"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        Tree::prepare(&mut cmd);
        let mut child = cmd.spawn().unwrap();
        let tree = Tree::start(&child).unwrap();
        assert!(tree.job.is_some());
        drop(tree);
        let t0 = Instant::now();
        while child.try_wait().unwrap().is_none() {
            assert!(t0.elapsed() < Duration::from_secs(10), "the tree outlived its job");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}
