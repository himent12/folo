//! A child process together with everything it starts, so that all of it can be ended at once.
//!
//! An agent CLI is often a shim around the process that does the work (a batch file or a node script when it was
//! installed through npm), and killing the shim alone would leave that process running, and spending. On Unix
//! the child leads a process group and the group is killed; on Windows the child is put in a job object and the
//! job is terminated. Both still reach the descendants after the child itself has exited.

use std::process::{Child, Command, Stdio};

/// The process tree of one child.
pub struct Tree {
    #[cfg(windows)]
    job: Option<job::JobObject>,
}

impl Tree {
    /// Sets `cmd` up so that the process it starts can be tracked as a tree, and has no console window of its own
    /// (the server may have none: a desktop app started it).
    pub fn prepare(cmd: &mut Command) {
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        #[cfg(windows)]
        no_window(cmd);
        #[cfg(not(any(unix, windows)))]
        let _ = cmd;
    }

    /// The tree of `child`, which was just started from a [`Tree::prepare`]d command.
    pub fn of(child: &Child) -> Tree {
        #[cfg(windows)]
        return Tree { job: job::JobObject::holding(child) };
        #[cfg(not(windows))]
        {
            let _ = child;
            Tree {}
        }
    }

    /// Kills everything that is left of the tree, the child included, and reaps the child.
    pub fn kill(&self, child: &mut Child) {
        #[cfg(windows)]
        match &self.job {
            Some(job) => job.terminate(),
            // Without a job (it could not be created), the tree is found through the child while it lives.
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
    no_window(cmd);
    let _ = cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status();
}

#[cfg(windows)]
fn no_window(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    cmd.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(windows)]
mod job {
    use std::os::windows::io::AsRawHandle;
    use std::process::Child;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, TerminateJobObject,
    };

    /// A Windows job object: the processes assigned to it, and every process they start, stay members.
    pub struct JobObject(HANDLE);

    impl JobObject {
        /// A new job with `child` in it, or `None` when Windows refuses either step. (A process the child
        /// started before it was assigned would be outside the job; a CLI does not get that far in the
        /// microseconds between its start and this call.)
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
}
