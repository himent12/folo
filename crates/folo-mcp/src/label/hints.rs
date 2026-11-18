//! The hints of labeling errors in each front end's words. The server's hints name HTTP requests and the
//! operations of [`super::ops`] give none; [`decorate`] replaces them by code, naming the MCP tools or the
//! `folo label` commands that help.

use super::CreateFailed;
use crate::errors::Error;

/// Who the error is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Front {
    /// An agent, through the MCP tools.
    Mcp,
    /// A person or a script, through `folo label`.
    Cli,
}

/// `e` with a hint in `front`'s words (an error whose code has nothing to add keeps its hint).
pub fn decorate(mut e: Error, front: Front) -> Error {
    let mcp = front == Front::Mcp;
    let hint = match e.body.code.as_str() {
        "label_job_not_found" => {
            Some(if mcp { "list_label_jobs lists the jobs" } else { "`folo label jobs` lists the jobs" })
        }
        // Only for an id the job does not have: an image whose file is gone keeps the server's hint (put the
        // folder back).
        "image_not_found" if !e.body.message.contains("no longer") => Some(if mcp {
            "list_label_images lists the job's images"
        } else {
            "`folo label images JOB` lists the job's images"
        }),
        "label_boxes_changed" => Some(if mcp {
            "view the image again (view_label_image with its image_id) and redo the edit on the boxes it lists"
        } else {
            "`folo label view JOB IMAGE` shows the boxes now; redo the edit on them"
        }),
        "label_job_active"
            if e.body.message.contains("agent run") || e.body.message.contains("with the agent") =>
        {
            Some(if mcp {
                "wait for the run (label_status), or end it with stop_labeling"
            } else {
                "wait for the run (`folo label status JOB`), or end it with `folo label stop JOB`"
            })
        }
        "label_job_not_active" => Some(if mcp {
            "label_status has the last run's result"
        } else {
            "`folo label status JOB` has the last run's result"
        }),
        "invalid_body" if e.body.message.starts_with("nothing to label") => Some(if mcp {
            "add images (create_label_job with job_id), or send scope: \"all\" to label every image again"
        } else {
            "add images (`folo label add JOB PATH`), or pass --all to label every image again"
        }),
        "invalid_dataset" if e.body.message.contains("no labeled image") => Some(if mcp {
            "label images first: start_labeling, or set_label_boxes"
        } else {
            "label images first: `folo label start JOB`, or `folo label set`"
        }),
        "not_supported" if e.body.message.contains("is not installed") => {
            let other =
                if mcp { "start_labeling with the other agent" } else { "--agent with the other one" };
            let install = e.body.hint.as_deref().unwrap_or("install it on the server's machine");
            e.body.hint = Some(format!("{other} (claude or codex), or: {install}"));
            None
        }
        "invalid_arguments" if !mcp => Some("see `folo label <command> --help`"),
        _ => None,
    };
    if let Some(h) = hint {
        e.body.hint = Some(h.to_string());
    }
    e
}

/// A failed create in `front`'s words: when a job was made before a path failed, the hint says how to add the
/// rest to it.
pub fn create_failed(f: CreateFailed, front: Front) -> Error {
    let e = decorate(f.error, front);
    match f.made {
        Some(id) => {
            let how = match front {
                Front::Mcp => format!("create_label_job with job_id: \"{id}\""),
                Front::Cli => format!("`folo label add {id} PATH`"),
            };
            let hint = format!("job {id} was made with the paths before this one; add the rest with {how}");
            e.hint(hint)
        }
        None => e,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::StatusCode;

    #[test]
    fn hints_name_the_front_ends_own_commands() {
        let e = || Error::conflict("label_job_active", "labeling job 'lj_1' has an agent run in progress; x");
        assert!(decorate(e(), Front::Mcp).body.hint.unwrap().contains("stop_labeling"));
        assert!(decorate(e(), Front::Cli).body.hint.unwrap().contains("folo label stop"));
        let nothing =
            Error::new(StatusCode::BAD_REQUEST, "invalid_body", "nothing to label: every image is labeled")
                .hint("send \"scope\": \"all\"");
        assert!(decorate(nothing, Front::Cli).body.hint.unwrap().contains("--all"));
        let made = CreateFailed { error: Error::bad_request("no such folder"), made: Some("lj_9".into()) };
        let hint = create_failed(made, Front::Mcp).body.hint.unwrap();
        assert!(hint.contains("job_id: \"lj_9\""), "{hint}");
        assert!(!hint.contains("  "), "{hint}");
        let gone =
            Error::new(StatusCode::NOT_FOUND, "image_not_found", "2 images no longer on disk, such as a.jpg")
                .hint("put the folder back where it was");
        assert_eq!(decorate(gone, Front::Mcp).body.hint.unwrap(), "put the folder back where it was");
        let unknown = Error::new(StatusCode::NOT_FOUND, "image_not_found", "image 'x' does not exist");
        assert!(decorate(unknown, Front::Mcp).body.hint.unwrap().contains("list_label_images"));
    }
}
