//! RunPod: GPU pods rented per run. (Skeleton: filled in by the provider work.)

use folo_api::{CloudOffer, CloudProvider};

use crate::{Account, Error, ErrorKind, Provider, Result, Values};

/// RunPod: GPU pods rented per run.
#[derive(Clone, Debug, Default)]
pub struct RunPod;

impl RunPod {
    /// The provider against its real service.
    pub fn new() -> RunPod {
        RunPod
    }
}

impl Provider for RunPod {
    fn info(&self) -> CloudProvider {
        CloudProvider {
            id: "runpod".into(),
            name: "RunPod".into(),
            about: String::new(),
            docs_url: None,
            fields: vec![folo_api::CloudField {
                key: "api_key".into(),
                label: "api_key".into(),
                secret: true,
                required: true,
                placeholder: None,
                help: None,
            }],
            provisions: true,
            self_terminates: true,
        }
    }

    fn check(&self, _values: &Values) -> Result<Account> {
        Err(Error::new(ErrorKind::Unavailable, "not implemented yet"))
    }

    fn offers(&self, _values: &Values) -> Result<Vec<CloudOffer>> {
        Err(Error::new(ErrorKind::Unavailable, "not implemented yet"))
    }
}
