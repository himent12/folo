//! A FOLO server that already runs. (Skeleton: filled in by the provider work.)

use folo_api::{CloudOffer, CloudProvider};

use crate::{Account, Error, ErrorKind, Provider, Result, Values};

/// A FOLO server that already runs.
#[derive(Clone, Debug, Default)]
pub struct Remote;

impl Remote {
    /// The provider against its real service.
    pub fn new() -> Remote {
        Remote
    }
}

impl Provider for Remote {
    fn info(&self) -> CloudProvider {
        CloudProvider {
            id: "remote".into(),
            name: "Remote FOLO server".into(),
            about: String::new(),
            docs_url: None,
            fields: vec![folo_api::CloudField {
                key: "token".into(),
                label: "token".into(),
                secret: true,
                required: true,
                placeholder: None,
                help: None,
            }],
            provisions: false,
            self_terminates: false,
        }
    }

    fn check(&self, _values: &Values) -> Result<Account> {
        Err(Error::new(ErrorKind::Unavailable, "not implemented yet"))
    }

    fn offers(&self, _values: &Values) -> Result<Vec<CloudOffer>> {
        Err(Error::new(ErrorKind::Unavailable, "not implemented yet"))
    }
}
