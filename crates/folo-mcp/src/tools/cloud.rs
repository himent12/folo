//! The cloud tools: connections, offers, instances and terminating one.

use folo_api::{CloudConnectionList, CloudInstanceList};

use super::results::{Output, out};
use crate::errors::Error;
use crate::server::Inner;
use crate::types::*;

impl Inner {
    pub(super) async fn cloud_connections(&self, _: NoArgs) -> Result<Output, Error> {
        let connections = self.blocking(|s| s.cloud_connections()).await?;
        let names: Vec<String> = connections
            .iter()
            .map(|c| {
                format!("{} ({}, {})", c.id, c.provider, serde_json::to_value(c.status).unwrap_or_default())
            })
            .collect();
        let summary = match names.len() {
            0 => "no cloud connections; make one in Studio, Desktop or with `folo cloud connect`".to_string(),
            n => format!("{n} cloud connection(s): {}", names.join(", ")),
        };
        out(&CloudConnectionList { connections }, summary)
    }

    pub(super) async fn cloud_offers(&self, a: CloudOffersArgs) -> Result<Output, Error> {
        let id = a.connection_id.clone();
        let list = self.blocking(move |s| s.cloud_offers(&id, a.refresh)).await?;
        let cheapest =
            list.offers.iter().filter(|o| o.availability != folo_api::CloudAvailability::None).find_map(
                |o| o.price_per_hour.map(|p| format!("; the cheapest available: {} at ${p:.2}/h", o.gpu)),
            );
        let summary =
            format!("{} offers of {}{}", list.offers.len(), a.connection_id, cheapest.unwrap_or_default());
        out(&list, summary)
    }

    pub(super) async fn cloud_instances(&self, _: NoArgs) -> Result<Output, Error> {
        let instances = self.blocking(|s| s.cloud_instances()).await?;
        let alive = instances.iter().filter(|i| i.status.is_alive()).count();
        let summary =
            format!("{alive} alive, {} listed (the ended ones of the last 7 days)", instances.len());
        out(&CloudInstanceList { instances }, summary)
    }

    pub(super) async fn cloud_terminate(&self, a: InstanceIdArgs) -> Result<Output, Error> {
        let i = self.blocking(move |s| s.terminate_instance(&a.instance_id)).await?;
        let status = serde_json::to_value(i.status).unwrap_or_default();
        let summary = match &i.error {
            Some(e) => format!("{}: {status} ({e}; FOLO retries)", i.id),
            None => format!("{}: {status}", i.id),
        };
        out(&i, summary)
    }
}
