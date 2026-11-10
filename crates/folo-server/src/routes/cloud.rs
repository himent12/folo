//! `/api/cloud`: providers, connections, offers and instances.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use folo_api::{
    CloudConnection, CloudConnectionList, CloudInstance, CloudInstanceList, CloudOfferList,
    CloudProviderList, ConnectCloud,
};
use serde::Deserialize;

use crate::AppState;
use crate::error::{ApiJson, ApiPath, ApiQuery, Result};

use super::blocking;

// ---- cloud -------------------------------------------------------------------------------------------------------

pub(super) async fn cloud_providers(State(st): State<AppState>) -> Result<Json<CloudProviderList>> {
    Ok(Json(CloudProviderList { providers: st.store.cloud_providers()? }))
}

pub(super) async fn cloud_connections(State(st): State<AppState>) -> Result<Json<CloudConnectionList>> {
    Ok(Json(CloudConnectionList { connections: blocking(&st, |s| s.cloud_connections()).await? }))
}

/// `POST /api/cloud/connections`: checks the values with the provider, then stores them (`201`).
pub(super) async fn connect_cloud(
    State(st): State<AppState>,
    ApiJson(req): ApiJson<ConnectCloud>,
) -> Result<Response> {
    let c = blocking(&st, move |s| s.connect_cloud(req)).await?;
    Ok((StatusCode::CREATED, Json(c)).into_response())
}

pub(super) async fn check_cloud(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<Json<CloudConnection>> {
    Ok(Json(blocking(&st, move |s| s.check_cloud(&id)).await?))
}

pub(super) async fn disconnect_cloud(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<StatusCode> {
    blocking(&st, move |s| s.disconnect_cloud(&id)).await?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
pub(super) struct OffersParams {
    #[serde(default)]
    refresh: bool,
}

pub(super) async fn cloud_offers(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiQuery(p): ApiQuery<OffersParams>,
) -> Result<Json<CloudOfferList>> {
    Ok(Json(blocking(&st, move |s| s.cloud_offers(&id, p.refresh)).await?))
}

pub(super) async fn cloud_instances(State(st): State<AppState>) -> Result<Json<CloudInstanceList>> {
    Ok(Json(CloudInstanceList { instances: blocking(&st, |s| s.cloud_instances()).await? }))
}

pub(super) async fn terminate_instance(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<Json<CloudInstance>> {
    Ok(Json(blocking(&st, move |s| s.terminate_instance(&id)).await?))
}
