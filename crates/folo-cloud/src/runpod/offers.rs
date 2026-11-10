//! RunPod's two clouds and how a GPU type of its catalog becomes a FOLO offer (`<GPU type>@secure|community`).

use folo_api::{CloudAvailability, CloudOffer};

use super::GpuType;

/// A RunPod cloud.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Cloud {
    Secure,
    Community,
}

impl Cloud {
    /// The API's spelling.
    pub(super) fn wire(self) -> &'static str {
        match self {
            Cloud::Secure => "SECURE",
            Cloud::Community => "COMMUNITY",
        }
    }

    /// The suffix of an offer id.
    fn suffix(self) -> &'static str {
        match self {
            Cloud::Secure => "secure",
            Cloud::Community => "community",
        }
    }

    /// The offer's detail.
    fn label(self) -> &'static str {
        match self {
            Cloud::Secure => "Secure Cloud",
            Cloud::Community => "Community Cloud",
        }
    }
}

/// `<gpu type id>@secure|community` → its parts.
pub(super) fn parse_offer(id: &str) -> Option<(&str, Cloud)> {
    let (gpu, cloud) = id.rsplit_once('@')?;
    let cloud = [Cloud::Secure, Cloud::Community].into_iter().find(|c| c.suffix() == cloud)?;
    (!gpu.trim().is_empty()).then_some((gpu, cloud))
}

/// The offer of `gpu` on `cloud`, when it is offered there with a price.
pub(super) fn offer(gpu: &GpuType, cloud: Cloud) -> Option<CloudOffer> {
    // Only NVIDIA: the worker finds its GPU through the NVIDIA container runtime's Vulkan driver (bootstrap).
    if gpu.manufacturer != "NVIDIA" || gpu.id.is_empty() {
        return None;
    }
    let (offered, price) = match cloud {
        Cloud::Secure => (gpu.secure, gpu.price.secure),
        Cloud::Community => (gpu.community, gpu.price.community),
    };
    if !offered {
        return None;
    }
    let price = price.filter(|p| p.is_finite() && *p > 0.0)?;
    let availability = match gpu.availability.as_deref() {
        Some("HIGH") => CloudAvailability::High,
        Some("MEDIUM") => CloudAvailability::Medium,
        Some("LOW") => CloudAvailability::Low,
        Some("NONE") => CloudAvailability::None,
        _ => CloudAvailability::Unknown,
    };
    let gib = gpu.memory.filter(|m| m.is_finite() && *m > 0.0).unwrap_or(0.0);
    Some(CloudOffer {
        id: format!("{}@{}", gpu.id, cloud.suffix()),
        gpu: if gpu.name.is_empty() { gpu.id.clone() } else { gpu.name.clone() },
        gpu_count: 1,
        vram_bytes: (gib * (1u64 << 30) as f64) as u64,
        price_per_hour: Some(price),
        region: None,
        detail: Some(cloud.label().into()),
        availability,
    })
}
