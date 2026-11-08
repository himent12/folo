//! Devices a tensor can live on.

/// Where a tensor's storage lives.
///
/// `Cpu` is always available. `Gpu(ordinal)` needs a backend registered at runtime (see
/// [`crate::backend::register_backend`]); folo-gpu does that.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// Host memory, computed by the built-in CPU backend.
    #[default]
    Cpu,
    /// A GPU adapter by ordinal, computed by a registered backend.
    Gpu(usize),
}

impl Device {
    /// True for [`Device::Cpu`].
    pub fn is_cpu(self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// True for any [`Device::Gpu`].
    pub fn is_gpu(self) -> bool {
        matches!(self, Device::Gpu(_))
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Gpu(i) => write!(f, "gpu:{i}"),
        }
    }
}
