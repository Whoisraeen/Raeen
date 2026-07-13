//! Error types for XPS5X.
//!
//! Defines the unified error hierarchy used across all XPS5X crates.
//! Each subsystem has its own error variant for clean error propagation.

use thiserror::Error;

/// Top-level XPS5X error type.
#[derive(Debug, Error)]
pub enum XPS5XError {
    /// Binary loader errors (ELF, SELF, PKG parsing).
    #[error("Loader error: {0}")]
    Loader(#[from] LoaderError),

    /// Kernel / syscall errors.
    #[error("Kernel error: {0}")]
    Kernel(#[from] KernelError),

    /// GPU / graphics errors.
    #[error("GPU error: {0}")]
    Gpu(#[from] GpuError),

    /// Audio errors.
    #[error("Audio error: {0}")]
    Audio(#[from] AudioError),

    /// I/O complex errors.
    #[error("I/O error: {0}")]
    Io(#[from] IoError),

    /// Firmware ingestion errors (PUP, module loading, dynamic linking).
    #[error("Firmware error: {0}")]
    Firmware(#[from] FirmwareError),

    /// Configuration errors.
    #[error("Config error: {0}")]
    Config(String),

    /// Generic / uncategorized errors.
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

/// Errors from the binary loader subsystem.
#[derive(Debug, Error)]
pub enum LoaderError {
    #[error("Invalid ELF magic: expected 0x7F454C46, got {0:#010x}")]
    InvalidElfMagic(u32),

    #[error("Unsupported ELF class: expected 64-bit (2), got {0}")]
    UnsupportedElfClass(u8),

    #[error("Unsupported ELF architecture: expected x86-64 (0x3E), got {0:#x}")]
    UnsupportedArchitecture(u16),

    #[error("Invalid SELF magic: expected 0x4F15D17E, got {0:#010x}")]
    InvalidSelfMagic(u32),

    #[error("SELF decryption required — encrypted firmware modules cannot be loaded directly")]
    EncryptedSelf,

    #[error("Invalid PKG magic: expected 0x7F434E54, got {0:#010x}")]
    InvalidPkgMagic(u32),

    #[error("Failed to load segment at address {address:#x}, size {size:#x}: {reason}")]
    SegmentLoadFailed {
        address: u64,
        size: u64,
        reason: String,
    },

    #[error("Dynamic library not found: {0}")]
    LibraryNotFound(String),

    #[error("Unresolved symbol: {0}")]
    UnresolvedSymbol(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors from the kernel HLE subsystem.
#[derive(Debug, Error)]
pub enum KernelError {
    #[error("Unimplemented syscall: {name} (number {number})")]
    UnimplementedSyscall { number: u64, name: String },

    #[error("Memory mapping failed: address {address:#x}, size {size:#x}")]
    MmapFailed { address: u64, size: u64 },

    #[error("Invalid memory access at address {0:#x}")]
    InvalidMemoryAccess(u64),

    #[error("Thread creation failed: {0}")]
    ThreadCreationFailed(String),

    #[error("File not found in virtual filesystem: {0}")]
    FileNotFound(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),
}

/// Errors from the GPU translation subsystem.
#[derive(Debug, Error)]
pub enum GpuError {
    #[error("Vulkan initialization failed: {0}")]
    VulkanInitFailed(String),

    #[error("Metal initialization failed: {0}")]
    MetalInitFailed(String),

    #[error("No suitable GPU device found")]
    NoSuitableDevice,

    #[error("Shader compilation failed: {0}")]
    ShaderCompilationFailed(String),

    #[error("Unknown PM4 opcode: {0:#x}")]
    UnknownPm4Opcode(u32),

    #[error("Invalid GPU register write: register {register:#x}, value {value:#x}")]
    InvalidRegisterWrite { register: u32, value: u32 },

    #[error("Pipeline creation failed: {0}")]
    PipelineCreationFailed(String),

    #[error("Unsupported texture format: {0:#x}")]
    UnsupportedTextureFormat(u32),
}

/// Errors from the audio subsystem.
#[derive(Debug, Error)]
pub enum AudioError {
    #[error("Audio device initialization failed: {0}")]
    DeviceInitFailed(String),

    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),
}

/// Errors from the I/O complex subsystem.
#[derive(Debug, Error)]
pub enum IoError {
    #[error("Decompression failed: {0}")]
    DecompressionFailed(String),

    #[error("DMA transfer error: source {src:#x}, dest {dst:#x}, size {size:#x}")]
    DmaTransferError { src: u64, dst: u64, size: u64 },
}
