use vulkanalia::prelude::v1_0::*;

pub const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

pub const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

pub const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
