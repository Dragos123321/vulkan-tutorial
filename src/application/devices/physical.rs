use anyhow::{anyhow, Result};
use log::*;

use vulkanalia::prelude::v1_0::*;
use vulkanalia::Instance;

use crate::application::errors::SuitabilityError;
use crate::application::queue_families::QueueFamilyIndices;
use crate::application::swapchain_support::SwapchainSupport;
use crate::application::{AppData, DEVICE_EXTENSIONS};

pub unsafe fn pick_physical_device(instance: &Instance, app_data: &mut AppData) -> Result<()> {
    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);

        if let Err(error) = check_physical_device(instance, app_data, physical_device) {
            warn!(
                "Skipping physical device (`{}`): {}",
                properties.device_name, error
            )
        } else {
            info!("Selected physical device (`{}`)", properties.device_name);
            app_data.physical_device = physical_device;
            return Ok(());
        }
    }

    Err(anyhow!("Failed to find suitable physical device."))
}

pub unsafe fn check_physical_device(
    instance: &Instance,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let properties = instance.get_physical_device_properties(physical_device);

    if properties.device_type != vk::PhysicalDeviceType::INTEGRATED_GPU {
        return Err(anyhow!(SuitabilityError(
            "Only integrated GPUs are supported."
        )));
    }

    let features = instance.get_physical_device_features(physical_device);

    if features.geometry_shader != vk::TRUE {
        return Err(anyhow!(SuitabilityError(
            "Missing geometry shader support."
        )));
    }

    QueueFamilyIndices::get(instance, data, physical_device)?;
    check_physical_device_extensions(instance, physical_device)?;

    let swapchain_support = SwapchainSupport::get(instance, data, physical_device)?;

    if swapchain_support.formats.is_empty() || swapchain_support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
    }

    Ok(())
}

pub unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<Vec<_>>();

    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError(
            "Missing required device extensions."
        )))
    }
}
