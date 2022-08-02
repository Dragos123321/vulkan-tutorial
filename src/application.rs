#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_argements
)]

pub mod command_buffers;
pub mod constants;
pub mod devices;
pub mod errors;
pub mod framebuffers;
pub mod queue_families;
pub mod render_pass;
pub mod shaders;
pub mod swapchain_support;

use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::c_void;

use anyhow::{anyhow, Result};
use log::*;

use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::ExtDebugUtilsExtension;
use vulkanalia::vk::KhrSurfaceExtension;
use vulkanalia::vk::KhrSwapchainExtension;
use vulkanalia::window as vk_window;

use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use self::command_buffers::*;
use self::constants::*;
use self::devices::logical::*;
use self::devices::physical::*;
use self::framebuffers::*;
use self::render_pass::*;
use self::shaders::*;
use self::swapchain_support::*;

#[derive(Clone, Debug)]
pub struct App {
    entry: Entry,
    instance: Instance,
    data: AppData,
    device: Device,
    frame: usize,
}

impl App {
    pub unsafe fn new(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|e| anyhow!("{}", e))?;
        let mut data = AppData::default();
        let instance = Self::create_instance(window, &entry, &mut data)?;

        data.surface = vk_window::create_surface(&instance, window)?;

        pick_physical_device(&instance, &mut data)?;

        let device = create_logical_device(&instance, &mut data)?;

        create_swapchain(window, &instance, &device, &mut data)?;
        create_swapchain_image_views(&device, &mut data)?;

        create_render_pass(&instance, &device, &mut data)?;
        create_pipeline(&device, &mut data)?;

        create_framebuffers(&device, &mut data)?;

        create_command_pool(&instance, &device, &mut data)?;
        create_command_buffers(&device, &mut data)?;
        create_sync_objects(&device, &mut data)?;

        Ok(Self {
            entry,
            instance,
            data,
            device,
            frame: 0,
        })
    }

    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        self.device.wait_for_fences(
            &[self.data.in_flight_fences[self.frame]],
            true,
            u64::max_value(),
        )?;

        let image_index = self
            .device
            .acquire_next_image_khr(
                self.data.swapchain,
                u64::max_value(),
                self.data.image_available_semaphores[self.frame],
                vk::Fence::null(),
            )?
            .0 as usize;

        if !self.data.images_in_flight[image_index as usize].is_null() {
            self.device.wait_for_fences(
                &[self.data.images_in_flight[image_index as usize]],
                true,
                u64::max_value(),
            )?;
        }

        self.data.images_in_flight[image_index as usize] = self.data.in_flight_fences[self.frame];

        let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index as usize]];
        let signal_semaphores = &[self.data.render_finished_sempahores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores)
            .wait_dst_stage_mask(wait_stages);

        self.device
            .reset_fences(&[self.data.in_flight_fences[self.frame]])?;

        self.device.queue_submit(
            self.data.graphics_queue,
            &[submit_info],
            self.data.in_flight_fences[self.frame],
        )?;

        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        self.device
            .queue_present_khr(self.data.present_queue, &present_info)?;

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    pub unsafe fn destroy(&mut self) {
        self.data
            .in_flight_fences
            .iter()
            .for_each(|f| self.device.destroy_fence(*f, None));

        self.data
            .render_finished_sempahores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));

        self.data
            .image_available_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));

        self.device
            .destroy_command_pool(self.data.command_pool, None);

        self.data
            .framebuffers
            .iter()
            .for_each(|f| self.device.destroy_framebuffer(*f, None));

        self.device.destroy_pipeline(self.data.pipeline, None);

        self.device
            .destroy_pipeline_layout(self.data.pipeline_layout, None);

        self.device.destroy_render_pass(self.data.render_pass, None);

        self.data
            .swapchain_image_views
            .iter()
            .for_each(|image_view| self.device.destroy_image_view(*image_view, None));

        self.device.destroy_swapchain_khr(self.data.swapchain, None);
        self.device.destroy_device(None);

        if VALIDATION_ENABLED {
            self.instance
                .destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }

        self.instance.destroy_surface_khr(self.data.surface, None);
        self.instance.destroy_instance(None);
    }

    pub fn run() -> Result<()> {
        pretty_env_logger::init();

        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title("Vulkan Tutorial (Rust)")
            .with_inner_size(LogicalSize::new(1024, 768))
            .build(&event_loop)?;

        let mut app = unsafe { App::new(&window)? };
        let mut destroying = false;
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            match event {
                Event::MainEventsCleared if !destroying => {
                    unsafe { app.render(&window) }.unwrap();
                }
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    destroying = true;
                    *control_flow = ControlFlow::Exit;
                    unsafe {
                        app.device.device_wait_idle().unwrap();
                    }
                    unsafe {
                        app.destroy();
                    }
                }
                _ => {}
            }
        })
    }

    unsafe fn create_instance(
        window: &Window,
        entry: &Entry,
        data: &mut AppData,
    ) -> Result<Instance> {
        let application_info = vk::ApplicationInfo::builder()
            .application_name(b"Vulkan Tutorial\0")
            .application_version(vk::make_version(1, 3, 0))
            .engine_name(b"No Engine\0")
            .engine_version(vk::make_version(1, 3, 0))
            .api_version(vk::make_version(1, 3, 0));

        let available_layers = entry
            .enumerate_instance_layer_properties()?
            .iter()
            .map(|l| l.layer_name)
            .collect::<HashSet<_>>();

        if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
            return Err(anyhow!("Validation layer requested but not supported."));
        }

        let layers = if VALIDATION_ENABLED {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            Vec::new()
        };

        let mut extensions = vk_window::get_required_instance_extensions(window)
            .iter()
            .map(|e| e.as_ptr())
            .collect::<Vec<_>>();

        if VALIDATION_ENABLED {
            extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
        }

        let mut info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);

        let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
            .user_callback(Some(debug_callback));

        if VALIDATION_ENABLED {
            info = info.push_next(&mut debug_info);
        }

        let instance = entry.create_instance(&info, None)?;

        if VALIDATION_ENABLED {
            data.messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
        }

        Ok(instance)
    }
}

extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({:?}) {}", type_, message);
    } else {
        trace!("({:?}) {}", type_, message);
    }

    vk::FALSE
}

#[derive(Clone, Debug, Default)]
pub struct AppData {
    surface: vk::SurfaceKHR,
    messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_sempahores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,
}

unsafe fn create_sync_objects(device: &Device, data: &mut AppData) -> Result<()> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        data.image_available_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);
        data.render_finished_sempahores
            .push(device.create_semaphore(&semaphore_info, None)?);
        data.in_flight_fences
            .push(device.create_fence(&fence_info, None)?);
    }

    data.images_in_flight = data
        .swapchain_images
        .iter()
        .map(|_| vk::Fence::null())
        .collect();

    Ok(())
}
