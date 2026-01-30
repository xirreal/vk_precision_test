use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;
use vulkano::{
    VulkanLibrary,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, QueueCreateInfo, QueueFlags,
        physical::PhysicalDeviceType,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    sync::{self, GpuFuture},
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PushConsts {
    base_low: u32,
    base_high: u32,
}

const BATCH_EXP: u32 = 24;
const BATCH_SIZE: usize = 1 << BATCH_EXP;

fn format_duration(seconds: f64) -> String {
    let total_seconds = seconds as u64;
    let days = total_seconds / 86400;
    let hours = (total_seconds % 86400) / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let secs = total_seconds % 60;

    if days > 0 {
        format!("{}d {}h {}m", days, hours, minutes)
    } else if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, secs)
    } else {
        format!("{}m {}s", minutes, secs)
    }
}

fn ulp_diff(a: f64, b: f64) -> i64 {
    let mut ai = a.to_bits() as i64;
    let mut bi = b.to_bits() as i64;

    if ai < 0 {
        ai = i64::MIN - ai;
    }
    if bi < 0 {
        bi = i64::MIN - bi;
    }

    (ai - bi).abs()
}

struct ThreadContext {
    device: Arc<Device>,
    queue: Arc<vulkano::device::Queue>,
    pipeline: Arc<ComputePipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set: Arc<DescriptorSet>,
    output_buffer: Subbuffer<[f64]>,
}

impl ThreadContext {
    fn new(
        device: Arc<Device>,
        queue: Arc<vulkano::device::Queue>,
        pipeline: Arc<ComputePipeline>,
        mem_alloc: Arc<StandardMemoryAllocator>,
        ds_alloc: Arc<StandardDescriptorSetAllocator>,
        cb_alloc: Arc<StandardCommandBufferAllocator>,
    ) -> Self {
        let output_buffer = Buffer::from_iter(
            mem_alloc,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST // it only makes sense because we are much faster at writing than reading!
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (0..BATCH_SIZE).map(|_| 0.0f64),
        )
        .unwrap();

        let descriptor_set = DescriptorSet::new(
            ds_alloc,
            pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::buffer(0, output_buffer.clone())],
            [],
        )
        .unwrap();

        Self {
            device,
            queue,
            pipeline,
            command_buffer_allocator: cb_alloc,
            descriptor_set,
            output_buffer,
        }
    }

    fn process_batch(&mut self, base_bits: u64, end_limit_bits: u64) -> i64 {
        let push = PushConsts {
            base_low: base_bits as u32,
            base_high: (base_bits >> 32) as u32,
        };

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        unsafe {
            builder
                .bind_pipeline_compute(self.pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.pipeline.layout().clone(),
                    0,
                    self.descriptor_set.clone(),
                )
                .unwrap()
                .push_constants(self.pipeline.layout().clone(), 0, push)
                .unwrap()
                .dispatch([(BATCH_SIZE as u32 / 64), 1, 1])
                .unwrap();
        }

        let cmd = builder.build().unwrap();

        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), cmd)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let mut max_ulp = 0;
        let data = self.output_buffer.read().unwrap();

        for i in 0..BATCH_SIZE {
            let bits = base_bits.wrapping_add(i as u64);
            if bits >= end_limit_bits {
                break;
            }

            let denom = f64::from_bits(bits);
            let cpu = 1.0 / denom;
            let gpu = data[i];

            if cpu.to_bits() != gpu.to_bits() {
                let ulp = ulp_diff(cpu, gpu);
                if ulp > 0 {
                    max_ulp = i64::max(max_ulp, ulp);
                }
            }
        }
        max_ulp
    }
}

fn main() {
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .unwrap();

    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.contains(QueueFlags::COMPUTE))
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            _ => 2,
        })
        .unwrap();

    println!("Using device: {}", physical_device.properties().device_name);

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_features: DeviceFeatures {
                shader_float64: true,
                ..Default::default()
            },
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let pipeline = {
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                src: r#"
#version 450
#extension GL_ARB_gpu_shader_fp64 : enable
layout(local_size_x = 64) in;

layout(set = 0, binding = 0) writeonly buffer Output {
    double outData[];
};

layout(push_constant) uniform PushConsts {
    uint base_low;
    uint base_high;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint res_low = pc.base_low + idx;
    uint carry   = (res_low < pc.base_low) ? 1u : 0u;
    uint res_high = pc.base_high + carry;

    double val = packDouble2x32(uvec2(res_low, res_high));
    outData[idx] = 1.0 / val;
}
"#
            }
        }

        let cs = cs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let stage = PipelineShaderStageCreateInfo::new(cs);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .unwrap()
    };

    let mem_alloc_shared = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let ds_alloc_shared = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let cb_alloc_shared = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));

    let start_bits: u64 = 1.0f64.to_bits();
    let end_bits: u64 = 2.0f64.to_bits();
    let total_elements = end_bits - start_bits;

    let step_stride = BATCH_SIZE as u64;
    let total_batches = (total_elements + step_stride - 1) / step_stride;

    println!("Start Bits: {:#018x}", start_bits);
    println!("End Bits:   {:#018x}", end_bits);
    println!(
        "Batch Size: {} elements ({:.1} MB)",
        BATCH_SIZE,
        (BATCH_SIZE * 8) as f64 / 1024.0 / 1024.0
    );
    println!("Total Batches to process: {}", total_batches);

    let start_time = Instant::now();
    let processed_batches = Arc::new(AtomicU64::new(0));

    let global_max_ulp = (start_bits..end_bits)
        .step_by(BATCH_SIZE)
        .par_bridge()
        .map_init(
            || {
                ThreadContext::new(
                    device.clone(),
                    queue.clone(),
                    pipeline.clone(),
                    mem_alloc_shared.clone(),
                    ds_alloc_shared.clone(),
                    cb_alloc_shared.clone(),
                )
            },
            |ctx, base_idx| {
                let max_ulp_local = ctx.process_batch(base_idx, end_bits);

                let count = processed_batches.fetch_add(1, Ordering::Relaxed) + 1;

                if count % 100 == 0 {
                    let elapsed_secs = start_time.elapsed().as_secs_f64();

                    let batches_remaining = total_batches.saturating_sub(count);
                    let seconds_per_batch = elapsed_secs / count as f64;
                    let eta_seconds = seconds_per_batch * batches_remaining as f64;

                    let percent = (count as f64 / total_batches as f64) * 100.0;

                    println!(
                        "[{:.2}%] Batch {}/{} | ETA: {} | Max ULP (local): {}",
                        percent,
                        count,
                        total_batches,
                        format_duration(eta_seconds),
                        max_ulp_local
                    );
                }

                max_ulp_local
            },
        )
        .reduce(|| 0, |a, b| a.max(b));

    println!("Done! Max ULP difference found: {}", global_max_ulp);
}
