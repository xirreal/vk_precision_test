use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::io::Write;
use std::process::Command;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

const MAX_WORST_CASES: usize = 10;

#[derive(Clone, Debug)]
struct WorstCase {
    input_bits: u64,
    cpu_bits: u64,
    gpu_bits: u64,
    ulp: i64,
    is_f32: bool,
}

impl WorstCase {
    fn input_value(&self) -> String {
        if self.is_f32 {
            format!("{:e}", f32::from_bits(self.input_bits as u32))
        } else {
            format!("{:e}", f64::from_bits(self.input_bits))
        }
    }

    fn cpu_value(&self) -> String {
        if self.is_f32 {
            format!("{:e}", f32::from_bits(self.cpu_bits as u32))
        } else {
            format!("{:e}", f64::from_bits(self.cpu_bits))
        }
    }

    fn gpu_value(&self) -> String {
        if self.is_f32 {
            format!("{:e}", f32::from_bits(self.gpu_bits as u32))
        } else {
            format!("{:e}", f64::from_bits(self.gpu_bits))
        }
    }
}

impl PartialEq for WorstCase {
    fn eq(&self, other: &Self) -> bool {
        self.ulp == other.ulp
    }
}

impl Eq for WorstCase {}

impl PartialOrd for WorstCase {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorstCase {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.ulp.cmp(&other.ulp)
    }
}

struct WorstCaseTracker {
    cases: Mutex<BinaryHeap<std::cmp::Reverse<WorstCase>>>,
    min_ulp: AtomicU64,
}

impl WorstCaseTracker {
    fn new() -> Self {
        Self {
            cases: Mutex::new(BinaryHeap::new()),
            min_ulp: AtomicU64::new(0),
        }
    }

    fn try_add(&self, case: WorstCase) {
        let current_min = self.min_ulp.load(Ordering::Relaxed);
        if case.ulp <= current_min as i64 {
            return;
        }

        let mut cases = self.cases.lock().unwrap();
        cases.push(std::cmp::Reverse(case));

        while cases.len() > MAX_WORST_CASES {
            cases.pop();
        }

        if let Some(std::cmp::Reverse(min_case)) = cases.peek() {
            self.min_ulp.store(min_case.ulp as u64, Ordering::Relaxed);
        }
    }

    fn get_sorted(&self) -> Vec<WorstCase> {
        let cases = self.cases.lock().unwrap();
        let mut result: Vec<_> = cases.iter().map(|r| r.0.clone()).collect();
        result.sort_by(|a, b| b.ulp.cmp(&a.ulp));
        result
    }
}

struct BatchResult {
    max_ulp: i64,
    mismatch_count: u64,
}
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

const BATCH_MAX: u64 = 1 << 25;
const WORKGROUP_SIZE: u32 = 64;

#[derive(Clone, Copy, PartialEq)]
enum Precision {
    Fp32,
    Fp64,
}

#[derive(Clone, Copy, PartialEq)]
enum Operation {
    Reciprocal,
    Sqrt,
    Rsqrt,
    Sin,
    Cos,
}

#[derive(Clone, Default)]
struct FloatOptions {
    precise: bool,
    no_opt: bool,
    nan_clamp: bool,
    dump_spirv: bool,
    denorm_preserve: bool,
    signed_zero_inf_nan_preserve: bool,
    rounding_rte: bool,
    rounding_rtz: bool,
}

impl Operation {
    fn name(&self) -> &'static str {
        match self {
            Operation::Reciprocal => "reciprocal",
            Operation::Sqrt => "sqrt",
            Operation::Rsqrt => "rsqrt",
            Operation::Sin => "sin",
            Operation::Cos => "cos",
        }
    }

    fn glsl_expr_f32(&self, precise: bool) -> String {
        let precise_kw = if precise { "precise " } else { "" };
        match self {
            Operation::Reciprocal => format!("{}float result = 1.0 / val;", precise_kw),
            Operation::Sqrt => format!("{}float result = sqrt(val);", precise_kw),
            Operation::Rsqrt => format!("{}float result = 1.0 / sqrt(val);", precise_kw),
            Operation::Sin => format!("{}float result = sin(val);", precise_kw),
            Operation::Cos => format!("{}float result = cos(val);", precise_kw),
        }
    }

    fn glsl_expr_f64(&self, precise: bool) -> String {
        let precise_kw = if precise { "precise " } else { "" };
        match self {
            Operation::Reciprocal => format!("{}double result = 1.0lf / val;", precise_kw),
            Operation::Sqrt => format!("{}double result = sqrt(val);", precise_kw),
            Operation::Rsqrt => format!("{}double result = 1.0lf / sqrt(val);", precise_kw),
            Operation::Sin => format!("{}double result = sin(val);", precise_kw),
            Operation::Cos => format!("{}double result = cos(val);", precise_kw),
        }
    }

    fn cpu_op_f32(&self, x: f32) -> f32 {
        match self {
            Operation::Reciprocal => 1.0 / x,
            Operation::Sqrt => x.sqrt(),
            Operation::Rsqrt => 1.0 / x.sqrt(),
            Operation::Sin => x.sin(),
            Operation::Cos => x.cos(),
        }
    }

    fn cpu_op_f64(&self, x: f64) -> f64 {
        match self {
            Operation::Reciprocal => 1.0 / x,
            Operation::Sqrt => x.sqrt(),
            Operation::Rsqrt => 1.0 / x.sqrt(),
            Operation::Sin => x.sin(),
            Operation::Cos => x.cos(),
        }
    }
}

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

fn ulp_diff_f64(a: f64, b: f64) -> i64 {
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

fn ulp_diff_f32(a: f32, b: f32) -> i32 {
    let mut ai = a.to_bits() as i32;
    let mut bi = b.to_bits() as i32;

    if ai < 0 {
        ai = i32::MIN - ai;
    }
    if bi < 0 {
        bi = i32::MIN - bi;
    }

    (ai - bi).abs()
}

enum OutputBuffer {
    F32(Subbuffer<[f32]>),
    F64(Subbuffer<[f64]>),
}

struct ThreadContext {
    device: Arc<Device>,
    queue: Arc<vulkano::device::Queue>,
    pipeline: Arc<ComputePipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set: Arc<DescriptorSet>,
    output_buffer: OutputBuffer,
    batch_size: usize,
    operation: Operation,
    worst_tracker: Arc<WorstCaseTracker>,
}

impl ThreadContext {
    fn new_f32(
        device: Arc<Device>,
        queue: Arc<vulkano::device::Queue>,
        pipeline: Arc<ComputePipeline>,
        mem_alloc: Arc<StandardMemoryAllocator>,
        ds_alloc: Arc<StandardDescriptorSetAllocator>,
        cb_alloc: Arc<StandardCommandBufferAllocator>,
        batch_size: usize,
        operation: Operation,
        worst_tracker: Arc<WorstCaseTracker>,
    ) -> Self {
        let output_buffer = Buffer::from_iter(
            mem_alloc,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (0..batch_size).map(|_| 0.0f32),
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
            output_buffer: OutputBuffer::F32(output_buffer),
            batch_size,
            operation,
            worst_tracker,
        }
    }

    fn new_f64(
        device: Arc<Device>,
        queue: Arc<vulkano::device::Queue>,
        pipeline: Arc<ComputePipeline>,
        mem_alloc: Arc<StandardMemoryAllocator>,
        ds_alloc: Arc<StandardDescriptorSetAllocator>,
        cb_alloc: Arc<StandardCommandBufferAllocator>,
        batch_size: usize,
        operation: Operation,
        worst_tracker: Arc<WorstCaseTracker>,
    ) -> Self {
        let output_buffer = Buffer::from_iter(
            mem_alloc,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (0..batch_size).map(|_| 0.0f64),
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
            output_buffer: OutputBuffer::F64(output_buffer),
            batch_size,
            operation,
            worst_tracker,
        }
    }

    fn process_batch(&mut self, base_bits: u64, end_limit_bits: u64) -> BatchResult {
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

        let dispatch_x = (self.batch_size as u32) / WORKGROUP_SIZE;

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
                .dispatch([dispatch_x, 1, 1])
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

        match &self.output_buffer {
            OutputBuffer::F32(buf) => {
                let data = buf.read().unwrap();
                let mut max_ulp: i64 = 0;
                let mut mismatch_count: u64 = 0;

                for i in 0..self.batch_size {
                    let bits = (base_bits as u32).wrapping_add(i as u32);
                    if (bits as u64) >= end_limit_bits {
                        break;
                    }

                    let input = f32::from_bits(bits);
                    let cpu = self.operation.cpu_op_f32(input);
                    let gpu = data[i];

                    if cpu.to_bits() != gpu.to_bits() {
                        let ulp = ulp_diff_f32(cpu, gpu);
                        if ulp > 0 {
                            mismatch_count += 1;
                            let ulp_i64 = ulp as i64;
                            max_ulp = i64::max(max_ulp, ulp_i64);
                            self.worst_tracker.try_add(WorstCase {
                                input_bits: bits as u64,
                                cpu_bits: cpu.to_bits() as u64,
                                gpu_bits: gpu.to_bits() as u64,
                                ulp: ulp_i64,
                                is_f32: true,
                            });
                        }
                    }
                }
                BatchResult {
                    max_ulp,
                    mismatch_count,
                }
            }
            OutputBuffer::F64(buf) => {
                let data = buf.read().unwrap();
                let mut max_ulp: i64 = 0;
                let mut mismatch_count: u64 = 0;

                for i in 0..self.batch_size {
                    let bits = base_bits.wrapping_add(i as u64);
                    if bits >= end_limit_bits {
                        break;
                    }

                    let input = f64::from_bits(bits);
                    let cpu = self.operation.cpu_op_f64(input);
                    let gpu = data[i];

                    if cpu.to_bits() != gpu.to_bits() {
                        let ulp = ulp_diff_f64(cpu, gpu);
                        if ulp > 0 {
                            mismatch_count += 1;
                            max_ulp = i64::max(max_ulp, ulp);
                            self.worst_tracker.try_add(WorstCase {
                                input_bits: bits,
                                cpu_bits: cpu.to_bits(),
                                gpu_bits: gpu.to_bits(),
                                ulp,
                                is_f32: false,
                            });
                        }
                    }
                }
                BatchResult {
                    max_ulp,
                    mismatch_count,
                }
            }
        }
    }
}

fn patch_spirv_execution_modes(spirv: &[u8], opts: &FloatOptions, bit_width: u32) -> Vec<u8> {
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    if words.len() < 5 {
        return spirv.to_vec();
    }

    let mut patched_words = words.clone();

    let mut entry_point_id: Option<u32> = None;
    let mut insert_pos: Option<usize> = None;

    let mut i = 5;
    while i < words.len() {
        let word = words[i];
        let opcode = word & 0xFFFF;
        let word_count = (word >> 16) as usize;

        if word_count == 0 {
            break;
        }

        match opcode {
            17 => {
                if i + 2 < words.len() {
                    entry_point_id = Some(words[i + 2]);
                }
            }
            16 => {
                insert_pos = Some(i + word_count);
            }
            _ => {}
        }

        if opcode > 16 && insert_pos.is_none() {
            insert_pos = Some(i);
        }

        i += word_count;
    }

    if let (Some(entry_id), Some(pos)) = (entry_point_id, insert_pos) {
        let mut new_modes: Vec<u32> = Vec::new();

        if opts.denorm_preserve {
            new_modes.push((4 << 16) | 16);
            new_modes.push(entry_id);
            new_modes.push(4459);
            new_modes.push(bit_width);
        }

        if opts.signed_zero_inf_nan_preserve {
            new_modes.push((4 << 16) | 16);
            new_modes.push(entry_id);
            new_modes.push(4461);
            new_modes.push(bit_width);
        }

        if opts.rounding_rte {
            new_modes.push((4 << 16) | 16);
            new_modes.push(entry_id);
            new_modes.push(4462);
            new_modes.push(bit_width);
        }

        if opts.rounding_rtz {
            new_modes.push((4 << 16) | 16);
            new_modes.push(entry_id);
            new_modes.push(4463);
            new_modes.push(bit_width);
        }

        if !new_modes.is_empty() {
            let mut capabilities_to_add: Vec<u32> = Vec::new();

            if opts.denorm_preserve {
                capabilities_to_add.push((2 << 16) | 17);
                capabilities_to_add.push(4464);
            }
            if opts.signed_zero_inf_nan_preserve {
                capabilities_to_add.push((2 << 16) | 17);
                capabilities_to_add.push(4466);
            }
            if opts.rounding_rte {
                capabilities_to_add.push((2 << 16) | 17);
                capabilities_to_add.push(4467);
            }
            if opts.rounding_rtz {
                capabilities_to_add.push((2 << 16) | 17);
                capabilities_to_add.push(4468);
            }

            let cap_insert = 5;
            patched_words.splice(cap_insert..cap_insert, capabilities_to_add.iter().cloned());

            let adjusted_pos = pos + (patched_words.len() - words.len());
            patched_words.splice(adjusted_pos..adjusted_pos, new_modes.iter().cloned());

            patched_words[3] = patched_words
                .iter()
                .skip(5)
                .map(|&w| {
                    let opcode = w & 0xFFFF;
                    if opcode == 19
                        || opcode == 20
                        || opcode == 21
                        || opcode == 22
                        || opcode == 27
                        || opcode == 28
                        || opcode == 29
                        || opcode == 30
                        || opcode == 31
                        || opcode == 32
                        || opcode == 33
                    {
                        (w >> 16) as u32
                    } else {
                        0
                    }
                })
                .max()
                .unwrap_or(words[3])
                .max(words[3]);
        }
    }

    patched_words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn compile_shader_glslc(
    source: &str,
    target_env: &str,
    opts: &FloatOptions,
    bit_width: u32,
) -> Vec<u8> {
    let temp_dir = std::env::temp_dir();
    let src_path = temp_dir.join("shader.comp");
    let spv_path = temp_dir.join("shader.spv");

    let mut src_file = std::fs::File::create(&src_path).unwrap();
    src_file.write_all(source.as_bytes()).unwrap();
    drop(src_file);

    let mut cmd = Command::new("glslc");
    cmd.arg(format!("--target-env={}", target_env));

    if opts.no_opt {
        cmd.arg("-O0");
    } else {
        cmd.arg("-O");
    }

    if opts.nan_clamp {
        cmd.arg("-fnan-clamp");
    }

    cmd.arg(&src_path).arg("-o").arg(&spv_path);

    let output = cmd
        .output()
        .expect("Failed to run glslc. Make sure it's installed and in PATH.");

    if !output.status.success() {
        eprintln!("glslc stderr: {}", String::from_utf8_lossy(&output.stderr));
        eprintln!("Shader source:\n{}", source);
        panic!("Shader compilation failed");
    }

    let spirv = std::fs::read(&spv_path).unwrap();

    let needs_patching = opts.denorm_preserve
        || opts.signed_zero_inf_nan_preserve
        || opts.rounding_rte
        || opts.rounding_rtz;

    let final_spirv = if needs_patching {
        patch_spirv_execution_modes(&spirv, opts, bit_width)
    } else {
        spirv
    };

    if opts.dump_spirv {
        let asm_output = Command::new("glslc")
            .arg(format!("--target-env={}", target_env))
            .arg("-S")
            .arg(&src_path)
            .arg("-o")
            .arg("-")
            .output();

        if let Ok(asm) = asm_output {
            if needs_patching {
                eprintln!("=== SPIR-V Assembly (before patching) ===");
            } else {
                eprintln!("=== SPIR-V Assembly ===");
            }
            eprintln!("{}", String::from_utf8_lossy(&asm.stdout));
        }

        if needs_patching {
            eprintln!("\n=== Patched execution modes ===");
            if opts.denorm_preserve {
                eprintln!("  OpExecutionMode DenormPreserve {}", bit_width);
            }
            if opts.signed_zero_inf_nan_preserve {
                eprintln!("  OpExecutionMode SignedZeroInfNanPreserve {}", bit_width);
            }
            if opts.rounding_rte {
                eprintln!("  OpExecutionMode RoundingModeRTE {}", bit_width);
            }
            if opts.rounding_rtz {
                eprintln!("  OpExecutionMode RoundingModeRTZ {}", bit_width);
            }
        }
    }

    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&spv_path);

    final_spirv
}

fn create_pipeline_f32(
    device: Arc<Device>,
    operation: Operation,
    opts: &FloatOptions,
) -> Arc<ComputePipeline> {
    let shader_src = format!(
        r#"#version 450
layout(local_size_x = 64) in;

layout(set = 0, binding = 0) writeonly buffer Output {{
    float outData[];
}};

layout(push_constant) uniform PushConsts {{
    uint base_low;
    uint base_high;
}} pc;

void main() {{
    uint idx = gl_GlobalInvocationID.x;
    uint bits = pc.base_low + idx;
    float val = uintBitsToFloat(bits);
    {expr}
    outData[idx] = result;
}}
"#,
        expr = operation.glsl_expr_f32(opts.precise)
    );

    let spirv = compile_shader_glslc(&shader_src, "vulkan1.2", opts, 32);

    let cs = unsafe {
        vulkano::shader::ShaderModule::new(
            device.clone(),
            vulkano::shader::ShaderModuleCreateInfo::new(
                &vulkano::shader::spirv::bytes_to_words(&spirv).unwrap(),
            ),
        )
        .unwrap()
    };

    let entry = cs.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(entry);
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
}

fn create_pipeline_f64(
    device: Arc<Device>,
    operation: Operation,
    opts: &FloatOptions,
) -> Arc<ComputePipeline> {
    let shader_src = format!(
        r#"#version 450
#extension GL_ARB_gpu_shader_fp64 : enable
layout(local_size_x = 64) in;

layout(set = 0, binding = 0) writeonly buffer Output {{
    double outData[];
}};

layout(push_constant) uniform PushConsts {{
    uint base_low;
    uint base_high;
}} pc;

void main() {{
    uint idx = gl_GlobalInvocationID.x;
    uint res_low = pc.base_low + idx;
    uint carry = (res_low < pc.base_low) ? 1u : 0u;
    uint res_high = pc.base_high + carry;

    double val = packDouble2x32(uvec2(res_low, res_high));
    {expr}
    outData[idx] = result;
}}
"#,
        expr = operation.glsl_expr_f64(opts.precise)
    );

    let spirv = compile_shader_glslc(&shader_src, "vulkan1.2", opts, 64);

    let cs = unsafe {
        vulkano::shader::ShaderModule::new(
            device.clone(),
            vulkano::shader::ShaderModuleCreateInfo::new(
                &vulkano::shader::spirv::bytes_to_words(&spirv).unwrap(),
            ),
        )
        .unwrap()
    };

    let entry = cs.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(entry);
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
}

fn print_usage() {
    println!("Usage: vk_precision_test <precision> <operation> [options]");
    println!();
    println!("Precision:");
    println!("  fp32    - Test 32-bit floats");
    println!("  fp64    - Test 64-bit doubles");
    println!();
    println!("Operations:");
    println!("  rcp     - Reciprocal (1.0 / x)");
    println!("  sqrt    - Square root");
    println!("  rsqrt   - Reciprocal square root (1.0 / sqrt(x))");
    println!("  sin     - Sine");
    println!("  cos     - Cosine");
    println!();
    println!("Float control options:");
    println!("  --precise       Use 'precise' qualifier (adds NoContraction SPIR-V decoration)");
    println!("  --no-opt        Disable SPIR-V optimization (-O0)");
    println!("  --nan-clamp     Use NaN-safe min/max/clamp (glslc -fnan-clamp)");
    println!("  --dump-spirv    Print generated SPIR-V assembly");
    println!();
    println!("VK_KHR_shader_float_controls options (requires device support):");
    println!("  --denorm-preserve       Preserve denormalized values (DenormPreserve)");
    println!("  --preserve-nan-inf      Preserve signed zero, inf, NaN (SignedZeroInfNanPreserve)");
    println!("  --rounding-rte          Use round-to-nearest-even (RoundingModeRTE)");
    println!("  --rounding-rtz          Use round-towards-zero (RoundingModeRTZ)");
    println!();
    println!("Range options (fp32 only):");
    println!("  --exp=N         Test only exponent N");
    println!("  --exp=N-M       Test exponents N through M");
    println!("  --exp=all       Test all exponents (default)");
    println!();
    println!("Examples:");
    println!("  vk_precision_test fp32 sqrt");
    println!("  vk_precision_test fp32 rcp --exp=0 --precise");
    println!("  vk_precision_test fp32 sin --exp=-10-10 --precise --no-opt");
    println!("  vk_precision_test fp64 rcp --precise --dump-spirv");
    println!("  vk_precision_test fp32 rcp --denorm-preserve --preserve-nan-inf");
}

fn parse_operation(s: &str) -> Option<Operation> {
    match s.to_lowercase().as_str() {
        "rcp" | "reciprocal" => Some(Operation::Reciprocal),
        "sqrt" => Some(Operation::Sqrt),
        "rsqrt" | "invsqrt" => Some(Operation::Rsqrt),
        "sin" => Some(Operation::Sin),
        "cos" => Some(Operation::Cos),
        _ => None,
    }
}

fn parse_exp_range(s: &str) -> (i32, i32) {
    if s == "all" {
        return (-126, 127);
    }

    if let Some(pos) = s.find('-') {
        if pos == 0 {
            // Starts with '-', could be -N or -N-M
            let rest = &s[1..];
            if let Some(pos2) = rest.find('-') {
                let start: i32 = -rest[..pos2].parse().unwrap_or(126);
                let end: i32 = rest[pos2 + 1..].parse().unwrap_or(127);
                return (start, end);
            } else {
                let exp: i32 = s.parse().unwrap_or(0);
                return (exp, exp);
            }
        } else {
            // Format: N-M
            let start: i32 = s[..pos].parse().unwrap_or(-126);
            let end: i32 = s[pos + 1..].parse().unwrap_or(127);
            return (start, end);
        }
    }

    let exp: i32 = s.parse().unwrap_or(0);
    (exp, exp)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        print_usage();
        return;
    }

    let precision = match args[1].to_lowercase().as_str() {
        "fp32" | "f32" | "float" => Precision::Fp32,
        "fp64" | "f64" | "double" => Precision::Fp64,
        _ => {
            eprintln!("Unknown precision: {}", args[1]);
            print_usage();
            return;
        }
    };

    let operation = match parse_operation(&args[2]) {
        Some(op) => op,
        None => {
            eprintln!("Unknown operation: {}", args[2]);
            print_usage();
            return;
        }
    };

    let mut float_opts = FloatOptions::default();
    let mut exp_range: Option<(i32, i32)> = None;

    for arg in &args[3..] {
        match arg.as_str() {
            "--precise" => float_opts.precise = true,
            "--no-opt" => float_opts.no_opt = true,
            "--nan-clamp" => float_opts.nan_clamp = true,
            "--dump-spirv" => float_opts.dump_spirv = true,
            "--denorm-preserve" => float_opts.denorm_preserve = true,
            "--preserve-nan-inf" => float_opts.signed_zero_inf_nan_preserve = true,
            "--rounding-rte" => float_opts.rounding_rte = true,
            "--rounding-rtz" => float_opts.rounding_rtz = true,
            _ if arg.starts_with("--exp=") => {
                exp_range = Some(parse_exp_range(&arg[6..]));
            }
            _ => {
                eprintln!("Unknown option: {}", arg);
                print_usage();
                return;
            }
        }
    }

    let (exp_start, exp_end) = if precision == Precision::Fp32 {
        exp_range.unwrap_or((-126, 127))
    } else {
        (0, 0)
    };

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

    let properties = physical_device.properties();
    println!("Using device: {}", properties.device_name);

    println!("\nFloat controls support:");
    println!(
        "  Denorm preserve fp32: {}",
        if properties.shader_denorm_preserve_float32.unwrap_or(false) {
            "yes"
        } else {
            "no"
        }
    );
    println!(
        "  Denorm preserve fp64: {}",
        if properties.shader_denorm_preserve_float64.unwrap_or(false) {
            "yes"
        } else {
            "no"
        }
    );
    println!(
        "  Denorm flush-to-zero fp32: {}",
        if properties
            .shader_denorm_flush_to_zero_float32
            .unwrap_or(false)
        {
            "yes"
        } else {
            "no"
        }
    );
    println!(
        "  Signed zero/inf/nan preserve fp32: {}",
        if properties
            .shader_signed_zero_inf_nan_preserve_float32
            .unwrap_or(false)
        {
            "yes"
        } else {
            "no"
        }
    );
    println!(
        "  Rounding mode RTE fp32: {}",
        if properties.shader_rounding_mode_rte_float32.unwrap_or(false) {
            "yes"
        } else {
            "no"
        }
    );

    let denorm_preserve_supported = match precision {
        Precision::Fp32 => properties.shader_denorm_preserve_float32.unwrap_or(false),
        Precision::Fp64 => properties.shader_denorm_preserve_float64.unwrap_or(false),
    };
    let signed_zero_supported = match precision {
        Precision::Fp32 => properties
            .shader_signed_zero_inf_nan_preserve_float32
            .unwrap_or(false),
        Precision::Fp64 => properties
            .shader_signed_zero_inf_nan_preserve_float64
            .unwrap_or(false),
    };
    let rte_supported = match precision {
        Precision::Fp32 => properties.shader_rounding_mode_rte_float32.unwrap_or(false),
        Precision::Fp64 => properties.shader_rounding_mode_rte_float64.unwrap_or(false),
    };
    let rtz_supported = match precision {
        Precision::Fp32 => properties.shader_rounding_mode_rtz_float32.unwrap_or(false),
        Precision::Fp64 => properties.shader_rounding_mode_rtz_float64.unwrap_or(false),
    };

    if float_opts.denorm_preserve && !denorm_preserve_supported {
        eprintln!("WARNING: --denorm-preserve requested but device does not support it!");
        eprintln!("         Shader will fail to load. Disabling option.");
        float_opts.denorm_preserve = false;
    }
    if float_opts.signed_zero_inf_nan_preserve && !signed_zero_supported {
        eprintln!("WARNING: --preserve-nan-inf requested but device does not support it!");
        eprintln!("         Shader will fail to load. Disabling option.");
        float_opts.signed_zero_inf_nan_preserve = false;
    }
    if float_opts.rounding_rte && !rte_supported {
        eprintln!("WARNING: --rounding-rte requested but device does not support it!");
        eprintln!("         Shader will fail to load. Disabling option.");
        float_opts.rounding_rte = false;
    }
    if float_opts.rounding_rtz && !rtz_supported {
        eprintln!("WARNING: --rounding-rtz requested but device does not support it!");
        eprintln!("         Shader will fail to load. Disabling option.");
        float_opts.rounding_rtz = false;
    }

    let mut mode_parts = Vec::new();
    if float_opts.precise {
        mode_parts.push("precise (NoContraction)".to_string());
    }
    if float_opts.no_opt {
        mode_parts.push("no-opt (-O0)".to_string());
    }
    if float_opts.nan_clamp {
        mode_parts.push("nan-clamp".to_string());
    }
    if float_opts.denorm_preserve {
        mode_parts.push("denorm-preserve".to_string());
    }
    if float_opts.signed_zero_inf_nan_preserve {
        mode_parts.push("preserve-nan-inf".to_string());
    }
    if float_opts.rounding_rte {
        mode_parts.push("rounding-rte".to_string());
    }
    if float_opts.rounding_rtz {
        mode_parts.push("rounding-rtz".to_string());
    }
    if mode_parts.is_empty() {
        println!("\nFloat mode: default");
    } else {
        println!("\nFloat mode: {}", mode_parts.join(", "));
    }

    let max_group_count = properties.max_compute_work_group_count[0] as u64;
    let max_dispatch_size = max_group_count * (WORKGROUP_SIZE as u64);
    let batch_size = std::cmp::min(max_dispatch_size, BATCH_MAX) as usize;

    let features = if precision == Precision::Fp64 {
        DeviceFeatures {
            shader_float64: true,
            ..Default::default()
        }
    } else {
        DeviceFeatures::default()
    };

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_features: features,
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

    let pipeline = match precision {
        Precision::Fp32 => create_pipeline_f32(device.clone(), operation, &float_opts),
        Precision::Fp64 => create_pipeline_f64(device.clone(), operation, &float_opts),
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

    match precision {
        Precision::Fp32 => {
            let start_biased_exp = (exp_start + 127) as u32;
            let end_biased_exp = (exp_end + 127) as u32;

            let start_bits: u64 = ((start_biased_exp << 23) & 0x7F800000) as u64;
            let end_bits: u64 = if end_biased_exp >= 254 {
                0x7F800000
            } else {
                (((end_biased_exp + 1) << 23) & 0x7F800000) as u64
            };

            let total_elements = end_bits - start_bits;
            let step_stride = batch_size as u64;
            let total_batches = (total_elements + step_stride - 1) / step_stride;

            println!(
                "Testing {} on fp32, exponents {} to {}",
                operation.name(),
                exp_start,
                exp_end
            );
            println!("Start Bits: {:#010x}", start_bits);
            println!("End Bits:   {:#010x}", end_bits);
            println!(
                "Batch Size: {} elements ({:.1} MB)",
                batch_size,
                (batch_size * 4) as f64 / 1024.0 / 1024.0
            );
            println!(
                "Total: {} values, {} batches",
                total_elements, total_batches
            );

            let start_time = Instant::now();
            let processed_batches = Arc::new(AtomicU64::new(0));
            let worst_tracker = Arc::new(WorstCaseTracker::new());

            let (global_max_ulp, total_mismatches) = (start_bits..end_bits)
                .step_by(batch_size)
                .par_bridge()
                .map_init(
                    || {
                        ThreadContext::new_f32(
                            device.clone(),
                            queue.clone(),
                            pipeline.clone(),
                            mem_alloc_shared.clone(),
                            ds_alloc_shared.clone(),
                            cb_alloc_shared.clone(),
                            batch_size,
                            operation,
                            worst_tracker.clone(),
                        )
                    },
                    |ctx, base_idx| {
                        let result = ctx.process_batch(base_idx, end_bits);

                        let count = processed_batches.fetch_add(1, Ordering::Relaxed) + 1;

                        if count % 100 == 0 {
                            let elapsed_secs = start_time.elapsed().as_secs_f64();
                            let batches_remaining = total_batches.saturating_sub(count);
                            let seconds_per_batch = elapsed_secs / count as f64;
                            let eta_seconds = seconds_per_batch * batches_remaining as f64;
                            let percent = (count as f64 / total_batches as f64) * 100.0;

                            println!(
                                "[{:.2}%] Batch {}/{} | ETA: {}",
                                percent,
                                count,
                                total_batches,
                                format_duration(eta_seconds)
                            );
                        }

                        result
                    },
                )
                .map(|r| (r.max_ulp, r.mismatch_count))
                .reduce(
                    || (0, 0),
                    |(a_ulp, a_cnt), (b_ulp, b_cnt)| (a_ulp.max(b_ulp), a_cnt + b_cnt),
                );

            println!("\nDone! Max ULP difference: {}", global_max_ulp);
            print_worst_cases(&worst_tracker, total_elements, total_mismatches);
        }
        Precision::Fp64 => {
            let start_bits: u64 = 1.0f64.to_bits();
            let end_bits: u64 = 2.0f64.to_bits();
            let total_elements = end_bits - start_bits;
            let step_stride = batch_size as u64;
            let total_batches = (total_elements + step_stride - 1) / step_stride;

            println!("Testing {} on fp64, range [1.0, 2.0)", operation.name());
            println!("Start Bits: {:#018x}", start_bits);
            println!("End Bits:   {:#018x}", end_bits);
            println!(
                "Batch Size: {} elements ({:.1} MB)",
                batch_size,
                (batch_size * 8) as f64 / 1024.0 / 1024.0
            );
            println!("Total Batches: {}", total_batches);

            let start_time = Instant::now();
            let processed_batches = Arc::new(AtomicU64::new(0));
            let worst_tracker = Arc::new(WorstCaseTracker::new());

            let (global_max_ulp, total_mismatches) = (start_bits..end_bits)
                .step_by(batch_size)
                .par_bridge()
                .map_init(
                    || {
                        ThreadContext::new_f64(
                            device.clone(),
                            queue.clone(),
                            pipeline.clone(),
                            mem_alloc_shared.clone(),
                            ds_alloc_shared.clone(),
                            cb_alloc_shared.clone(),
                            batch_size,
                            operation,
                            worst_tracker.clone(),
                        )
                    },
                    |ctx, base_idx| {
                        let result = ctx.process_batch(base_idx, end_bits);

                        let count = processed_batches.fetch_add(1, Ordering::Relaxed) + 1;

                        if count % 100 == 0 {
                            let elapsed_secs = start_time.elapsed().as_secs_f64();
                            let batches_remaining = total_batches.saturating_sub(count);
                            let seconds_per_batch = elapsed_secs / count as f64;
                            let eta_seconds = seconds_per_batch * batches_remaining as f64;
                            let percent = (count as f64 / total_batches as f64) * 100.0;

                            println!(
                                "[{:.2}%] Batch {}/{} | ETA: {}",
                                percent,
                                count,
                                total_batches,
                                format_duration(eta_seconds),
                            );
                        }

                        result
                    },
                )
                .map(|r| (r.max_ulp, r.mismatch_count))
                .reduce(
                    || (0, 0),
                    |(a_ulp, a_cnt), (b_ulp, b_cnt)| (a_ulp.max(b_ulp), a_cnt + b_cnt),
                );

            println!("Done! Max ULP difference: {}", global_max_ulp);
            print_worst_cases(&worst_tracker, total_elements, total_mismatches);
        }
    }
}

fn print_worst_cases(tracker: &WorstCaseTracker, total_values: u64, total_mismatches: u64) {
    let cases = tracker.get_sorted();

    let mismatch_pct = (total_mismatches as f64 / total_values as f64) * 100.0;
    println!(
        "Total mismatches: {} / {} ({:.6}%)",
        total_mismatches, total_values, mismatch_pct
    );

    if cases.is_empty() {
        return;
    }

    println!("\nTop {} worst cases:", cases.len());
    println!(
        "{:>4} | {:>16} | {:>24} | {:>24} | {:>24}",
        "#", "ULP", "Input", "CPU Result", "GPU Result"
    );
    println!("{}", "-".repeat(104));

    for (i, case) in cases.iter().enumerate() {
        let input_hex = if case.is_f32 {
            format!("{:#010x}", case.input_bits as u32)
        } else {
            format!("{:#018x}", case.input_bits)
        };

        println!(
            "{:>4} | {:>16} | {:>24} | {:>24} | {:>24}",
            i + 1,
            case.ulp,
            format!("{} ({})", case.input_value(), input_hex),
            case.cpu_value(),
            case.gpu_value(),
        );
    }
}
