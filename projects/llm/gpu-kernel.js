export class GPUKernel {
  constructor() {
    this.adapter = null;
    this.device = null;
    this.pipeline = null;
    this.bindGroupLayout = null;
    this.isSupported = false;
  }

  async init() {
    if (!navigator.gpu) {
      this.isSupported = false;
      return false;
    }

    try {
      this.adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!this.adapter) {
        this.isSupported = false;
        return false;
      }
      this.device = await this.adapter.requestDevice();
      this.isSupported = true;
      await this.initShaders();
      return true;
    } catch (err) {
      this.isSupported = false;
      return false;
    }
  }

  async initShaders() {
    const shaderModule = this.device.createShaderModule({
      label: 'Q4_K_M_GEMV',
      code: `
        struct Uniforms {
          inDim: u32,
          outDim: u32,
        };

        @group(0) @binding(0) var<uniform> u: Uniforms;
        @group(0) @binding(1) var<storage, read> x: array<f32>;
        @group(0) @binding(2) var<storage, read> weights: array<u32>;
        @group(0) @binding(3) var<storage, read_write> output: array<f32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          let row = id.x;
          if (row >= u.outDim) { return; }

          let inDim = u.inDim;
          let rowBytes = ((inDim * 9u + 15u) / 16u) + 32u;
          let rowWordOffset = (row * rowBytes) / 4u;

          var sum: f32 = 0.0;
          let blocks = (inDim + 255u) / 256u;
          var xIdx: u32 = 0u;
          var byteOff: u32 = 0u;

          for (var b: u32 = 0u; b < blocks && xIdx < inDim; b = b + 1u) {
            let headerWord = weights[rowWordOffset + (byteOff / 4u)];
            let raw_d = f32(headerWord & 0xFFFFu);
            let raw_dmin = f32((headerWord >> 16u) & 0xFFFFu);
            let d = raw_d / 65535.0;
            let dmin = raw_dmin / 65535.0;
            byteOff = byteOff + 4u;

            var bytesToRead = (inDim - xIdx) / 2u;
            if (bytesToRead > 128u) { bytesToRead = 128u; }

            for (var i: u32 = 0u; i < bytesToRead; i = i + 1u) {
              let wIdx = rowWordOffset + ((byteOff + i) / 4u);
              let shift = ((byteOff + i) % 4u) * 8u;
              let packedByte = (weights[wIdx] >> shift) & 0xFFu;

              let w0 = f32(packedByte & 0x0Fu) * d - dmin;
              let w1 = f32((packedByte >> 4u) & 0x0Fu) * d - dmin;

              sum = sum + x[xIdx] * w0;
              xIdx = xIdx + 1u;
              if (xIdx < inDim) {
                sum = sum + x[xIdx] * w1;
                xIdx = xIdx + 1u;
              }
            }
            byteOff = byteOff + 128u;
          }

          output[row] = sum;
        }
      `
    });

    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ]
    });

    this.pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: 'main' }
    });
  }

  createBuffer(data, usage) {
    const buffer = this.device.createBuffer({
      size: Math.ceil(data.byteLength / 4) * 4,
      usage,
      mappedAtCreation: true
    });
    if (data instanceof Float32Array) {
      new Float32Array(buffer.getMappedRange()).set(data);
    } else if (data instanceof Uint8Array) {
      new Uint8Array(buffer.getMappedRange()).set(data);
    } else if (data instanceof Uint32Array) {
      new Uint32Array(buffer.getMappedRange()).set(data);
    }
    buffer.unmap();
    return buffer;
  }

  async runMatVec(xArray, weightsArray, outDim, inDim) {
    const uniforms = new Uint32Array([inDim, outDim]);
    const uBuffer = this.createBuffer(uniforms, GPUBufferUsage.UNIFORM);
    const xBuffer = this.createBuffer(xArray, GPUBufferUsage.STORAGE);
    const wBuffer = this.createBuffer(weightsArray, GPUBufferUsage.STORAGE);
    const outBuffer = this.device.createBuffer({
      size: outDim * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    const readBuffer = this.device.createBuffer({
      size: outDim * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uBuffer } },
        { binding: 1, resource: { buffer: xBuffer } },
        { binding: 2, resource: { buffer: wBuffer } },
        { binding: 3, resource: { buffer: outBuffer } }
      ]
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(outDim / 64));
    pass.end();

    encoder.copyBufferToBuffer(outBuffer, 0, readBuffer, 0, outDim * 4);
    this.device.queue.submit([encoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange()).slice();
    readBuffer.unmap();

    uBuffer.destroy();
    xBuffer.destroy();
    wBuffer.destroy();
    outBuffer.destroy();
    readBuffer.destroy();

    return result;
  }
}

