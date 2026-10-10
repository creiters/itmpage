export class GGUFParser {
  static GGUF_MAGIC = 0x46554747; // 'GGUF' in Little Endian

  constructor(arrayBuffer) {
    this.buffer = arrayBuffer;
    this.view = new DataView(arrayBuffer);
    this.offset = 0;
    this.metadata = {};
    this.tensors = new Map();
  }

  readUint32() {
    const val = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return val;
  }

  readUint64() {
    const val = this.view.getBigUint64(this.offset, true);
    this.offset += 8;
    return Number(val);
  }

  readString() {
    const len = this.readUint64();
    const bytes = new Uint8Array(this.buffer, this.offset, len);
    this.offset += len;
    return new TextDecoder('utf-8').decode(bytes);
  }

  readMetadataValue(type) {
    switch (type) {
      case 0: return this.view.getUint8(this.offset++);
      case 1: { const v = this.view.getInt8(this.offset); this.offset += 1; return v; }
      case 2: { const v = this.view.getUint16(this.offset, true); this.offset += 2; return v; }
      case 3: { const v = this.view.getInt16(this.offset, true); this.offset += 2; return v; }
      case 4: { const v = this.view.getUint32(this.offset, true); this.offset += 4; return v; }
      case 5: { const v = this.view.getInt32(this.offset, true); this.offset += 4; return v; }
      case 6: { const v = this.view.getFloat32(this.offset, true); this.offset += 4; return v; }
      case 7: { return this.view.getUint8(this.offset++) !== 0; }
      case 8: return this.readString();
      case 9: { // Array
        const itemType = this.readUint32();
        const len = this.readUint64();
        const arr = [];
        for (let i = 0; i < len; i++) {
          arr.push(this.readMetadataValue(itemType));
        }
        return arr;
      }
      case 10: return this.readUint64();
      case 11: { const v = this.view.getInt64(this.offset, true); this.offset += 8; return Number(v); }
      case 12: { const v = this.view.getFloat64(this.offset, true); this.offset += 8; return v; }
      default: throw new Error(`Unknown GGUF metadata type: ${type}`);
    }
  }

  parse() {
    const magic = this.readUint32();
    if (magic !== GGUFParser.GGUF_MAGIC) {
      throw new Error('Not a valid GGUF binary (invalid magic header).');
    }

    const version = this.readUint32();
    const tensorCount = this.readUint64();
    const metadataCount = this.readUint64();

    // 1. Parse Metadata Key-Value pairs
    for (let i = 0; i < metadataCount; i++) {
      const key = this.readString();
      const valType = this.readUint32();
      this.metadata[key] = this.readMetadataValue(valType);
    }

    // 2. Parse Tensor Information Table
    for (let i = 0; i < tensorCount; i++) {
      const name = this.readString();
      const nDims = this.readUint32();
      const dims = [];
      for (let d = 0; d < nDims; d++) {
        dims.push(this.readUint64());
      }
      const type = this.readUint32();
      const offset = this.readUint64();

      this.tensors.set(name, { dims, type, offset });
    }

    // Tensor payload starts aligned to metadata alignment (typically 32 bytes)
    const alignment = this.metadata['general.alignment'] || 32;
    const padding = (alignment - (this.offset % alignment)) % alignment;
    const tensorDataStart = this.offset + padding;

    return {
      version,
      metadata: this.metadata,
      tensors: this.tensors,
      tensorDataStart
    };
  }
}
