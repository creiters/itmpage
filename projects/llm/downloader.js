export class HuggingFaceDownloader {
  static MODEL_MAP = {
    'smollm2-360m': {
      title: 'SmolLM2-360M (~245 MB, ~245 Chunks)',
      repo: 'HuggingFaceTB/SmolLM2-360M-Instruct-GGUF',
      filename: 'smollm2-360m-instruct-q4_k_m.gguf',
      approxBytes: 245000000
    },
    'llama-3.2-1b': {
      title: 'Llama-3.2-1B (~780 MB, ~780 Chunks)',
      repo: 'bartowski/Llama-3.2-1B-Instruct-GGUF',
      filename: 'Llama-3.2-1B-Instruct-Q4_K_M.gguf',
      approxBytes: 780000000
    },
    'qwen2.5-1.5b': {
      title: 'Qwen2.5-1.5B (~980 MB, ~980 Chunks)',
      repo: 'Qwen/Qwen2.5-1.5B-Instruct-GGUF',
      filename: 'qwen2.5-1.5b-instruct-q4_k_m.gguf',
      approxBytes: 980000000
    },
    'smollm2-1.7b': {
      title: 'SmolLM2-1.7B (~1.0 GB, ~1060 Chunks)',
      repo: 'HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF',
      filename: 'smollm2-1.7b-instruct-q4_k_m.gguf',
      approxBytes: 1060000000
    },
    'phi-3.5-mini': {
      title: 'Phi-3.5-Mini (~2.3 GB, ~2390 Chunks)',
      repo: 'bartowski/Phi-3.5-mini-instruct-GGUF',
      filename: 'Phi-3.5-mini-instruct-Q4_K_M.gguf',
      approxBytes: 2390000000
    }
  };

  static async fetchStream(modelKey, onProgress) {
    const config = this.MODEL_MAP[modelKey];
    if (!config) throw new Error(`Model ${modelKey} is not registered.`);

    const targetUrl = `https://huggingface.co/${config.repo}/resolve/main/${config.filename}`;
    const response = await fetch(targetUrl);

    if (!response.ok) {
      throw new Error(`Failed HTTP ${response.status}: ${response.statusText}`);
    }

    const totalExpected = +(response.headers.get('Content-Length') || config.approxBytes);
    const reader = response.body.getReader();
    let accumulated = 0;
    const slices = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      slices.push(value);
      accumulated += value.length;

      if (onProgress) {
        onProgress({
          loaded: accumulated,
          total: totalExpected,
          percent: Math.min(100, (accumulated / totalExpected) * 100)
        });
      }
    }

    return {
      name: config.filename,
      modelKey,
      blob: new Blob(slices, { type: 'application/octet-stream' }),
      totalSize: accumulated
    };
  }
}
