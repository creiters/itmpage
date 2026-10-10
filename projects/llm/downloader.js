export class HuggingFaceDownloader {
  static MODEL_MAP = {
    'smollm2-360m': {
      title: 'SmolLM2-360M (~271 MB, ~271 Chunks)',
      repo: 'bartowski/SmolLM2-360M-Instruct-GGUF',
      filename: 'SmolLM2-360M-Instruct-Q4_K_M.gguf',
      approxBytes: 271000000
    },
    'llama-3.2-1b': {
      title: 'Llama-3.2-1B (~780 MB, ~780 Chunks)',
      repo: 'bartowski/Llama-3.2-1B-Instruct-GGUF',
      filename: 'Llama-3.2-1B-Instruct-Q4_K_M.gguf',
      approxBytes: 780000000
    },
    'qwen2.5-1.5b': {
      title: 'Qwen2.5-1.5B (~990 MB, ~990 Chunks)',
      repo: 'bartowski/Qwen2.5-1.5B-Instruct-GGUF',
      filename: 'Qwen2.5-1.5B-Instruct-Q4_K_M.gguf',
      approxBytes: 990000000
    },
    'smollm2-1.7b': {
      title: 'SmolLM2-1.7B (~1.06 GB, ~1060 Chunks)',
      repo: 'bartowski/SmolLM2-1.7B-Instruct-GGUF',
      filename: 'SmolLM2-1.7B-Instruct-Q4_K_M.gguf',
      approxBytes: 1060000000
    },
    'phi-3.5-mini': {
      title: 'Phi-3.5-Mini (~2.39 GB, ~2390 Chunks)',
      repo: 'bartowski/Phi-3.5-mini-instruct-GGUF',
      filename: 'Phi-3.5-mini-instruct-Q4_K_M.gguf',
      approxBytes: 2390000000
    }
  };

  static async fetchStream(modelKey, onProgress) {
    const config = this.MODEL_MAP[modelKey];
    if (!config) throw new Error(`Model ${modelKey} is not registered.`);

    // Hugging Face direct resolve endpoint
    const targetUrl = `https://huggingface.co/${config.repo}/resolve/main/${config.filename}`;
    
    const response = await fetch(targetUrl, {
      method: 'GET',
      headers: {
        'Accept': 'application/octet-stream'
      }
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status} (${response.statusText}): Could not retrieve ${config.filename}`);
    }

    // Handle Content-Length when stripped across CDN redirects
    const headerLen = response.headers.get('Content-Length');
    const totalExpected = (headerLen && !isNaN(+headerLen) && +headerLen > 0) 
      ? +headerLen 
      : config.approxBytes;

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
