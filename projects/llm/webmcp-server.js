import { NanoTransformer } from './nano-engine.js';
import { StorageService } from './storage.js';

export class WebMCPServer {
  constructor(options = {}) {
    this.serverName = options.name || 'browser-native-llm';
    this.version = options.version || '1.0.0';
    this.engine = new NanoTransformer();
    this.storage = new StorageService();
    this.mesh = options.mesh || null;
    this.initialized = false;
  }

  async init() {
    await this.storage.init();
    return this;
  }

  async handleMessage(request) {
    if (!request || request.jsonrpc !== '2.0') {
      return this.errorResponse(request?.id, -32600, 'Invalid Request: missing jsonrpc: "2.0"');
    }

    const { id, method, params } = request;

    try {
      switch (method) {
        case 'initialize':
          return this.handleInitialize(id, params);
        case 'notifications/initialized':
          this.initialized = true;
          return null;
        case 'tools/list':
          return this.handleToolsList(id);
        case 'tools/call':
          return await this.handleToolsCall(id, params);
        case 'sampling/createMessage':
          return await this.handleSamplingCreateMessage(id, params);
        case 'resources/list':
          return await this.handleResourcesList(id);
        case 'resources/read':
          return await this.handleResourcesRead(id, params);
        default:
          return this.errorResponse(id, -32601, `Method not found: ${method}`);
      }
    } catch (err) {
      return this.errorResponse(id, -32603, `Internal error: ${err.message}`);
    }
  }

  handleInitialize(id, params) {
    return {
      jsonrpc: '2.0',
      id,
      result: {
        protocolVersion: '2024-11-05',
        capabilities: {
          sampling: { contextWindow: 1024, maxTokens: 128 },
          tools: { listChanged: true },
          resources: { subscribe: false, listChanged: true }
        },
        serverInfo: { name: this.serverName, version: this.version }
      }
    };
  }

  handleToolsList(id) {
    return {
      jsonrpc: '2.0',
      id,
      result: {
        tools: [
          {
            name: 'storage_query_chunks',
            description: 'Queries SI-Radix 4MB segment index records from IndexedDB',
            inputSchema: {
              type: 'object',
              properties: { modelName: { type: 'string' } },
              required: ['modelName']
            }
          },
          {
            name: 'mesh_dispatch_compute',
            description: 'Dispatches tasks across WebRTC mesh PIN channels',
            inputSchema: {
              type: 'object',
              properties: { prompt: { type: 'string' } },
              required: ['prompt']
            }
          },
          {
            name: 'train_model_step',
            description: 'Executes backpropagation on text and updates model parameters using AdamW',
            inputSchema: {
              type: 'object',
              properties: {
                text: { type: 'string', description: 'Corpus text segment' },
                learningRate: { type: 'number', default: 0.001 }
              },
              required: ['text']
            }
          },
          {
            name: 'get_training_metrics',
            description: 'Inspects active parameter counts and optimization steps',
            inputSchema: { type: 'object', properties: {} }
          }
        ]
      }
    };
  }

  async handleToolsCall(id, params) {
    const { name, arguments: args } = params || {};

    if (name === 'storage_query_chunks') {
      const summary = await this.storage.getChunkSummary(args.modelName || 'browser-nano-llm');
      return {
        jsonrpc: '2.0',
        id,
        result: { content: [{ type: 'text', text: JSON.stringify(summary, null, 2) }], isError: false }
      };
    }

    if (name === 'mesh_dispatch_compute') {
      if (!this.mesh) return this.errorResponse(id, -32000, 'Mesh unavailable');
      const task = this.mesh.dispatchCompute(args.prompt);
      return {
        jsonrpc: '2.0',
        id,
        result: {
          content: [{ type: 'text', text: task ? `Dispatched taskId ${task.taskId} to peer ${task.peerKey}` : 'No peers' }],
          isError: !task
        }
      };
    }

    if (name === 'train_model_step') {
      const tokens = this.engine.tokenizer.encode(args.text);
      const loss = this.engine.trainSequence(tokens, args.learningRate || 0.001);
      return {
        jsonrpc: '2.0',
        id,
        result: {
          content: [{ type: 'text', text: `AdamW Step complete. Sequence Loss: ${loss.toFixed(4)}, Perplexity: ${Math.exp(loss).toFixed(2)}` }],
          isError: false
        }
      };
    }

    if (name === 'get_training_metrics') {
      let totalParams = 0;
      for (const k in this.engine.weights) totalParams += this.engine.weights[k].length;
      return {
        jsonrpc: '2.0',
        id,
        result: {
          content: [{
            type: 'text',
            text: JSON.stringify({
              totalParameters: totalParams,
              totalStepsTrained: this.engine.stepCount,
              vocabSize: this.engine.vocabSize,
              dim: this.engine.dim,
              layers: this.engine.layers
            }, null, 2)
          }],
          isError: false
        }
      };
    }

    return this.errorResponse(id, -32602, `Unknown tool: ${name}`);
  }

  async handleSamplingCreateMessage(id, params) {
    const { messages, maxTokens = 64 } = params || {};
    const prompt = messages && messages.length ? messages[messages.length - 1].content.text : '';
    const generated = this.engine.generate(prompt, maxTokens);

    return {
      jsonrpc: '2.0',
      id,
      result: {
        role: 'assistant',
        content: { type: 'text', text: generated },
        model: 'browser-native-nano',
        stopReason: 'endTurn'
      }
    };
  }

  async handleResourcesList(id) {
    return {
      jsonrpc: '2.0',
      id,
      result: {
        resources: [
          { uri: 'resource://model/si-radix-manifest', name: 'SI-Radix Interval Manifest', mimeType: 'application/json' },
          { uri: 'resource://storage/status', name: 'IndexedDB Quota Metrics', mimeType: 'application/json' }
        ]
      }
    };
  }

  async handleResourcesRead(id, params) {
    const { uri } = params || {};
    if (uri === 'resource://model/si-radix-manifest') {
      const manifest = await this.storage.getManifest('browser-nano-llm');
      return {
        jsonrpc: '2.0',
        id,
        result: { contents: [{ uri, mimeType: 'application/json', text: JSON.stringify(manifest || { status: 'uninitialized' }) }] }
      };
    }
    if (uri === 'resource://storage/status') {
      const quota = await this.storage.getStorageQuota();
      return {
        jsonrpc: '2.0',
        id,
        result: { contents: [{ uri, mimeType: 'application/json', text: JSON.stringify(quota) }] }
      };
    }
    return this.errorResponse(id, -32602, `Resource not found: ${uri}`);
  }

  errorResponse(id, code, message) {
    return { jsonrpc: '2.0', id: id !== undefined ? id : null, error: { code, message } };
  }
}
