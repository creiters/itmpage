export class WebMCPServer {
  constructor(engineInstance, storageInstance, ragInstance, chunkIndexerClass, jsonLlmCodecClass) {
    this.engine = engineInstance;
    this.storage = storageInstance;
    this.rag = ragInstance;
    this.ChunkIndexer = chunkIndexerClass;
    this.JSONLLMCodec = jsonLlmCodecClass;
  }

  async handleRequest(rpc) {
    const { id, method, params } = rpc;

    try {
      switch (method) {
        case 'initialize':
          return {
            jsonrpc: '2.0',
            id,
            result: {
              protocolVersion: '2024-11-05',
              serverInfo: { name: 'browser-jsonllm-server', version: '1.0.0' },
              capabilities: { tools: {}, resources: {}, sampling: {} }
            }
          };

        case 'tools/list':
          return {
            jsonrpc: '2.0',
            id,
            result: {
              tools: [
                {
                  name: 'train_model_step',
                  description: 'Execute training backpropagation on text',
                  inputSchema: { type: 'object', properties: { text: { type: 'string' } }, required: ['text'] }
                },
                {
                  name: 'rag_index_document',
                  description: 'Index document into Vector RAG store',
                  inputSchema: { type: 'object', properties: { docId: { type: 'string' }, text: { type: 'string' } }, required: ['docId', 'text'] }
                },
                {
                  name: 'save_model_si_radix',
                  description: 'Export weights to 4MB SI-Radix JSONLLM slices',
                  inputSchema: { type: 'object', properties: { modelName: { type: 'string' } } }
                }
              ]
            }
          };

        case 'tools/call': {
          const { name, arguments: args } = params;
          if (name === 'train_model_step') {
            const toks = Array.from(new TextEncoder().encode(args.text));
            const loss = this.engine.trainStep(toks, args.learningRate || 0.002);
            return { jsonrpc: '2.0', id, result: { content: [{ type: 'text', text: `Trained step. Loss: ${loss.toFixed(4)}` }] } };
          }
          if (name === 'rag_index_document') {
            await this.rag.indexDocument(args.docId, args.text);
            return { jsonrpc: '2.0', id, result: { content: [{ type: 'text', text: `Document ${args.docId} indexed into RAG.` }] } };
          }
          if (name === 'save_model_si_radix') {
            const mName = args.modelName || 'agent-dialog';
            const jsonStr = await this.JSONLLMCodec.exportToJSONLLM(
              { dim: this.engine.dim, layers: this.engine.layers, vocabSize: this.engine.vocabSize },
              this.engine.weights
            );
            const slices = this.ChunkIndexer.slicePayload(jsonStr, mName);
            await this.storage.batchSaveSegments(slices);
            await this.storage.putModel(mName, jsonStr);
            return { jsonrpc: '2.0', id, result: { content: [{ type: 'text', text: `Saved ${slices.length} SI-Radix slices.` }] } };
          }
          throw new Error(`Tool not found: ${name}`);
        }

        case 'resources/list':
          return {
            jsonrpc: '2.0',
            id,
            result: {
              resources: [
                { uri: 'jsonllm://models/active', name: 'Active In-Memory Small LLM' },
                { uri: 'jsonllm://rag/documents', name: 'Vector RAG Semantic Index' }
              ]
            }
          };

        default:
          return { jsonrpc: '2.0', id, error: { code: -32601, message: `Method not found: ${method}` } };
      }
    } catch (err) {
      return { jsonrpc: '2.0', id, error: { code: -32000, message: err.message } };
    }
  }
}
