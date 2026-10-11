export class WebMCPServer {
  constructor(engineInstance, storageInstance, ragInstance, jsonLlmCodecClass) {
    this.engine = engineInstance;
    this.storage = storageInstance;
    this.rag = ragInstance;
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
              serverInfo: { name: 'vimtollm-server', version: '3.0.0' },
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
                  description: 'Train on-device weights on input text',
                  inputSchema: { type: 'object', properties: { text: { type: 'string' } }, required: ['text'] }
                },
                {
                  name: 'generate_jsonllm',
                  description: 'Packs active weights into an FP4/Q4 Deflate JSONLLM container',
                  inputSchema: { type: 'object', properties: { modelName: { type: 'string' }, mode: { type: 'string' } } }
                },
                {
                  name: 'rag_index_document',
                  description: 'Index document into Vector RAG store',
                  inputSchema: { type: 'object', properties: { docId: { type: 'string' }, text: { type: 'string' } }, required: ['docId', 'text'] }
                }
              ]
            }
          };

        case 'tools/call': {
          const { name, arguments: args } = params;
          if (name === 'train_model_step') {
            const toks = this.engine.tokenizer.encode(args.text);
            const loss = this.engine.trainSequence(toks, args.learningRate || 0.003);
            return { jsonrpc: '2.0', id, result: { content: [{ type: 'text', text: `Trained step. Loss: ${loss.toFixed(4)}` }] } };
          }
          if (name === 'generate_jsonllm') {
            const mName = args.modelName || 'vimtollm-4m';
            const mode = args.mode || 'fp4';
            const jsonStr = await this.JSONLLMCodec.exportToJSONLLM(
              { dim: this.engine.dim, layers: this.engine.layers, heads: this.engine.heads, ffnDim: this.engine.ffnDim, vocabSize: this.engine.vocabSize, maxSeqLen: this.engine.maxSeqLen },
              this.engine.tokenizer,
              this.engine.weights,
              mode
            );
            await this.storage.putModel(mName, jsonStr);
            return { jsonrpc: '2.0', id, result: { content: [{ type: 'text', text: `Generated vimtoLLM (${mode.toUpperCase()}) for ${mName} (${(jsonStr.length / 1024).toFixed(1)} KB).` }] } };
          }
          if (name === 'rag_index_document') {
            await this.rag.indexDocument(args.docId, args.text);
            return { jsonrpc: '2.0', id, result: { content: [{ type: 'text', text: `Document ${args.docId} indexed into Vector RAG.` }] } };
          }
          throw new Error(`Tool not found: ${name}`);
        }

        case 'resources/list':
          return {
            jsonrpc: '2.0',
            id,
            result: {
              resources: [
                { uri: 'jsonllm://models/active', name: 'Active In-Memory vimtoLLM Model' },
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
