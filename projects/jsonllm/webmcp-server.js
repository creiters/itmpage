/**
 * webmcp-server.js
 * Clean WebMCP JSON-RPC 2.0 Protocol Engine
 */

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
              serverInfo: { name: 'clean-jsonllm-server', version: '2.0.0' },
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
                  name: 'save_model_to_json',
                  description: 'Persist active weights into IndexedDB clean JSON store',
                  inputSchema: { type: 'object', properties: { modelName: { type: 'string' } } }
                },
                {
                  name: 'rag_index_fact',
                  description: 'Index fact into local Vector RAG store',
                  inputSchema: { type: 'object', properties: { docId: { type: 'string' }, text: { type: 'string' } }, required: ['docId', 'text'] }
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
          if (name === 'save_model_to_json') {
            const mName = args.modelName || 'agent-dialog';
            const jsonStr = await this.JSONLLMCodec.exportToJSONLLM(
              { dim: this.engine.dim, layers: this.engine.layers, vocabSize: this.engine.vocabSize, maxSeqLen: this.engine.maxSeqLen },
              this.engine.weights
            );
            await this.storage.putModel(mName, jsonStr);
            return { jsonrpc: '2.0', id, result: { content: [{ type: 'text', text: `Saved ${mName} directly to IndexedDB (${(jsonStr.length / 1024).toFixed(1)} KB).` }] } };
          }
          if (name === 'rag_index_fact') {
            await this.rag.indexDocument(args.docId, args.text);
            return { jsonrpc: '2.0', id, result: { content: [{ type: 'text', text: `Document ${args.docId} indexed into RAG.` }] } };
          }
          throw new Error(`Tool not found: ${name}`);
        }

        case 'resources/list':
          return {
            jsonrpc: '2.0',
            id,
            result: {
              resources: [
                { uri: 'jsonllm://models/active', name: 'Active In-Memory JSON Model' },
                { uri: 'jsonllm://rag/index', name: 'Vector RAG Index' }
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
