/**
 * WebMCP Specification Implementation for Project Meshi
 * Allows browser-based AI agents to query the P2P engine and dispatch state deltas.
 */
export function registerWebMCP(meshiEngine) {
  const tools = [
    {
      name: "meshi_get_node_status",
      description: "Returns the local Meshi WebRTC node ID, connection state, and current replica state snapshot.",
      parameters: { type: "object", properties: {} },
      handler: async () => {
        return {
          nodeId: meshiEngine.nodeId,
          connectionState: meshiEngine.connection?.iceConnectionState || "disconnected",
          channelReady: meshiEngine.channel?.readyState === "open",
          state: meshiEngine.getStateSnapshot()
        };
      }
    },
    {
      name: "meshi_sync_delta",
      description: "Publishes a CRDT key-value state update to all connected peers.",
      parameters: {
        type: "object",
        properties: {
          key: { type: "string", description: "Storage identifier key" },
          value: { description: "Arbitrary JSON value or string payload" }
        },
        required: ["key", "value"]
      },
      handler: async ({ key, value }) => {
        meshiEngine.set(key, value);
        return { status: "dispatched", key, timestamp: Date.now() };
      }
    },
    {
      name: "meshi_create_offer",
      description: "Generates a WebRTC session offer envelope for peer-to-peer pairing.",
      parameters: { type: "object", properties: {} },
      handler: async () => {
        const envelope = await meshiEngine.createOfferEnvelope();
        return { offer: envelope };
      }
    }
  ];

  // W3C document.modelContext standard registration
  if (typeof document !== 'undefined' && 'modelContext' in document) {
    tools.forEach((tool) => {
      try {
        document.modelContext.registerTool({
          name: tool.name,
          description: tool.description,
          inputSchema: tool.parameters,
          execute: tool.handler
        });
      } catch (err) {
        console.warn('WebMCP tool registration skipped:', tool.name, err);
      }
    });
  }

  // Global browser agent fallback
  window.__WEBMCP__ = {
    version: "1.0.0",
    protocol: "mcp/1.0",
    listTools: () => tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
    callTool: async (name, args) => {
      const t = tools.find((tool) => tool.name === name);
      if (!t) throw new Error(`Tool '${name}' not found.`);
      return await t.handler(args);
    }
  };
}
