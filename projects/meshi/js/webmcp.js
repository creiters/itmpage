export function registerWebMCP(meshCoordinator, db) {
  const tools = [
    {
      name: "meshi_get_topology",
      description: "Returns active subnet WebRTC DataChannel connections and local node state.",
      parameters: { type: "object", properties: {} },
      handler: async () => ({
        localNode: meshCoordinator.nodeId,
        activeSocket: meshCoordinator.activeSocket ? "CONNECTED" : "DISCONNECTED",
        documentsCount: (await db.getAllDocuments()).length,
        timestamp: new Date().toISOString()
      })
    },
    {
      name: "meshi_publish_document",
      description: "Creates and pushes a JSON document payload across active mesh sockets.",
      parameters: {
        type: "object",
        properties: {
          id: { type: "string", description: "Unique document identifier" },
          content: { type: "string", description: "Body text or serialised payload" }
        },
        required: ["id", "content"]
      },
      handler: async ({ id, content }) => {
        const saved = await db.putDocument({ id, content });
        if (meshCoordinator.activeSocket?.readyState === 1) {
          meshCoordinator.activeSocket.send({ type: 'DOC_SYNC', payload: saved });
        }
        return { status: "persisted_and_broadcast", doc: saved };
      }
    }
  ];

  if (typeof document !== 'undefined' && 'modelContext' in document) {
    tools.forEach((t) => {
      try {
        document.modelContext.registerTool({
          name: t.name,
          description: t.description,
          inputSchema: t.parameters,
          execute: t.handler
        });
      } catch (err) {
        console.warn("WebMCP registration skipped:", t.name, err);
      }
    });
  }

  window.__WEBMCP__ = {
    version: "1.0.0",
    protocol: "mcp/1.0",
    listTools: () => tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
    callTool: async (name, args) => {
      const tool = tools.find((t) => t.name === name);
      if (!tool) throw new Error(`WebMCP Tool '${name}' not found`);
      return await tool.handler(args);
    }
  };
}
