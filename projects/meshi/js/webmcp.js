export function registerWebMCP(appInstance, db) {
  const tools = [
    {
      name: "meshi_get_network_status",
      description: "Returns local node identity, authorization state, active mesh links, and logical clock value.",
      parameters: { type: "object", properties: {} },
      handler: async () => ({
        localNode: appInstance.nodeId,
        publicKey: appInstance.crypto ? appInstance.crypto.publicKeyBase64.substring(0, 32) + "..." : null,
        socketReadyState: appInstance.activeSocket ? appInstance.activeSocket.readyState : 3,
        isAuthorized: appInstance.activeSocket ? appInstance.activeSocket.isAuthorized : false,
        lamportClock: db.clock,
        storedRecords: (await db.getAllDocuments()).length,
        timestamp: new Date().toISOString()
      })
    },
    {
      name: "meshi_trust_peer",
      description: "Whitelists a peer's public key in IndexedDB to permit mutual WebCrypto authentication.",
      parameters: {
        type: "object",
        properties: {
          publicKey: { type: "string", description: "Base64 SPKI public key string" },
          alias: { type: "string", description: "Human-readable peer identifier" }
        },
        required: ["publicKey"]
      },
      handler: async ({ publicKey, alias }) => {
        await db.addTrustedPeer(publicKey, alias || "Agent Provisioned Peer");
        return { status: "trusted", publicKey };
      }
    },
    {
      name: "meshi_dispatch_record",
      description: "Persists a JSON document in the local store and broadcasts it across open authorized channels.",
      parameters: {
        type: "object",
        properties: {
          id: { type: "string", description: "Unique document key" },
          payload: { type: "object", description: "Arbitrary JSON payload content" }
        },
        required: ["id", "payload"]
      },
      handler: async ({ id, payload }) => {
        const saved = await db.putDocument({ id, payload, sender: appInstance.nodeId });
        if (appInstance.activeSocket && appInstance.activeSocket.isAuthorized) {
          appInstance.activeSocket.send({ type: 'DOC_BROADCAST', payload: saved });
        }
        return { status: "persisted_and_dispatched", record: saved };
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
      if (!tool) throw new Error(`WebMCP Tool '${name}' not found.`);
      return await tool.handler(args);
    }
  };
}
