/**
 * Web Model Context Protocol (WebMCP) Bridge
 * Exposes page tools directly to browser AI agents and autonomous assistants.
 */
export function registerWebMCP(appInstance) {
  const tools = [
    {
      name: "list_portfolio_projects",
      description: "Returns the complete catalog of projects hosted on CreITers / ITMpage node.",
      parameters: { type: "object", properties: {} },
      handler: async () => {
        return {
          node: "creiters.cz",
          repository: "https://github.com/creiters/itmpage",
          projects: appInstance.projects
        };
      }
    },
    {
      name: "navigate_presentation_slide",
      description: "Selects and displays a presentation slide by its zero-based index or project identifier.",
      parameters: {
        type: "object",
        properties: {
          slide: { type: "string", description: "Project ID (e.g. 'itmpage') or numeric slide index" }
        },
        required: ["slide"]
      },
      handler: async ({ slide }) => {
        const result = appInstance.goToSlide(slide);
        return { status: "success", activeSlide: result };
      }
    },
    {
      name: "dispatch_terminal_command",
      description: "Executes a command on the live SSHAnet command console (status, ping, awaken, projects, clear, help).",
      parameters: {
        type: "object",
        properties: {
          command: { type: "string", description: "Command string to execute" }
        },
        required: ["command"]
      },
      handler: async ({ command }) => {
        appInstance.executeCommand(command);
        return { status: "acknowledged", command, timestamp: new Date().toISOString() };
      }
    }
  ];

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
        console.warn("WebMCP registration skipped:", tool.name, err);
      }
    });
  }

  window.__WEBMCP__ = {
    version: "1.0.0",
    protocol: "mcp/1.0",
    listTools: () => tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
    callTool: async (name, args) => {
      const tool = tools.find((t) => t.name === name);
      if (!tool) throw new Error(`Tool '${name}' not found.`);
      return await tool.handler(args);
    }
  };
}
