/**
 * Web Model Context Protocol (WebMCP) Registration
 * Exposes portfolio inspection, slide carousel navigation, and SSHAnet terminal dispatch.
 */
export function registerWebMCP(appInstance) {
  const tools = [
    {
      name: "list_portfolio_projects",
      description: "Returns the complete catalog of CreITers projects with descriptions, tags, and repo links.",
      parameters: { type: "object", properties: {} },
      handler: async () => {
        return {
          node: "creiters.cz/projects/cyberpear",
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
          slide: { type: "string", description: "Project id (e.g., 'itmpage') or numeric slide index" }
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
          command: { type: "string", description: "Command line input string" }
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
