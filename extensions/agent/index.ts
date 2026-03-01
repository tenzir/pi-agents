import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { type Message, StringEnum } from "@mariozechner/pi-ai";
import type {
  AgentToolResult,
  ExtensionAPI,
} from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import {
  type Agent,
  type Scope,
  type Source,
  buildSkillsPrompt,
  discoverAgents,
  formatAgentList,
  type Thinking,
} from "./agents.js";

interface UsageStats {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  cost: number;
  contextTokens: number;
  turns: number;
}

interface AgentToolExecutionResult<T> extends AgentToolResult<T> {
  isError?: boolean;
}

interface AgentRunDetails {
  agent: string;
  agentSource: Source | "unknown";
  model?: string;
  thinking?: Thinking;
  skills: string[];
  missingSkills: string[];
  exitCode: number;
  stopReason?: string;
  errorMessage?: string;
  stderr: string;
  usage: UsageStats;
  discoveryDiagnostics: string[];
  scope: Scope;
}

class DelegatedAgentRunError extends Error {
  readonly details: AgentRunDetails;

  constructor(message: string, details: AgentRunDetails) {
    super(message);
    this.name = "DelegatedAgentRunError";
    this.details = details;
  }
}

export type SpawnProcess = typeof spawn;

function getFinalOutput(messages: Message[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role !== "assistant") continue;
    const chunks: string[] = [];
    for (const part of msg.content) {
      if (part.type === "text") chunks.push(part.text);
    }
    if (chunks.length > 0) return chunks.join("\n").trim();
  }
  return "";
}

function writePromptToTempFile(
  agentName: string,
  prompt: string,
): { dir: string; filePath: string } {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agent-run-"));
  const safeName = agentName.replace(/[^\w.-]+/g, "_") || "agent";
  const filePath = path.join(dir, `append-system-${safeName}.md`);
  fs.writeFileSync(filePath, prompt, { encoding: "utf-8", mode: 0o600 });
  return { dir, filePath };
}

function initialUsage(): UsageStats {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    cost: 0,
    contextTokens: 0,
    turns: 0,
  };
}

function toDiagnosticText(
  scope: Scope,
  diagnostics: Array<{ filePath: string; message: string }>,
): string[] {
  const prefix = `scope=${scope}`;
  return diagnostics.map((d) => `${prefix}: ${d.filePath}: ${d.message}`);
}

function formatAgentsOverview(
  scope: Scope,
  agents: Agent[],
  diagnostics: string[],
): string {
  if (agents.length === 0) {
    const parts = [
      `No agents found for scope=${scope}.`,
      "Expected locations:",
      "- ~/.pi/agents/*.md",
      "- nearest .pi/agents/*.md",
    ];
    if (diagnostics.length > 0) {
      parts.push("", "Diagnostics:", ...diagnostics.map((d) => `- ${d}`));
    }
    return parts.join("\n");
  }

  const lines = [`Available agents (${agents.length}) [scope=${scope}]:`];
  for (const agent of agents) {
    lines.push(`- ${agent.name} (${agent.source}) — ${agent.description}`);
  }
  lines.push("", "Use /agents <name> for full details.");
  if (diagnostics.length > 0) {
    lines.push("", "Diagnostics:", ...diagnostics.map((d) => `- ${d}`));
  }
  return lines.join("\n");
}

function formatAgentDetails(
  scope: Scope,
  agent: Agent,
  diagnostics: string[],
): string {
  const lines = [
    `Agent: ${agent.name}`,
    `Scope: ${scope}`,
    `Source: ${agent.source}`,
    `Path: ${agent.filePath}`,
    `Description: ${agent.description}`,
    `Model: ${agent.model ?? "(inherit from current session model)"}`,
    `Thinking: ${agent.thinking ?? "(inherit from current session setting)"}`,
    `Skills: ${agent.skills.length > 0 ? agent.skills.join(", ") : "(none)"}`,
    "",
    "System prompt:",
    agent.systemPrompt || "(empty)",
  ];

  if (diagnostics.length > 0) {
    lines.push("", "Diagnostics:", ...diagnostics.map((d) => `- ${d}`));
  }

  return lines.join("\n");
}

function parseMessageEndEvent(line: string): Message | null {
  try {
    const parsed: unknown = JSON.parse(line);
    if (typeof parsed !== "object" || parsed === null) return null;
    const event = parsed as { type?: unknown; message?: unknown };
    if (event.type !== "message_end") return null;
    if (typeof event.message !== "object" || event.message === null)
      return null;
    return event.message as Message;
  } catch {
    return null;
  }
}

export function isChildProcessRunning(
  proc: Pick<ChildProcessWithoutNullStreams, "exitCode" | "signalCode">,
): boolean {
  return proc.exitCode === null && proc.signalCode === null;
}

function stripStackTrace(text: string): string {
  const lines = text.split(/\r?\n/).map((line) => line.trimEnd());
  const cleaned: string[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      if (cleaned.length > 0 && cleaned[cleaned.length - 1] !== "")
        cleaned.push("");
      continue;
    }
    if (trimmed.startsWith("at ")) continue;
    if (trimmed.startsWith("file://")) continue;
    if (/^Node\.js v\d+/i.test(trimmed)) continue;
    cleaned.push(trimmed);
  }

  return cleaned.join("\n").trim();
}

export function formatFailureReason(
  rawReason: string,
  modelHint?: string,
): string {
  const compact = stripStackTrace(rawReason);
  const source = compact || rawReason;
  const missingKeyMatch = source.match(/No API key found for ([^.\n]+)/i);
  if (missingKeyMatch) {
    const provider = missingKeyMatch[1]?.trim() || "the selected provider";
    const model = modelHint ? ` Model: ${modelHint}.` : "";
    return `No credentials configured for provider "${provider}".${model} Run /login or configure the provider API key, then retry.`;
  }

  return compact || "(no output)";
}

const ScopeSchema = StringEnum(["user", "project", "both"] as const, {
  description:
    'Which agents to load. "user" reads ~/.pi/agents. "project" reads nearest .pi/agents. "both" merges both (project wins).',
  default: "both",
});

const ParamsSchema = Type.Object({
  name: Type.String({
    description: "Name of the agent definition from markdown frontmatter",
  }),
  task: Type.String({ description: "Task to delegate" }),
  scope: Type.Optional(ScopeSchema),
  cwd: Type.Optional(
    Type.String({
      description: "Working directory for the delegated agent process",
    }),
  ),
});

export function createAgentExtension(options?: {
  spawnProcess?: SpawnProcess;
}) {
  const spawnProcess = options?.spawnProcess ?? spawn;
  return function agentExtension(pi: ExtensionAPI) {
    pi.registerCommand("agents", {
      description:
        "List available agents, or show details for a specific agent",
      getArgumentCompletions: (prefix) => {
        const discovery = discoverAgents(process.cwd(), "both");
        const items = discovery.agents
          .filter((agent) => agent.name.startsWith(prefix))
          .map((agent) => ({
            value: agent.name,
            label: agent.name,
            description: `${agent.source}: ${agent.description}`,
          }));
        return items.length > 0 ? items : null;
      },
      handler: async (args, ctx) => {
        const scope: Scope = "both";
        const query = args.trim();
        const discovery = discoverAgents(ctx.cwd, scope);
        const diagnostics = toDiagnosticText(scope, discovery.diagnostics);

        if (!query) {
          pi.sendMessage({
            customType: "agents",
            content: formatAgentsOverview(scope, discovery.agents, diagnostics),
            display: true,
          });
          return;
        }

        const agent = discovery.agents.find((a) => a.name === query);
        if (!agent) {
          pi.sendMessage({
            customType: "agents",
            content: `Unknown agent "${query}". Available: ${formatAgentList(discovery.agents)}`,
            display: true,
          });
          return;
        }

        pi.sendMessage({
          customType: "agents",
          content: formatAgentDetails(scope, agent, diagnostics),
          display: true,
        });
      },
    });

    pi.registerTool({
      name: "agent",
      label: "Agent",
      description:
        "Run an isolated pi agent from an agent markdown definition (name, description, model, thinking, skills).",
      parameters: ParamsSchema,
      async execute(
        _toolCallId,
        params,
        signal,
        onUpdate,
        ctx,
      ): Promise<AgentToolExecutionResult<AgentRunDetails>> {
        const scope: Scope = params.scope ?? "both";
        const discovery = discoverAgents(ctx.cwd, scope);
        const diagnostics = toDiagnosticText(scope, discovery.diagnostics);
        const agent = discovery.agents.find((a) => a.name === params.name);

        if (!agent) {
          const available = formatAgentList(discovery.agents);
          const message = `Unknown agent "${params.name}". Available: ${available}`;
          return {
            content: [{ type: "text", text: message }],
            details: {
              agent: params.name,
              agentSource: "unknown",
              skills: [],
              missingSkills: [],
              exitCode: 1,
              stderr: "",
              usage: initialUsage(),
              discoveryDiagnostics: diagnostics,
              scope,
            },
            isError: true,
          };
        }

        return runSingleAgent({
          agent,
          task: params.task,
          cwd: params.cwd ?? ctx.cwd,
          signal,
          onUpdate,
          scope,
          discoveryDiagnostics: diagnostics,
          spawnProcess,
        });
      },
    });
  };
}

export default createAgentExtension();

async function runSingleAgent(options: {
  agent: Agent;
  task: string;
  cwd: string;
  signal: AbortSignal | undefined;
  onUpdate: ((result: AgentToolResult<AgentRunDetails>) => void) | undefined;
  scope: Scope;
  discoveryDiagnostics: string[];
  spawnProcess: SpawnProcess;
}): Promise<AgentToolExecutionResult<AgentRunDetails>> {
  const {
    agent,
    task,
    cwd,
    signal,
    onUpdate,
    scope,
    discoveryDiagnostics,
    spawnProcess,
  } = options;

  const args: string[] = ["--mode", "json", "-p", "--no-session"];
  if (agent.model) {
    args.push("--model", agent.model);
  }
  if (agent.thinking) {
    args.push("--thinking", agent.thinking);
  }

  const { prompt: skillsPrompt, missingSkills } = buildSkillsPrompt(
    agent.skills,
    cwd,
  );
  const appendParts = [agent.systemPrompt.trim(), skillsPrompt.trim()].filter(
    Boolean,
  );

  let tempDir: string | undefined;
  let tempPromptPath: string | undefined;
  if (appendParts.length > 0) {
    const tmp = writePromptToTempFile(agent.name, appendParts.join("\n\n"));
    tempDir = tmp.dir;
    tempPromptPath = tmp.filePath;
    args.push("--append-system-prompt", tempPromptPath);
  }

  // Pipe the delegated task via stdin to avoid CLI argv parsing ambiguities.

  const messages: Message[] = [];
  const usage = initialUsage();
  let stopReason: string | undefined;
  let errorMessage: string | undefined;
  let resolvedModel = agent.model;
  let stderr = "";
  let wasAborted = false;

  const details = (exitCode: number): AgentRunDetails => ({
    agent: agent.name,
    agentSource: agent.source,
    model: resolvedModel,
    thinking: agent.thinking,
    skills: [...agent.skills],
    missingSkills,
    exitCode,
    stopReason,
    errorMessage,
    stderr,
    usage: { ...usage },
    discoveryDiagnostics,
    scope,
  });

  const emitUpdate = () => {
    if (!onUpdate) return;
    onUpdate({
      content: [
        { type: "text", text: getFinalOutput(messages) || "(running...)" },
      ],
      details: details(-1),
    });
  };

  try {
    const exitCode = await new Promise<number>((resolve) => {
      const proc = spawnProcess("pi", args, {
        cwd,
        shell: false,
        stdio: ["pipe", "pipe", "pipe"],
      });
      let buffered = "";
      let forceKillTimer: ReturnType<typeof setTimeout> | undefined;
      let settled = false;

      const finish = (code: number) => {
        if (settled) return;
        settled = true;
        if (forceKillTimer) {
          clearTimeout(forceKillTimer);
          forceKillTimer = undefined;
        }
        resolve(code);
      };

      const parseLine = (line: string) => {
        if (!line.trim()) return;
        const msg = parseMessageEndEvent(line);
        if (!msg) return;

        messages.push(msg);
        if (msg.role === "assistant") {
          usage.turns += 1;
          if (msg.usage) {
            usage.input += msg.usage.input || 0;
            usage.output += msg.usage.output || 0;
            usage.cacheRead += msg.usage.cacheRead || 0;
            usage.cacheWrite += msg.usage.cacheWrite || 0;
            usage.cost += msg.usage.cost?.total || 0;
            usage.contextTokens = msg.usage.totalTokens || usage.contextTokens;
          }
          if (msg.model) {
            resolvedModel = msg.model;
          }
          if (msg.stopReason) {
            stopReason = msg.stopReason;
          }
          if (msg.errorMessage) {
            errorMessage = msg.errorMessage;
          }
        }
        emitUpdate();
      };

      proc.stdout.on("data", (chunk) => {
        buffered += chunk.toString();
        const lines = buffered.split("\n");
        buffered = lines.pop() || "";
        for (const line of lines) parseLine(line);
      });

      proc.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
      });

      proc.on("close", (code, signalCode) => {
        if (buffered.trim()) {
          parseLine(buffered);
        }
        if (signalCode) {
          const signalFailure = `Delegated "pi" process terminated by signal ${signalCode}.`;
          stderr = stderr ? `${stderr}\n${signalFailure}` : signalFailure;
          if (!errorMessage) errorMessage = signalFailure;
          finish(1);
          return;
        }
        finish(code ?? 0);
      });

      proc.on("error", (error) => {
        const errorText =
          error instanceof Error ? error.message : String(error);
        const spawnFailure = `Failed to spawn "pi": ${errorText}`;
        stderr = stderr ? `${stderr}\n${spawnFailure}` : spawnFailure;
        if (!errorMessage) errorMessage = spawnFailure;
        finish(1);
      });

      if (signal) {
        const kill = () => {
          if (wasAborted) return;
          wasAborted = true;
          proc.kill("SIGTERM");
          forceKillTimer = setTimeout(() => {
            if (isChildProcessRunning(proc)) proc.kill("SIGKILL");
          }, 5000);
        };
        if (signal.aborted) kill();
        else signal.addEventListener("abort", kill, { once: true });
      }

      proc.stdin.end(task);
    });

    if (wasAborted) {
      throw new DelegatedAgentRunError(`Agent ${agent.name} aborted.`, details(1));
    }

    const finalText = getFinalOutput(messages);
    const isError =
      exitCode !== 0 || stopReason === "error" || stopReason === "aborted";
    if (isError) {
      const rawReason = errorMessage || stderr || finalText || "(no output)";
      const reason = formatFailureReason(rawReason, resolvedModel);
      throw new DelegatedAgentRunError(
        `Agent ${agent.name} failed: ${reason}`,
        details(exitCode),
      );
    }

    return {
      content: [{ type: "text", text: finalText || "(no output)" }],
      details: details(exitCode),
    };
  } finally {
    if (tempPromptPath) {
      try {
        fs.unlinkSync(tempPromptPath);
      } catch {
        // ignore cleanup errors
      }
    }
    if (tempDir) {
      try {
        fs.rmSync(tempDir, { recursive: true, force: true });
      } catch {
        // ignore cleanup errors
      }
    }
  }
}
