---
## MCP Support in Yantra

**What "Supporting MCP" Means**

Yantra can connect to any MCP server. Yantra agent can call MCP tools during execution. User configures MCP servers once, agent uses them automatically.
---
## Implementation Approach

**MCP Is Just Tool Calling**

MCP server exposes tools with schemas. Client calls tools with parameters. Client receives results. This is the same pattern as LLM function calling.

**Yantra Already Has Tool Execution**

Agent executes code. Agent calls APIs. Adding MCP is adding another tool type.

---

## Architecture

**MCP Client Module**

```
src-tauri/src/mcp/
├── client.rs       # Connect to MCP servers
├── discovery.rs    # List available tools from server
├── executor.rs     # Call MCP tools
└── config.rs       # Store server configurations
```

**How It Fits**

User configures MCP server (URL, auth). Yantra discovers available tools from server. When agent needs external access, agent chooses: generate code and execute, OR call MCP tool directly. Agent decides based on context.

---

## Configuration Storage

**User Adds MCP Server**

```toml
# .yantra/mcp.toml
[[servers]]
name = "github"
url = "mcp://localhost:3000/github"
auth = "env:GITHUB_TOKEN"

[[servers]]
name = "postgres"
url = "mcp://localhost:3001/postgres"
auth = "env:DATABASE_URL"
```

**Credentials Never Stored Directly**

Reference environment variables. Reference system keychain. Reference secrets manager. Agent never sees raw credentials.

---

## Agent Integration

**Tool Discovery at Session Start**

Agent starts. Yantra loads configured MCP servers. Yantra queries each server: "What tools do you offer?" Tools added to agent's available actions.

**Agent Decides When to Use MCP**

Agent needs to query database.

Option A: Generate psycopg2 code, execute it. Option B: Call MCP postgres tool.

Agent chooses MCP when: credentials are managed through MCP, sandboxed access is required, complex auth (OAuth) is handled by MCP.

Agent chooses code when: simple operation, no MCP server configured, more flexibility needed.

**Prompt Enhancement**

System prompt includes available MCP tools:

"You have access to these MCP tools:

* github.create_pr(title, body, branch)
* github.list_issues(state, labels)
* postgres.query(sql) [read-only]
* postgres.schema(table_name)

Use these when appropriate, or generate code for direct access."

---

## User Experience

**Adding MCP Server**

Settings → Integrations → Add MCP Server. Enter server URL. Authenticate (OAuth flow or token). Yantra tests connection, shows available tools. Done.

**During Conversation**

User: "Create a PR for this feature"

Agent has GitHub MCP configured. Agent calls `github.create_pr()` directly. No code generation needed for this action.

User: "Query users who signed up last week"

Agent has Postgres MCP configured. Agent calls `postgres.query()` with SQL. Returns results directly.

**Transparency**

Show user when MCP tool is called. "Using GitHub MCP to create PR..." User sees what's happening, can intervene if needed.

---

## When Agent Uses MCP vs Code

**Prefer MCP When**

One-off operations (create PR, send notification). Managed credentials required. Sandbox enforcement required. Complex auth already handled by MCP server.

**Prefer Code When**

Logic needed around the operation. Multiple operations in sequence. MCP tool doesn't exist for the operation. More control or customization needed.

**Agent Heuristic**

If MCP tool exists AND operation is simple AND no custom logic needed → use MCP. Otherwise → generate code.

---

## Implementation Phases

**Phase 1: Basic MCP Client**

Connect to MCP servers. Discover tools. Call tools with parameters. Return results. Manual configuration only.

**Phase 2: Agent Integration**

Tools appear in agent's action space. Agent decides when to use MCP vs code. Transparency in UI.

**Phase 3: Smart Suggestions**

Auto-detect when MCP would help. Suggest servers based on codebase. Streamline OAuth flows.

---

## Technical Details

**MCP Protocol**

JSON-RPC over stdio or HTTP. Standard message format for tool discovery and execution. Well-documented, reference implementations exist.

**Rust Implementation**

Use existing MCP client library if available. Or implement minimal client - protocol is simple. Async with Tokio for non-blocking calls.

**Security**

Credentials in environment or keychain, never in config files. TLS for remote MCP servers. User approves each new MCP server connection.

---

## MVP Scope

**What to Build**

MCP client that connects to servers. Configuration UI for adding servers. Tool discovery and listing. Tool execution from agent. Basic transparency (show when MCP used).


## Opportunity for Yantra

**Option 1: Curated Integration List - MVP**

Yantra maintains its own list of vetted MCP servers. Pre-configured connection settings. One-click setup for common integrations. Yantra controls quality and security.

* Github, Gitlab, Postgresql
* Linear

**Option 2: Bundle Common Servers -Post MVP**

Ship popular MCP servers with Yantra. GitHub, PostgreSQL, Slack, etc. Run locally, no external dependency. User just enables, no setup.

---

## Bottom Line

MCP support in Yantra is straightforward:

Build MCP client module. Let users configure servers. Expose MCP tools to agent. Agent chooses MCP or code based on context.

The value is managed credentials, sandboxed access, and pre-built complexity for OAuth flows. The implementation is just another tool type in the agent's toolkit.

---



## **Medium Term: Bundled Servers**

Ship MCP servers for top 5 integrations. Zero configuration for common cases. Differentiator over Cursor.

## **Bundled MCP Setup (Zero Touch)**

User opens Yantra settings. User clicks "Enable GitHub Integration." User authenticates with GitHub (OAuth popup). Done. Yantra handles server internally.

## Technical Implementation

**MCP Servers as Rust Modules**

Don't bundle external servers. Implement MCP server logic directly in Yantra. No separate process. No dependencies. Just Rust code.

```
src-tauri/src/mcp/servers/
├── mod.rs
├── github.rs      # GitHub API wrapper
├── postgres.rs    # PostgreSQL client
├── slack.rs       # Slack API wrapper
├── filesystem.rs  # Local file access
└── http.rs        # Generic REST API
```

**Each Server Is Just Code**

## Credential Management

**OAuth Integrations (GitHub, GitLab, Slack, Google)**

Yantra handles OAuth flow. Tokens stored in system keychain. Refresh handled automatically. User never sees tokens.

**Token-Based Integrations (AWS, generic APIs)**

User enters token in settings. Stored in system keychain. Environment variable reference also supported.

**Connection String Integrations (PostgreSQL, MySQL)**

User enters connection string. Or selects from detected environment variables. Stored securely.

**Keychain Storage**

macOS: Keychain Access. Windows: Credential Manager. Linux: libsecret/GNOME Keyring.

User credentials never in plain text files. Agent accesses through Yantra, never sees raw credentials.

## Which Servers to Bundle

**Tier 1: Essential (Ship in MVP+1)**

* GitHub - PR creation, issue tracking, branch management. PostgreSQL - Schema discovery, query validation. Filesystem - Already have this, formalize as MCP interface.

**Tier 2: High Value (Phase 2)**

* GitLab - GitHub alternative, many enterprises use it. Slack - Notifications, team updates. AWS - Deployment, S3, secrets manager.

**Tier 3: Expanded (Phase 3)**

* MySQL, MongoDB - Database alternatives. Google Drive, Notion - Documentation access. Jira, Linear - Project management. SendGrid, Twilio - Communications.

**Tier 4: Generic Fallback**

* HTTP/REST - Generic API caller with auth handling. Custom MCP - Connect to external MCP servers for anything not bundled.

## Benefits Over External MCP

**Zero Configuration**

No server process to run. No ports to configure. No dependencies to install.

**Faster**

No IPC overhead. No JSON-RPC serialization. Direct function calls.

**More Reliable**

No server crashes. No connection issues. No version mismatches.

**Better Security**

Credentials in system keychain. No credential passing over protocol. Yantra controls all access.

**Cross-Platform**

Works identically on macOS, Windows, Linux. No platform-specific server setup

**Long Term: Evaluate Registry**

* If MCP adoption grows and no one else builds registry, consider it. But this is infrastructure play, not core product. Only if strategic.
