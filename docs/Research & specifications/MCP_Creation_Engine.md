# MCP Creation Engine - Technical Specification

**Version:** 2.0  
**Date:** December 9, 2025  
**Status:** Specification - Ready for Implementation  
**Phase:** Integration with Yantra Platform Core  
**Updated:** Aligned with Unified Tools Interface and Multi-Tenant Cloud Architecture

---

## 1. Executive Summary

### 1.1 Vision

The MCP Creation Engine is a code generation system integrated into **Yantra Develop** that enables **enterprises to create and manage their own MCP servers** reliably, without depending on community marketplaces. Generated MCPs integrate with the **Yantra Platform Core** via the **Unified Tools Interface (UTI)** and are automatically available across all Yantra products (Develop, Flow, Data).

**Strategic Differentiation vs. VSCode MCP Marketplace:**

| Approach          | VSCode MCP Marketplace            | Yantra MCP Creation Engine              |
| ----------------- | --------------------------------- | --------------------------------------- |
| **Philosophy**    | Community-maintained marketplace  | Enterprise self-service creation        |
| **Control**       | Depend on community contributions | Full control over integrations          |
| **Security**      | Trust third-party MCPs            | Enterprise controls entire stack        |
| **Customization** | Use what's available              | Generate exactly what you need          |
| **Reliability**   | Community maintenance (may break) | Enterprise owns & maintains             |
| **IP Protection** | Public MCPs (exposed logic)       | Private MCPs (proprietary integrations) |
| **Compliance**    | Unknown security posture          | SOC2/HIPAA/GDPR compliant               |
| **Speed**         | Wait for community                | Generate in 60 seconds                  |

**Key Innovation:** "Create once, use everywhere" - MCPs created in Yantra Develop are automatically:

- Registered in the Tool Registry (Platform Core)
- Available in Yantra Flow (for workflow automation)
- Available in Yantra Data (for data analytics)
- Deployed to Kubernetes MCP Runtime (multi-tenant cloud)
- Managed with enterprise security (Vault, RBAC, audit logs)

### 1.2 Problem Statement

**Current Pain Points (Community Marketplace Model):**

- **Dependency Risk:** Enterprises depend on community-maintained MCPs that may break or become unmaintained
- **Security Concerns:** Third-party MCPs have unknown security posture, potential vulnerabilities
- **Customization Limits:** Community MCPs are generic, not tailored to enterprise needs
- **IP Exposure:** Publishing custom integrations to public marketplace exposes proprietary business logic
- **Compliance Gaps:** Community MCPs may not meet SOC2/HIPAA/GDPR requirements
- **No Governance:** Can't enforce who uses which MCPs, no audit trails
- **Integration Gaps:** Community may not have MCPs for internal/legacy systems

**Yantra's Enterprise-First Solution:**

Generate complete, tested, production-ready MCP servers from natural language in <60 seconds, giving enterprises full control over their integration layer. No waiting for community, no security risks from third-party code, no compliance concerns. Enterprise developers create exactly what they need, with enterprise-grade security, governance, and reliability built-in.

### 1.3 Success Criteria

**MVP Success Metrics:**

- ✅ Generate working MCP server in <60 seconds
- ✅ Support 4 common templates (Database, REST API, File System, Custom)
- ✅ Generated code implements Unified Tools Interface specification
- ✅ Automatic registration in Tool Registry
- ✅ One-click deployment to Kubernetes MCP Runtime
- ✅ Credentials managed via Vault (per-tenant encryption)
- ✅ MCP immediately available in Flow and Data after deployment
- ✅ 90%+ of generated servers work without manual editing
- ✅ RBAC enforced (who can create, deploy, use MCPs)

**Note:** Browser automation is already built into Yantra Develop IDE via CDP (Chrome DevTools Protocol) as documented in Specifications.md section 3.1.8. It is a core IDE feature for UI validation, testing, and self-healing - not an MCP template.

### 1.4 Enterprise Use Cases (Not Served by Community Marketplaces)

**Internal Systems Integration:**

- Legacy mainframe systems (COBOL, AS/400) - Too niche for community
- Custom internal databases with proprietary schemas
- Internal APIs with company-specific authentication
- On-premise systems behind corporate firewalls

**Compliance-Critical Integrations:**

- Healthcare systems requiring HIPAA compliance (can't trust community code)
- Financial systems requiring SOC2 audit trails
- Government systems requiring FedRAMP certification
- PCI-DSS compliant payment processing

**Competitive Advantage:**

- Proprietary data sources (can't expose to community)
- Custom business logic (trade secrets)
- Unique vendor integrations (competitive differentiation)
- Internal tools and workflows (IP protection)

**Governance & Control:**

- Enterprise-managed credentials (Vault integration)
- Role-based access control (who can use which MCPs)
- Audit logging for compliance
- Version control and rollback capabilities
- Approval workflows for production deployment

**Example:** A healthcare company needs an MCP for their custom EHR system. They can't:

1. Wait for community to build it (may never happen)
2. Publish their EHR schema publicly (HIPAA violation)
3. Trust third-party MCP code (security risk)
4. Use generic database MCP (doesn't understand EHR semantics)

**Yantra Solution:** Generate HIPAA-compliant EHR MCP in 60 seconds, with enterprise credentials in Vault, audit logs enabled, and immediate availability across Develop, Flow, and Data.

---

## 2. Architecture Overview

### 2.1 System Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│  Yantra Develop (Desktop IDE)                                                  │
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │  UI Layer (Tauri Frontend)                                               │ │
│  │  - Natural language MCP creator                                          │ │
│  │  - Visual wizard with templates                                          │ │
│  │  - Code editor (Monaco) for customization                                │ │
│  │  - Test runner UI                                                        │ │
│  │  - Deployment UI (local test → cloud deploy)                             │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
│                         ↓                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │  MCP Creation Engine (Rust Backend)                                      │ │
│  │  ┌────────────────────────────────────────────────────────────────────┐ │ │
│  │  │  Intent Parser                                                      │ │ │
│  │  │  - NL → structured requirements                                     │ │ │
│  │  │  - Template selection (Database, API, File System, Custom)         │ │ │
│  │  │  - Parameter extraction                                             │ │ │
│  │  └────────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌────────────────────────────────────────────────────────────────────┐ │ │
│  │  │  UTI Code Generator                                                 │ │ │
│  │  │  - Generate TypeScript implementing Tool interface                 │ │ │
│  │  │  - Generate resources[], actions[], capabilities()                 │ │ │
│  │  │  - Generate authentication hooks (OAuth, API key, etc.)            │ │ │
│  │  │  - Generate Docker + K8s manifests                                 │ │ │
│  │  └────────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌────────────────────────────────────────────────────────────────────┐ │ │
│  │  │  Validator                                                          │ │ │
│  │  │  - UTI interface compliance check                                   │ │ │
│  │  │  - Security audit (no hardcoded secrets, SQL injection checks)     │ │ │
│  │  │  - MCP protocol validation                                         │ │ │
│  │  └────────────────────────────────────────────────────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
│                         ↓                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │  Local Testing                                                           │ │
│  │  - Run MCP locally with test credentials                                │ │
│  │  - Validate with sample queries                                         │ │
│  │  - Test all actions and resources                                       │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
                         ↓ (Deploy button clicked)
┌───────────────────────────────────────────────────────────────────────────────┐
│  Yantra Platform Core (Cloud)                                                  │
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │  Deployment Service                                                      │ │
│  │  1. Upload generated MCP code to artifact storage                       │ │
│  │  2. Build Docker image                                                  │ │
│  │  3. Deploy to Kubernetes MCP Runtime                                    │ │
│  │  4. Register in Tool Registry with UTI metadata                         │ │
│  │  5. Configure RBAC (who can use this MCP)                               │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
│                         ↓                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │  Tool Registry (Platform Core)                                          │ │
│  │  - MCP registered as Tool implementing UTI                              │ │
│  │  - Metadata: name, version, author, capabilities, auth requirements     │ │
│  │  - Permissions: which orgs/users can access                             │ │
│  │  - Secrets: stored in Vault (per-tenant encrypted)                      │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
│                         ↓                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │  MCP Runtime (Kubernetes)                                                │ │
│  │  - MCP runs as pod with auto-scaling                                    │ │
│  │  - Health checks + monitoring                                           │ │
│  │  - Load balancing across replicas                                       │ │
│  │  - Credentials injected from Vault                                      │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
                         ↓ (Now available)
┌───────────────────────────────────────────────────────────────────────────────┐
│  Automatically Available In All Products                                       │
│                                                                                │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐      │
│  │ Yantra Develop     │  │ Yantra Flow        │  │ Yantra Data        │      │
│  │ - Test locally     │  │ - Use in workflows │  │ - Use for analytics│      │
│  │ - Debug            │  │ - Automation       │  │ - Cross-source join│      │
│  │ - Update/redeploy  │  │ - Trigger actions  │  │ - Query via NL     │      │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘      │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Integration with Unified Tools Interface

**All generated MCPs implement the UTI Tool interface:**

```typescript
// Generated MCP implements this interface (from Platform Core)
interface Tool {
  metadata: ToolMetadata;
  resources(): Promise<Resource[]>;
  actions(): Promise<Action[]>;
  capabilities(): Promise<Capability[]>;
  execute(action: string, parameters: any, context: ExecutionContext): Promise<ExecutionResult>;
  authenticate(credentials: any): Promise<AuthResult>;
}
```

**Example Generated MCP Structure:**

```typescript
// Generated file: src/index.ts

import {
  Tool,
  ToolMetadata,
  Resource,
  Action,
  Capability,
  ExecutionContext,
  ExecutionResult,
  AuthResult,
} from '@yantra/platform-core';

export class DatabaseMCPTool implements Tool {
  metadata: ToolMetadata = {
    id: 'mcp-postgres-mydb',
    name: 'My Database',
    description: 'PostgreSQL database for customer data',
    version: '1.0.0',
    author: 'user@example.com',
    category: 'database',
    auth_type: 'connection_string',
    tags: ['postgresql', 'database', 'customers'],
  };

  async resources(): Promise<Resource[]> {
    return [
      {
        uri: 'postgres://tables/users',
        name: 'users',
        description: 'User accounts table',
        schema: {
          /* ... */
        },
      },
      {
        uri: 'postgres://tables/orders',
        name: 'orders',
        description: 'Customer orders',
        schema: {
          /* ... */
        },
      },
    ];
  }

  async actions(): Promise<Action[]> {
    return [
      {
        name: 'query',
        description: 'Execute SQL query',
        parameters: {
          /* ... */
        },
      },
      {
        name: 'insert',
        description: 'Insert rows',
        parameters: {
          /* ... */
        },
      },
    ];
  }

  async capabilities(): Promise<Capability[]> {
    return [
      { type: 'read', supported: true },
      { type: 'write', supported: true },
      { type: 'batch', supported: true },
    ];
  }

  async execute(
    action: string,
    parameters: any,
    context: ExecutionContext
  ): Promise<ExecutionResult> {
    // Get credentials from Vault (injected by Platform Core)
    const connectionString = await context.getSecret('database_connection_string');

    // Execute action
    switch (action) {
      case 'query':
        return await this.executeQuery(parameters.sql, connectionString);
      case 'insert':
        return await this.insertRows(parameters.table, parameters.rows, connectionString);
      default:
        throw new Error(`Unknown action: ${action}`);
    }
  }

  async authenticate(credentials: any): Promise<AuthResult> {
    // Validate credentials
    // Store in Vault (handled by Platform Core)
    return { success: true };
  }

  // ... implementation methods
}
```

### 2.2 Component Breakdown

#### 2.2.1 Intent Parser

**Purpose:** Convert natural language to structured MCP requirements

**Input:**

```rust
struct MCPRequest {
    description: String,           // "Create MCP for PostgreSQL database"
    context: Option<String>,       // Additional context
    parameters: HashMap<String, String>, // Key-value pairs
}
```

**Output:**

```rust
struct MCPIntent {
    template_type: TemplateType,   // Database | API | FileSystem
    name: String,                  // "postgres-connector"
    resources: Vec<Resource>,      // List of resources to expose
    tools: Vec<Tool>,              // List of tools to implement
    auth_type: AuthType,           // None | ApiKey | OAuth
    config: HashMap<String, Value>, // Template-specific config
}
```

**Processing:**

1. Use LLM (Claude/GPT via user's API key) to parse description
2. Identify template type from keywords
3. Extract resource names, operations, authentication needs
4. Map to template schema
5. Validate completeness (ask clarifying questions if needed)

#### 2.2.2 Template Engine

**Purpose:** Load and apply template definitions

**Template Structure:**

```rust
struct Template {
    id: String,                    // "database-postgres"
    name: String,                  // "PostgreSQL Database MCP"
    description: String,
    files: Vec<TemplateFile>,      // List of files to generate
    dependencies: Vec<Dependency>, // npm packages
    environment_vars: Vec<EnvVar>, // Required .env variables
    validation_rules: Vec<ValidationRule>,
}

struct TemplateFile {
    path: String,                  // "src/resources.ts"
    template: String,              // Handlebars template
    required: bool,                // Must be generated
}
```

**Template Storage:**

- Location: `src-tauri/templates/mcp/`
- Format: JSON + Handlebars templates
- Built-in templates shipped with Yantra
- User can add custom templates (future)

#### 2.2.3 Code Generator

**Purpose:** Generate TypeScript code from templates

**Process:**

1. Load template files
2. Apply user parameters via Handlebars
3. Generate TypeScript code
4. Format with Prettier
5. Add type annotations
6. Generate tests
7. Generate README

**Output Files:**

```
mcp-server-{name}/
├── src/
│   ├── index.ts          # Server entry point
│   ├── server.ts         # MCP server setup
│   ├── resources.ts      # Resource handlers
│   ├── tools.ts          # Tool handlers
│   └── config.ts         # Configuration
├── tests/
│   ├── resources.test.ts
│   └── tools.test.ts
├── package.json
├── tsconfig.json
├── .env.example
├── .gitignore
└── README.md
```

#### 2.2.4 Validator

**Purpose:** Ensure generated code is correct and safe

**Validation Steps:**

1. **Syntax Validation:**
   - Parse TypeScript with SWC
   - Check for syntax errors
   - Verify imports

2. **MCP Protocol Validation:**
   - Check server implements required methods
   - Verify resource/tool schemas
   - Validate JSON-RPC structure

3. **Security Validation:**
   - Check for SQL injection vulnerabilities
   - Verify input sanitization
   - Flag hardcoded secrets
   - Check file path traversal

4. **Dependency Validation:**
   - Verify package versions exist
   - Check for known vulnerabilities
   - Validate semver ranges

**Output:**

```rust
struct ValidationResult {
    is_valid: bool,
    errors: Vec<ValidationError>,
    warnings: Vec<ValidationWarning>,
    suggestions: Vec<String>,
}
```

---

## 3. MVP Templates

### 3.1 Template 1: PostgreSQL Database MCP

**Use Case:** Expose PostgreSQL database tables as MCP resources

**Generated Resources:**

- List tables
- Read table schema
- Query table data (with filters)
- Get row by ID

**Generated Tools:**

- `execute_query(sql: string)` - Execute SELECT query
- `count_rows(table: string, filter?: object)` - Count rows
- `get_table_info(table: string)` - Get schema info

**Dependencies:**

```json
{
  "dependencies": {
    "@modelcontextprotocol/sdk": "^0.5.0",
    "pg": "^8.11.0",
    "zod": "^3.22.0"
  }
}
```

**Configuration:**

```typescript
interface PostgresConfig {
  host: string;
  port: number;
  database: string;
  user: string;
  password: string;
  tables?: string[]; // Whitelist (default: all)
  readOnly?: boolean; // Default: true
  maxRows?: number; // Default: 1000
}
```

**Security:**

- Read-only by default
- SQL injection prevention (parameterized queries)
- Table whitelist
- Row limit enforcement

### 3.2 Template 2: REST API Wrapper MCP

**Use Case:** Wrap REST API as MCP server

**Generated Resources:**

- API endpoints as resources
- Response schemas
- Pagination support

**Generated Tools:**

- HTTP methods: GET, POST, PUT, DELETE
- Authentication injection
- Error handling

**Dependencies:**

```json
{
  "dependencies": {
    "@modelcontextprotocol/sdk": "^0.5.0",
    "axios": "^1.6.0",
    "zod": "^3.22.0"
  }
}
```

**Configuration:**

```typescript
interface APIConfig {
  baseUrl: string;
  apiKey?: string;
  authType?: 'apiKey' | 'bearer' | 'basic';
  endpoints: Endpoint[];
  rateLimitPerMinute?: number;
  timeout?: number;
}

interface Endpoint {
  path: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  description: string;
  parameters?: Parameter[];
}
```

**Features:**

- Automatic retry with exponential backoff
- Rate limiting
- Request/response logging
- Error normalization

### 3.3 Template 3: File System MCP

**Use Case:** Access local files and directories via MCP

**Generated Resources:**

- List files in directory
- Read file contents
- File metadata (size, modified date)
- Directory tree

**Generated Tools:**

- `search_files(pattern: string, path: string)` - Search by name/content
- `read_file(path: string)` - Read file contents
- `list_directory(path: string)` - List directory contents
- `get_file_info(path: string)` - Get metadata

**Dependencies:**

```json
{
  "dependencies": {
    "@modelcontextprotocol/sdk": "^0.5.0",
    "fast-glob": "^3.3.2",
    "mime-types": "^2.1.35"
  }
}
```

**Configuration:**

```typescript
interface FileSystemConfig {
  rootPath: string;
  allowedExtensions?: string[]; // Default: all
  maxFileSize?: number; // Default: 10MB
  followSymlinks?: boolean; // Default: false
  readOnly?: boolean; // Default: true
}
```

**Security:**

- Path traversal prevention
- Sandboxed to rootPath
- File size limits
- Extension whitelist
- Read-only by default

---

## 4. User Interface

### 4.1 Chat Interface

**Natural Language Input:**

```
User: Create an MCP server for my PostgreSQL database

Yantra: I'll help you create a PostgreSQL MCP server.
        I need a few details:

        1. Database connection details:
           - Host: [localhost]
           - Port: [5432]
           - Database name: ?
           - Username: ?
           - Password: (will be stored in .env)

        2. Which tables should be accessible?
           - [All tables] or specify: ?

        3. Access level:
           - [Read-only] or Read-write?

User: Database is 'myapp', user 'admin', expose only
      'users' and 'orders' tables, read-only

Yantra: ✓ Generating MCP server...
        ✓ Created mcp-server-postgres/
        ✓ Generated 8 files
        ✓ Added tests

        Next steps:
        1. cd mcp-server-postgres
        2. cp .env.example .env
        3. Edit .env with your password
        4. npm install
        5. npm run build
        6. Add to Claude Desktop config (see README.md)

        📄 Full setup instructions in README.md
```

### 4.2 Wizard Interface

**Step-by-Step UI:**

```
┌─────────────────────────────────────────────────────┐
│  Create MCP Server                                  │
├─────────────────────────────────────────────────────┤
│  Step 1 of 4: Choose Template                       │
│                                                      │
│  ○ PostgreSQL Database                              │
│    Expose database tables as MCP resources          │
│                                                      │
│  ○ REST API Wrapper                                 │
│    Wrap any REST API as MCP server                  │
│                                                      │
│  ○ File System Access                               │
│    Access local files and directories               │
│                                                      │
│  [Cancel]                           [Next]          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Create MCP Server                                  │
├─────────────────────────────────────────────────────┤
│  Step 2 of 4: Configure                             │
│                                                      │
│  Server Name:                                       │
│  [postgres-myapp____________]                       │
│                                                      │
│  Database Host:                                     │
│  [localhost_________________]                       │
│                                                      │
│  Database Name:                                     │
│  [myapp_____________________]                       │
│                                                      │
│  Username:                                          │
│  [admin_____________________]                       │
│                                                      │
│  Tables (comma-separated, or leave blank for all):  │
│  [users, orders_____________]                       │
│                                                      │
│  ☑ Read-only access                                 │
│                                                      │
│  [Back]                             [Next]          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Create MCP Server                                  │
├─────────────────────────────────────────────────────┤
│  Step 3 of 4: Review                                │
│                                                      │
│  Template: PostgreSQL Database                      │
│  Name: mcp-server-postgres-myapp                    │
│                                                      │
│  Configuration:                                     │
│    Database: myapp on localhost:5432                │
│    User: admin                                      │
│    Tables: users, orders                            │
│    Access: Read-only                                │
│                                                      │
│  Will generate:                                     │
│    ✓ src/index.ts                                   │
│    ✓ src/resources.ts (2 tables)                    │
│    ✓ src/tools.ts (3 tools)                         │
│    ✓ tests/ (5 test files)                          │
│    ✓ README.md with setup instructions              │
│    ✓ package.json                                   │
│                                                      │
│  [Back]                             [Generate]      │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Create MCP Server                                  │
├─────────────────────────────────────────────────────┤
│  Step 4 of 4: Generating...                         │
│                                                      │
│  ✓ Parsing configuration                            │
│  ✓ Loading template                                 │
│  ✓ Generating TypeScript code                       │
│  ✓ Creating tests                                   │
│  ✓ Writing files                                    │
│  ✓ Validating output                                │
│                                                      │
│  Success! MCP server created.                       │
│                                                      │
│  📁 Location: /Users/you/projects/mcp-server-...    │
│                                                      │
│  Next steps:                                        │
│  1. cd mcp-server-postgres-myapp                    │
│  2. cp .env.example .env && edit .env               │
│  3. npm install                                     │
│  4. npm run build                                   │
│  5. Add to Claude Desktop (see README.md)           │
│                                                      │
│  [Open Folder]  [View README]  [Done]               │
└─────────────────────────────────────────────────────┘
```

### 4.3 Menu Integration

**File Menu:**

```
File
├── New
│   ├── File
│   ├── Folder
│   └── MCP Server...          ← NEW
├── Open Folder...
└── ...
```

**Tools Menu:**

```
Tools
├── Generate Code
├── Run Tests
├── MCP Server                  ← NEW
│   ├── Create New Server...
│   ├── Validate Existing Server
│   └── Test MCP Server
└── ...
```

---

## 5. Data Structures

### 5.1 Template Definition Schema

```rust
// src-tauri/src/mcp/types.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPTemplate {
    pub id: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub author: String,
    pub files: Vec<TemplateFile>,
    pub dependencies: HashMap<String, String>,
    pub dev_dependencies: HashMap<String, String>,
    pub environment_vars: Vec<EnvironmentVariable>,
    pub prompts: Vec<ConfigPrompt>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateFile {
    pub path: String,
    pub template: String,
    pub required: bool,
    pub executable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentVariable {
    pub name: String,
    pub description: String,
    pub required: bool,
    pub default: Option<String>,
    pub secret: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigPrompt {
    pub key: String,
    pub prompt: String,
    pub prompt_type: PromptType,
    pub required: bool,
    pub default: Option<String>,
    pub validation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptType {
    Text,
    Number,
    Boolean,
    Select,
    MultiSelect,
}
```

### 5.2 Generation Request/Response

```rust
// Request from UI to backend
#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateMCPRequest {
    pub template_id: String,
    pub name: String,
    pub output_path: String,
    pub config: HashMap<String, serde_json::Value>,
}

// Response from backend to UI
#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateMCPResponse {
    pub success: bool,
    pub output_path: String,
    pub files_created: Vec<String>,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub next_steps: Vec<String>,
}

// Progress updates during generation
#[derive(Debug, Serialize, Deserialize)]
pub struct GenerationProgress {
    pub step: String,
    pub progress: u8,        // 0-100
    pub message: String,
}
```

---

## 6. Implementation Details

### 6.1 File Structure

```
src-tauri/
├── src/
│   ├── mcp/
│   │   ├── mod.rs                  # Module exports
│   │   ├── types.rs                # Data structures
│   │   ├── parser.rs               # Intent parser
│   │   ├── template_engine.rs     # Template loading/rendering
│   │   ├── generator.rs            # Code generation
│   │   ├── validator.rs            # Validation logic
│   │   ├── commands.rs             # Tauri commands
│   │   └── templates/              # Built-in templates
│   │       ├── postgres/
│   │       │   ├── template.json
│   │       │   ├── index.ts.hbs
│   │       │   ├── resources.ts.hbs
│   │       │   └── ...
│   │       ├── api/
│   │       │   └── ...
│   │       └── filesystem/
│   │           └── ...
│   └── main.rs

src-ui/
├── components/
│   ├── MCPWizard.tsx              # Wizard UI
│   ├── MCPTemplateCard.tsx        # Template selection
│   └── MCPProgress.tsx            # Progress indicator
├── api/
│   └── mcp.ts                     # API calls to backend
└── stores/
    └── mcpStore.ts                # State management
```

### 6.2 Tauri Commands

```rust
// src-tauri/src/mcp/commands.rs

#[tauri::command]
pub async fn list_mcp_templates() -> Result<Vec<MCPTemplate>, String> {
    // Return list of available templates
}

#[tauri::command]
pub async fn get_mcp_template(template_id: String) -> Result<MCPTemplate, String> {
    // Get specific template details
}

#[tauri::command]
pub async fn generate_mcp_server(
    request: GenerateMCPRequest,
) -> Result<GenerateMCPResponse, String> {
    // Main generation command
}

#[tauri::command]
pub async fn validate_mcp_config(
    template_id: String,
    config: HashMap<String, serde_json::Value>,
) -> Result<ValidationResult, String> {
    // Validate configuration before generation
}

#[tauri::command]
pub async fn parse_mcp_intent(
    description: String,
    llm_api_key: String,
) -> Result<MCPIntent, String> {
    // Parse natural language to intent
}
```

### 6.3 Template Rendering (Handlebars)

**Example: resources.ts.hbs**

```typescript
// Generated resources.ts for {{name}}
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { z } from 'zod';
{{#if useDatabase}}
import { pool } from './db.js';
{{/if}}

export function registerResources(server: Server) {
  {{#each resources}}
  // Resource: {{this.name}}
  server.setRequestHandler('resources/list', async () => {
    return {
      resources: [
        {
          uri: '{{../name}}://{{this.name}}',
          mimeType: 'application/json',
          name: '{{this.displayName}}',
          description: '{{this.description}}',
        },
      ],
    };
  });

  server.setRequestHandler('resources/read', async (request) => {
    const uri = z.string().parse(request.params.uri);

    if (uri === '{{../name}}://{{this.name}}') {
      {{#if this.query}}
      const result = await pool.query('{{this.query}}');
      return {
        contents: [{
          uri,
          mimeType: 'application/json',
          text: JSON.stringify(result.rows, null, 2),
        }],
      };
      {{else}}
      return {
        contents: [{
          uri,
          mimeType: 'application/json',
          text: JSON.stringify({{this.data}}, null, 2),
        }],
      };
      {{/if}}
    }

    throw new Error(`Unknown resource: ${uri}`);
  });
  {{/each}}
}
```

### 6.4 LLM Integration for Intent Parsing

```rust
// src-tauri/src/mcp/parser.rs

use serde_json::json;

pub async fn parse_intent(
    description: &str,
    api_key: &str,
) -> Result<MCPIntent, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();

    let prompt = format!(
        r#"Parse the following MCP server request and extract structured information.

User request: "{}"

Return JSON with:
{{
  "template_type": "database" | "api" | "filesystem",
  "name": "server-name",
  "resources": [{{ "name": "...", "description": "..." }}],
  "tools": [{{ "name": "...", "parameters": [...] }}],
  "auth_type": "none" | "apiKey" | "oauth",
  "config": {{ template-specific configuration }}
}}
"#,
        description
    );

    let response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": prompt
            }]
        }))
        .send()
        .await?;

    let result: serde_json::Value = response.json().await?;
    let content = result["content"][0]["text"].as_str().unwrap();

    // Parse JSON response
    let intent: MCPIntent = serde_json::from_str(content)?;

    Ok(intent)
}
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

**Template Engine:**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_template() {
        let template = load_template("postgres").unwrap();
        assert_eq!(template.id, "postgres");
        assert!(template.files.len() > 0);
    }

    #[test]
    fn test_render_template() {
        let template = load_template("postgres").unwrap();
        let config = HashMap::from([
            ("name".to_string(), json!("test-db")),
            ("tables".to_string(), json!(["users", "orders"])),
        ]);

        let result = render_template(&template, &config).unwrap();
        assert!(result.contains("test-db"));
    }
}
```

**Code Generator:**

```rust
#[test]
fn test_generate_mcp_server() {
    let request = GenerateMCPRequest {
        template_id: "postgres".to_string(),
        name: "test-server".to_string(),
        output_path: "/tmp/test".to_string(),
        config: HashMap::new(),
    };

    let response = generate_mcp_server(request).await.unwrap();
    assert!(response.success);
    assert!(response.files_created.len() > 0);
}
```

### 7.2 Integration Tests

**End-to-End Generation:**

```rust
#[tokio::test]
async fn test_full_generation_flow() {
    // 1. Parse intent
    let intent = parse_intent(
        "Create MCP for PostgreSQL database",
        "test-api-key"
    ).await.unwrap();

    // 2. Generate server
    let request = GenerateMCPRequest {
        template_id: intent.template_type,
        name: intent.name,
        output_path: "/tmp/test-mcp".to_string(),
        config: intent.config,
    };

    let response = generate_mcp_server(request).await.unwrap();
    assert!(response.success);

    // 3. Validate generated files
    let index_path = format!("{}/src/index.ts", response.output_path);
    assert!(std::path::Path::new(&index_path).exists());

    // 4. Validate TypeScript compiles
    let output = std::process::Command::new("npx")
        .args(&["tsc", "--noEmit"])
        .current_dir(&response.output_path)
        .output()
        .unwrap();
    assert!(output.status.success());
}
```

### 7.3 Generated Code Tests

**Every generated MCP server includes:**

```typescript
// tests/server.test.ts
import { describe, it, expect } from 'vitest';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';

describe('MCP Server', () => {
  it('should initialize', async () => {
    const server = new Server(/* ... */);
    expect(server).toBeDefined();
  });

  it('should list resources', async () => {
    const server = new Server(/* ... */);
    const response = await server.request({
      method: 'resources/list',
    });
    expect(response.resources).toBeInstanceOf(Array);
  });

  it('should read resource', async () => {
    const server = new Server(/* ... */);
    const response = await server.request({
      method: 'resources/read',
      params: { uri: 'test://resource' },
    });
    expect(response.contents).toBeInstanceOf(Array);
  });
});
```

---

## 8. Security Considerations

### 8.1 Input Validation

**User Input Sanitization:**

- Validate server names (alphanumeric + dash only)
- Sanitize file paths (prevent path traversal)
- Validate database connection strings
- Check API endpoints for valid URLs

**Code Generation Safety:**

- Escape special characters in templates
- Prevent code injection in generated code
- Validate TypeScript syntax before writing files

### 8.2 Secrets Management

**Best Practices:**

- Never hardcode secrets in generated code
- Always use environment variables
- Generate `.env.example` with placeholders
- Add `.env` to `.gitignore`
- Warn users about secret exposure

**Generated .env.example:**

```bash
# Database Configuration
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=myapp
DATABASE_USER=admin
DATABASE_PASSWORD=your_password_here

# API Configuration (if applicable)
API_KEY=your_api_key_here
API_BASE_URL=https://api.example.com

# Security
MAX_ROWS=1000
READ_ONLY=true
```

### 8.3 Dependency Security

**Validation:**

- Check npm packages exist
- Verify version compatibility
- Scan for known vulnerabilities (npm audit)
- Warn about deprecated packages

**Generated package.json:**

```json
{
  "dependencies": {
    "@modelcontextprotocol/sdk": "^0.5.0",
    "zod": "^3.22.0"
  },
  "scripts": {
    "audit": "npm audit",
    "audit:fix": "npm audit fix"
  }
}
```

---

## 9. Error Handling

### 9.1 Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum MCPError {
    #[error("Template not found: {0}")]
    TemplateNotFound(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Code generation failed: {0}")]
    GenerationFailed(String),

    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    #[error("File system error: {0}")]
    FileSystemError(#[from] std::io::Error),

    #[error("Template rendering error: {0}")]
    RenderError(String),

    #[error("LLM API error: {0}")]
    LLMError(String),
}
```

### 9.2 Error Recovery

**Graceful Degradation:**

1. If LLM intent parsing fails → Fall back to wizard
2. If template rendering fails → Show detailed error
3. If file write fails → Clean up partial generation
4. If validation fails → Offer to generate anyway with warnings

**User-Friendly Messages:**

```rust
impl MCPError {
    pub fn user_message(&self) -> String {
        match self {
            MCPError::TemplateNotFound(id) => {
                format!(
                    "Template '{}' not found. Available templates: postgres, api, filesystem",
                    id
                )
            }
            MCPError::InvalidConfig(msg) => {
                format!("Configuration error: {}. Please check your inputs.", msg)
            }
            MCPError::GenerationFailed(msg) => {
                format!(
                    "Failed to generate MCP server: {}. Please try again or report this issue.",
                    msg
                )
            }
            // ... other error messages
        }
    }
}
```

---

## 10. Performance Requirements

### 10.1 Generation Speed

**Target Performance:**

- Intent parsing: <2 seconds (LLM call)
- Template loading: <100ms
- Code generation: <500ms
- File writing: <200ms
- Validation: <1 second
- **Total: <5 seconds** (under 60 seconds for complex servers)

**Optimization Strategies:**

- Cache loaded templates in memory
- Parallel file writing
- Async LLM calls
- Lazy validation (optional step)

### 10.2 Resource Usage

**Memory:**

- Template cache: <10MB
- Generation process: <50MB per server
- Total: <100MB overhead

**Disk:**

- Templates: <5MB
- Generated server: 500KB - 2MB (before npm install)
- With dependencies: 20-50MB (after npm install)

---

## 11. Documentation

### 11.1 Generated README.md

**Template:**

```markdown
# {{name}} - MCP Server

Generated by Yantra on {{date}}

## Description

{{description}}

## Setup

### 1. Install Dependencies

\`\`\`bash
npm install
\`\`\`

### 2. Configure Environment

Copy the example environment file and edit with your values:

\`\`\`bash
cp .env.example .env
\`\`\`

Edit `.env`:
{{#each envVars}}

- `{{this.name}}`: {{this.description}}
  {{/each}}

### 3. Build

\`\`\`bash
npm run build
\`\`\`

### 4. Test

\`\`\`bash
npm test
\`\`\`

## Usage

### With Claude Desktop

Add to your Claude Desktop configuration (`~/.config/claude/config.json`):

\`\`\`json
{
"mcpServers": {
"{{name}}": {
"command": "node",
"args": ["{{absolutePath}}/dist/index.js"]
}
}
}
\`\`\`

### Standalone

\`\`\`bash
node dist/index.js
\`\`\`

## Resources

{{#each resources}}

- **{{this.name}}**: {{this.description}}
  - URI: `{{../name}}://{{this.name}}`
    {{/each}}

## Tools

{{#each tools}}

- **{{this.name}}**: {{this.description}}
  - Parameters: {{#each this.parameters}}`{{this.name}}` ({{this.type}}){{#unless @last}}, {{/unless}}{{/each}}
    {{/each}}

## Development

### Run in Development

\`\`\`bash
npm run dev
\`\`\`

### Run Tests

\`\`\`bash
npm test
npm run test:watch
npm run test:coverage
\`\`\`

### Lint

\`\`\`bash
npm run lint
npm run lint:fix
\`\`\`

## Security

{{#if readOnly}}
⚠️ This server is configured in **read-only mode**. No write operations are allowed.
{{/if}}

- Always use environment variables for secrets
- Never commit `.env` file to version control
- Review generated code before deployment
- Run `npm audit` regularly

## Troubleshooting

### Connection Errors

{{troubleshooting}}

### Permission Errors

Ensure the user has appropriate permissions to access resources.

## License

{{license}}

## Generated By

Yantra MCP Creation Engine
https://github.com/yourusername/yantra
```

### 11.2 Inline Code Comments

**Generated code includes:**

- File headers with generation timestamp
- Function documentation
- Complex logic explanations
- Security notes
- TODO markers for user customization

---

## 12. Success Metrics

### 12.1 MVP Success Criteria

**Technical Metrics:**

- ✅ Generate working MCP server in <60 seconds
- ✅ 90%+ generated servers work without editing
- ✅ Pass MCP protocol validation
- ✅ Zero critical security vulnerabilities
- ✅ TypeScript compiles without errors

**User Metrics:**

- ✅ User completes setup in <5 minutes
- ✅ User successfully connects to Claude Desktop
- ✅ User can perform basic operations immediately
- ✅ User satisfaction: 4/5 stars or higher

### 12.2 Quality Gates

**Before Ship:**

1. All 3 templates generate successfully
2. Generated code passes all tests
3. Security audit clean
4. Documentation complete
5. Manual testing with real databases/APIs

---

## 13. Future Enhancements (Post-MVP)

### 13.1 Phase 2 Features

1. **Additional Templates:**
   - MongoDB
   - GraphQL API
   - Slack/Discord integration
   - Git repository access
   - Docker/Kubernetes management

2. **Visual Designer:**
   - Drag-drop resources/tools
   - Interactive configuration
   - Live preview

3. **Template Marketplace:**
   - Community templates
   - Rating/reviews
   - One-click install

4. **Deployment Automation:**
   - Railway integration
   - Docker build
   - GitHub Actions CI/CD

5. **Advanced Features:**
   - Custom middleware
   - Rate limiting configuration
   - Caching strategies
   - Webhook support

### 13.2 Phase 3 Features

1. **Yantra Cloud:**
   - Hosted MCP servers
   - One-click deploy
   - Managed infrastructure

2. **Collaboration:**
   - Share templates with team
   - Version control
   - Access control

3. **Monitoring:**
   - Usage analytics
   - Error tracking
   - Performance metrics

---

## 14. Implementation Plan

### 14.1 Week 1: Foundation

**Day 1-2: Setup**

- Create `mcp/` module structure
- Define data structures
- Setup Handlebars integration

**Day 3-4: Template Engine**

- Implement template loading
- Build rendering engine
- Create first template (PostgreSQL)

**Day 5: Validation**

- Implement basic validation
- Add security checks

### 14.2 Week 2: Generation & UI

**Day 1-2: Code Generator**

- Implement file generation
- Add TypeScript formatting
- Test generation pipeline

**Day 3-4: UI Components**

- Build wizard interface
- Add chat integration
- Create progress indicator

**Day 5: Integration**

- Connect UI to backend
- Add Tauri commands
- End-to-end testing

### 14.3 Week 3: Polish & Ship

**Day 1-2: Additional Templates**

- REST API template
- File System template
- Test all templates

**Day 3: Documentation**

- Write user guide
- Generate API docs
- Create video tutorial

**Day 4: Testing**

- Integration tests
- User acceptance testing
- Bug fixes

**Day 5: Release**

- Package for distribution
- Publish release notes
- Monitor feedback

---

## 15. Dependencies

### 15.1 Rust Crates

```toml
[dependencies]
# Existing Yantra dependencies
tauri = "1.5"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1", features = ["full"] }

# New dependencies for MCP Engine
handlebars = "5.0"          # Template rendering
reqwest = { version = "0.11", features = ["json"] }  # LLM API calls
thiserror = "1.0"           # Error handling
validator = "0.16"          # Input validation
glob = "0.3"                # File pattern matching
```

### 15.2 NPM Packages (Generated Servers)

```json
{
  "dependencies": {
    "@modelcontextprotocol/sdk": "^0.5.0",
    "zod": "^3.22.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "typescript": "^5.3.0",
    "vitest": "^1.0.0",
    "prettier": "^3.0.0",
    "eslint": "^8.0.0"
  }
}
```

---

## 16. Risks & Mitigations

### 16.1 Technical Risks

| Risk                          | Impact | Probability | Mitigation                               |
| ----------------------------- | ------ | ----------- | ---------------------------------------- |
| MCP spec changes              | High   | Medium      | Follow Anthropic repo, version templates |
| LLM API failures              | Medium | Low         | Fallback to wizard mode                  |
| Template complexity           | Medium | Medium      | Start simple, iterate                    |
| Security vulnerabilities      | High   | Low         | Security audit, input validation         |
| TypeScript compilation errors | Medium | Low         | Extensive testing, linting               |

### 16.2 User Experience Risks

| Risk                        | Impact | Probability | Mitigation                    |
| --------------------------- | ------ | ----------- | ----------------------------- |
| Generated code doesn't work | High   | Medium      | Extensive testing, validation |
| Setup too complex           | Medium | High        | Clear docs, wizard UI         |
| Limited template selection  | Low    | High        | Start with common use cases   |
| Users can't customize       | Medium | Low         | Clear code, allow editing     |

---

## 17. Open Questions

1. **Template Format:** Use JSON + Handlebars or custom DSL?
   - **Decision:** JSON + Handlebars (familiar, flexible)

2. **LLM Provider:** Support multiple LLMs or just Claude?
   - **Decision:** Support Claude + GPT-4 (user choice)

3. **Template Storage:** Bundle with app or download on-demand?
   - **Decision:** Bundle 3 core templates, allow custom later

4. **Validation Strictness:** Block generation or just warn?
   - **Decision:** Warn but allow generation (give user control)

5. **npm install:** Auto-run or let user run manually?
   - **Decision:** Manual (explain in docs, avoid permissions issues)

---

## 18. Acceptance Criteria

### 18.1 Definition of Done

**Feature is complete when:**

- ✅ All 3 templates generate working code
- ✅ Generated code passes TypeScript compilation
- ✅ Generated code passes all tests
- ✅ UI allows both chat and wizard creation
- ✅ Documentation is complete (inline + README)
- ✅ Security validation passes
- ✅ Manual testing successful with real databases/APIs
- ✅ User can integrate with Claude Desktop in <5 minutes

### 18.2 User Stories

**As a developer, I want to:**

1. Create an MCP server from natural language description
2. Choose from pre-built templates
3. Configure server through a simple wizard
4. Get production-ready code with tests
5. Integrate with Claude Desktop immediately
6. Customize generated code if needed
7. Deploy locally or to the cloud

**As Yantra, we want to:**

1. Make MCP adoption frictionless
2. Establish first-mover advantage in MCP tooling
3. Build ecosystem around Yantra
4. Enable users to extend AI capabilities
5. Differentiate from competitors

---

## 19. Glossary

- **MCP**: Model Context Protocol - Anthropic's standard for AI context
- **Resource**: Data that can be read by AI (files, database tables, API responses)
- **Tool**: Operation that AI can perform (query, search, create)
- **Template**: Pre-built MCP server scaffold
- **Intent**: Structured representation of user's natural language request
- **Handlebars**: Template engine for code generation
- **Claude Desktop**: Anthropic's desktop app that supports MCP
- **stdio**: Standard input/output - how Claude Desktop communicates with MCP servers

---

## 20. Conclusion

The MCP Creation Engine is a strategic feature that positions Yantra as the premier tool for MCP server development. By integrating with Yantra Platform Core and the Unified Tools Interface, generated MCPs work seamlessly across all Yantra products.

**Key Takeaways:**

- ✅ Generates locally, deploys to cloud (Platform Core)
- ✅ 4 core templates (Database, REST API, File System, Custom)
- ✅ Natural language + wizard interfaces
- ✅ Production-ready code implementing UTI
- ✅ Automatic registration in Tool Registry
- ✅ MCPs available in Flow and Data immediately
- ✅ Ship in 2-3 weeks

**Note:** Browser automation (CDP) is a built-in IDE feature per Specifications.md, not an MCP template.

**Next Steps:**

1. Review and approve this specification
2. Begin implementation (Week 1: Foundation)
3. Iterate based on early user feedback
4. Plan Phase 2 features after MVP validation

---

**Document Status:** Ready for Implementation  
**Approval Required From:** Engineering Lead, Product Owner  
**Expected Ship Date:** December 30, 2025 (3 weeks from spec date)
