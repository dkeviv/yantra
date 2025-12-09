# Yantra Platform Core - Technical Specification

**Version:** 1.0  
**Date:** December 9, 2025  
**Status:** Specification - Ready for Implementation  
**Phase:** Foundation for All Products

---

## 1. Executive Summary

### 1.1 Vision

The Yantra Platform Core is the foundational infrastructure that powers all Yantra products (Develop, Flow, and Data). It provides a unified tools interface, multi-tenant architecture, AI agent orchestration, and MCP runtime that enables seamless integration and automation across the entire platform.

### 1.2 Core Principles

**Create Once, Use Everywhere:**

- Developers create MCPs in Yantra Develop
- Business users use them in Yantra Flow
- Analysts query them in Yantra Data
- One tool, three products, zero duplication

**Unified Architecture:**

- Single tool interface (UTI) for all integrations
- Consistent permissions, audit logs, and monitoring
- Shared AI agent orchestration
- Multi-tenant by design

**Enterprise-First:**

- SSO/SAML authentication
- Role-based access control (RBAC)
- Secrets management with encryption
- Audit logging and compliance
- 99.9% SLA

---

## 2. Platform Architecture

### 2.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    YANTRA PLATFORM                            │
│                                                               │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐        │
│  │  DEVELOP   │    │    FLOW    │    │    DATA    │        │
│  │ (Desktop)  │    │  (Cloud)   │    │  (Cloud)   │        │
│  │            │    │            │    │            │        │
│  │ • IDE      │    │ • Workflows│    │ • Analytics│        │
│  │ • CDP      │    │ • Apps     │    │ • Virtual  │        │
│  │ • MCP Gen  │    │ • RPA      │    │   Schema   │        │
│  └──────┬─────┘    └──────┬─────┘    └──────┬─────┘        │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            │                                 │
│  ┌─────────────────────────▼──────────────────────────────┐ │
│  │     YANTRA PLATFORM CORE                               │ │
│  │                                                         │ │
│  │  ┌───────────────────────────────────────────────────┐│ │
│  │  │  Unified Tools Interface (UTI)                    ││ │
│  │  │  • Tool Registry (MCP, Built-in, Custom)          ││ │
│  │  │  • Tool Discovery & Capabilities                  ││ │
│  │  │  • Execution Engine                               ││ │
│  │  │  • Permission Enforcement                         ││ │
│  │  └───────────────────────────────────────────────────┘│ │
│  │                                                         │ │
│  │  ┌───────────────────────────────────────────────────┐│ │
│  │  │  AI Agent Orchestration                           ││ │
│  │  │  • Intent Understanding (LLM)                     ││ │
│  │  │  • Task Planning & Decomposition                  ││ │
│  │  │  • Tool Selection & Routing                       ││ │
│  │  │  • Context Management                             ││ │
│  │  │  • Error Recovery                                 ││ │
│  │  └───────────────────────────────────────────────────┘│ │
│  │                                                         │ │
│  │  ┌───────────────────────────────────────────────────┐│ │
│  │  │  Multi-Tenant Infrastructure                      ││ │
│  │  │  • Organization Management                        ││ │
│  │  │  • User Auth (SSO/SAML/OAuth)                     ││ │
│  │  │  • RBAC (Role-Based Access Control)               ││ │
│  │  │  • Tenant Isolation (Network + Data)              ││ │
│  │  └───────────────────────────────────────────────────┘│ │
│  │                                                         │ │
│  │  ┌───────────────────────────────────────────────────┐│ │
│  │  │  MCP Runtime (Kubernetes)                         ││ │
│  │  │  • MCP Server Hosting                             ││ │
│  │  │  • Auto-Scaling & Load Balancing                  ││ │
│  │  │  • Health Monitoring & Auto-Restart               ││ │
│  │  │  • Version Management (Blue/Green Deploy)         ││ │
│  │  └───────────────────────────────────────────────────┘│ │
│  │                                                         │ │
│  │  ┌───────────────────────────────────────────────────┐│ │
│  │  │  Security & Compliance                            ││ │
│  │  │  • Secrets Management (HashiCorp Vault)           ││ │
│  │  │  • Encryption (At Rest + In Transit)              ││ │
│  │  │  • Audit Logging (Immutable)                      ││ │
│  │  │  • Compliance Reports (SOC2, HIPAA, GDPR)         ││ │
│  │  └───────────────────────────────────────────────────┘│ │
│  │                                                         │ │
│  │  ┌───────────────────────────────────────────────────┐│ │
│  │  │  Monitoring & Observability                       ││ │
│  │  │  • Metrics (Prometheus)                           ││ │
│  │  │  • Logs (ELK Stack)                               ││ │
│  │  │  • Tracing (Jaeger)                               ││ │
│  │  │  • Alerting (PagerDuty)                           ││ │
│  │  └───────────────────────────────────────────────────┘│ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Unified Tools Interface (UTI)

### 3.1 Core Concept

**The UTI is the heart of the platform.** All external integrations, built-in capabilities, and custom tools are exposed through a single, consistent interface.

### 3.2 Tool Interface Definition

```typescript
// Core Tool Interface
interface Tool {
  // Identity
  id: string; // Unique identifier: "mcp-stripe", "builtin-browser"
  name: string; // Human-readable: "Stripe Payments"
  description: string; // What the tool does
  version: string; // Semantic version: "1.2.3"

  // Classification
  type: ToolType; // 'mcp' | 'builtin' | 'custom'
  provider: string; // 'stripe', 'postgres', 'browser'
  category: ToolCategory; // 'payment', 'database', 'automation'

  // Capabilities
  capabilities: Capability[]; // ['read', 'write', 'execute']
  resources?: Resource[]; // Things this tool exposes (data sources)
  actions?: Action[]; // Things this tool can do (operations)

  // Metadata
  owner: string; // User/org who created it
  organization_id: string; // Tenant ID
  created_at: string;
  updated_at: string;

  // Access Control
  permissions: Permission[]; // Who can use this tool and how
  required_scopes: string[]; // OAuth scopes needed

  // Configuration
  config_schema: JSONSchema; // What config is needed
  secrets_schema: JSONSchema; // What secrets are needed

  // Runtime
  status: ToolStatus; // 'active' | 'inactive' | 'error'
  health: HealthStatus; // Last health check result

  // Usage
  usage_limits?: UsageLimits; // Rate limits, quotas
  pricing?: PricingInfo; // Cost per operation
}

// Tool Types
type ToolType = 'mcp' | 'builtin' | 'custom';

// Tool Categories
type ToolCategory =
  | 'payment' // Stripe, PayPal
  | 'database' // PostgreSQL, MongoDB
  | 'api' // REST, GraphQL
  | 'filesystem' // Local, S3, Google Drive
  | 'communication' // Email, Slack, SMS
  | 'crm' // Salesforce, HubSpot
  | 'analytics' // Google Analytics, Mixpanel
  | 'automation' // Browser, RPA
  | 'ai' // OpenAI, Anthropic
  | 'other';

// Capabilities
type Capability =
  | 'read' // Can read data
  | 'write' // Can write data
  | 'execute' // Can perform actions
  | 'stream' // Can stream data
  | 'subscribe'; // Can subscribe to events

// Resources (data sources exposed by tool)
interface Resource {
  uri: string; // "stripe://customers"
  name: string; // "Customers"
  description: string;
  mime_type: string; // "application/json"
  schema?: JSONSchema; // Data structure
}

// Actions (operations tool can perform)
interface Action {
  name: string; // "create_payment"
  description: string;
  parameters: Parameter[]; // Required/optional params
  returns: JSONSchema; // Return type
}

// Parameters
interface Parameter {
  name: string;
  type: string; // "string" | "number" | "boolean" | "object"
  required: boolean;
  description: string;
  default?: any;
  validation?: ValidationRule;
}
```

### 3.3 Tool Registry

```rust
// src-platform/src/core/tool_registry.rs

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use tokio::sync::RwLock;

pub struct ToolRegistry {
    tools: Arc<RwLock<HashMap<String, Arc<dyn Tool>>>>,
    metadata_store: Arc<MetadataStore>,
    permission_manager: Arc<PermissionManager>,
    audit_logger: Arc<AuditLogger>,
    usage_tracker: Arc<UsageTracker>,
}

impl ToolRegistry {
    pub fn new(
        metadata_store: Arc<MetadataStore>,
        permission_manager: Arc<PermissionManager>,
        audit_logger: Arc<AuditLogger>,
        usage_tracker: Arc<UsageTracker>,
    ) -> Self {
        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
            metadata_store,
            permission_manager,
            audit_logger,
            usage_tracker,
        }
    }

    /// Register a new tool
    pub async fn register(&self, tool: Arc<dyn Tool>) -> Result<(), RegistryError> {
        let tool_id = tool.id().to_string();

        // Discover capabilities
        let capabilities = tool.capabilities().await?;

        // Store metadata
        self.metadata_store.save(&tool_id, &capabilities).await?;

        // Add to registry
        let mut tools = self.tools.write().await;
        tools.insert(tool_id.clone(), tool);

        info!("Registered tool: {}", tool_id);
        Ok(())
    }

    /// Unregister a tool
    pub async fn unregister(&self, tool_id: &str) -> Result<(), RegistryError> {
        let mut tools = self.tools.write().await;
        tools.remove(tool_id);

        self.metadata_store.delete(tool_id).await?;

        info!("Unregistered tool: {}", tool_id);
        Ok(())
    }

    /// List tools (with filtering)
    pub async fn list(&self, filter: ToolFilter) -> Vec<ToolInfo> {
        let tools = self.tools.read().await;

        tools
            .values()
            .filter(|tool| self.matches_filter(tool, &filter))
            .map(|tool| ToolInfo::from(tool.as_ref()))
            .collect()
    }

    /// Get a specific tool
    pub async fn get(&self, tool_id: &str) -> Option<Arc<dyn Tool>> {
        let tools = self.tools.read().await;
        tools.get(tool_id).cloned()
    }

    /// Execute tool action
    pub async fn execute(
        &self,
        tool_id: &str,
        action: &str,
        params: serde_json::Value,
        context: &ExecutionContext,
    ) -> Result<ToolResult, ExecutionError> {
        // 1. Get tool
        let tool = self.get(tool_id).await
            .ok_or(ExecutionError::ToolNotFound(tool_id.to_string()))?;

        // 2. Check permissions
        self.permission_manager
            .check_permission(context, tool_id, action)
            .await?;

        // 3. Check usage limits
        self.usage_tracker
            .check_limits(context.organization_id, tool_id)
            .await?;

        // 4. Audit log (before execution)
        self.audit_logger.log_tool_execution(
            context.user_id,
            context.organization_id,
            tool_id,
            action,
            &params,
        ).await?;

        // 5. Execute
        let start = std::time::Instant::now();
        let result = tool.execute(action, params.clone()).await;
        let duration = start.elapsed();

        // 6. Record usage
        if result.is_ok() {
            self.usage_tracker.record(
                context.organization_id,
                tool_id,
                duration,
                result.as_ref().unwrap().bytes_transferred,
            ).await?;
        }

        // 7. Audit log (after execution)
        self.audit_logger.log_tool_result(
            context.user_id,
            tool_id,
            action,
            &result,
            duration,
        ).await?;

        result
    }

    /// Get resource from tool
    pub async fn get_resource(
        &self,
        tool_id: &str,
        uri: &str,
        context: &ExecutionContext,
    ) -> Result<ResourceContent, ExecutionError> {
        let tool = self.get(tool_id).await
            .ok_or(ExecutionError::ToolNotFound(tool_id.to_string()))?;

        // Check read permission
        self.permission_manager
            .check_permission(context, tool_id, "read")
            .await?;

        // Audit log
        self.audit_logger.log_resource_access(
            context.user_id,
            tool_id,
            uri,
        ).await?;

        tool.get_resource(uri).await
    }

    fn matches_filter(&self, tool: &Arc<dyn Tool>, filter: &ToolFilter) -> bool {
        // Filter by type
        if let Some(ref types) = filter.types {
            if !types.contains(&tool.tool_type()) {
                return false;
            }
        }

        // Filter by category
        if let Some(ref categories) = filter.categories {
            if !categories.contains(&tool.category()) {
                return false;
            }
        }

        // Filter by capabilities
        if let Some(ref caps) = filter.capabilities {
            let tool_caps = tool.capabilities_sync();
            if !caps.iter().all(|c| tool_caps.contains(c)) {
                return false;
            }
        }

        true
    }
}

/// Tool trait that all tools must implement
#[async_trait]
pub trait Tool: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn tool_type(&self) -> ToolType;
    fn category(&self) -> ToolCategory;
    fn version(&self) -> &str;

    async fn capabilities(&self) -> Result<ToolCapabilities, ToolError>;
    fn capabilities_sync(&self) -> Vec<Capability>;

    async fn execute(
        &self,
        action: &str,
        params: serde_json::Value,
    ) -> Result<ToolResult, ExecutionError>;

    async fn get_resource(&self, uri: &str) -> Result<ResourceContent, ExecutionError>;

    async fn health_check(&self) -> HealthStatus;
}

/// Filter for listing tools
#[derive(Debug, Clone)]
pub struct ToolFilter {
    pub types: Option<Vec<ToolType>>,
    pub categories: Option<Vec<ToolCategory>>,
    pub capabilities: Option<Vec<Capability>>,
    pub organization_id: Option<String>,
    pub owner: Option<String>,
    pub status: Option<Vec<ToolStatus>>,
}

/// Tool execution context
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub user_id: String,
    pub organization_id: String,
    pub session_id: String,
    pub ip_address: String,
    pub user_agent: String,
    pub product: Product, // Develop | Flow | Data
}

#[derive(Debug, Clone, Copy)]
pub enum Product {
    Develop,
    Flow,
    Data,
}

/// Tool execution result
#[derive(Debug)]
pub struct ToolResult {
    pub success: bool,
    pub data: serde_json::Value,
    pub metadata: ResultMetadata,
    pub bytes_transferred: u64,
}

#[derive(Debug)]
pub struct ResultMetadata {
    pub execution_time_ms: u64,
    pub cached: bool,
    pub source: String,
}
```

### 3.4 Tool Implementations

#### 3.4.1 MCP Tool (Wrapper)

```rust
// src-platform/src/core/tools/mcp_tool.rs

pub struct MCPTool {
    id: String,
    name: String,
    description: String,
    version: String,
    category: ToolCategory,

    // MCP-specific
    mcp_server_url: String,
    mcp_client: Arc<MCPClient>,
    config: MCPConfig,

    // Cached capabilities
    capabilities: RwLock<Option<ToolCapabilities>>,
}

impl MCPTool {
    pub fn new(
        id: String,
        name: String,
        mcp_server_url: String,
        config: MCPConfig,
    ) -> Self {
        let mcp_client = Arc::new(MCPClient::new(&mcp_server_url));

        Self {
            id,
            name,
            description: String::new(),
            version: "1.0.0".to_string(),
            category: ToolCategory::Other,
            mcp_server_url,
            mcp_client,
            config,
            capabilities: RwLock::new(None),
        }
    }
}

#[async_trait]
impl Tool for MCPTool {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn tool_type(&self) -> ToolType {
        ToolType::MCP
    }

    fn category(&self) -> ToolCategory {
        self.category
    }

    fn version(&self) -> &str {
        &self.version
    }

    async fn capabilities(&self) -> Result<ToolCapabilities, ToolError> {
        // Check cache
        {
            let cached = self.capabilities.read().await;
            if let Some(caps) = cached.as_ref() {
                return Ok(caps.clone());
            }
        }

        // Query MCP server for capabilities
        let resources = self.mcp_client.list_resources().await?;
        let tools = self.mcp_client.list_tools().await?;

        let capabilities = ToolCapabilities {
            resources: resources.into_iter().map(Resource::from).collect(),
            actions: tools.into_iter().map(Action::from).collect(),
        };

        // Cache
        {
            let mut cached = self.capabilities.write().await;
            *cached = Some(capabilities.clone());
        }

        Ok(capabilities)
    }

    fn capabilities_sync(&self) -> Vec<Capability> {
        vec![Capability::Read, Capability::Execute]
    }

    async fn execute(
        &self,
        action: &str,
        params: serde_json::Value,
    ) -> Result<ToolResult, ExecutionError> {
        let start = std::time::Instant::now();

        // Call MCP server
        let response = self.mcp_client
            .call_tool(action, params)
            .await
            .map_err(|e| ExecutionError::MCPError(e.to_string()))?;

        let duration = start.elapsed();

        Ok(ToolResult {
            success: true,
            data: response.content,
            metadata: ResultMetadata {
                execution_time_ms: duration.as_millis() as u64,
                cached: false,
                source: self.mcp_server_url.clone(),
            },
            bytes_transferred: response.content.to_string().len() as u64,
        })
    }

    async fn get_resource(&self, uri: &str) -> Result<ResourceContent, ExecutionError> {
        let response = self.mcp_client
            .read_resource(uri)
            .await
            .map_err(|e| ExecutionError::MCPError(e.to_string()))?;

        Ok(ResourceContent {
            uri: uri.to_string(),
            mime_type: response.mime_type,
            content: response.text,
        })
    }

    async fn health_check(&self) -> HealthStatus {
        match self.mcp_client.ping().await {
            Ok(_) => HealthStatus::Healthy,
            Err(e) => HealthStatus::Unhealthy {
                reason: e.to_string(),
            },
        }
    }
}
```

#### 3.4.2 Built-in Browser Tool

```rust
// src-platform/src/core/tools/browser_tool.rs

pub struct BrowserTool {
    id: String,
    cdp_client: Arc<CDPClient>,
}

impl BrowserTool {
    pub fn new() -> Self {
        Self {
            id: "builtin-browser".to_string(),
            cdp_client: Arc::new(CDPClient::new()),
        }
    }
}

#[async_trait]
impl Tool for BrowserTool {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        "Browser Automation"
    }

    fn description(&self) -> &str {
        "Automate web browsers using Chrome DevTools Protocol"
    }

    fn tool_type(&self) -> ToolType {
        ToolType::Builtin
    }

    fn category(&self) -> ToolCategory {
        ToolCategory::Automation
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    async fn capabilities(&self) -> Result<ToolCapabilities, ToolError> {
        Ok(ToolCapabilities {
            resources: vec![],
            actions: vec![
                Action {
                    name: "navigate".to_string(),
                    description: "Navigate to URL".to_string(),
                    parameters: vec![
                        Parameter {
                            name: "url".to_string(),
                            param_type: "string".to_string(),
                            required: true,
                            description: "URL to navigate to".to_string(),
                            default: None,
                        }
                    ],
                },
                Action {
                    name: "click".to_string(),
                    description: "Click an element".to_string(),
                    parameters: vec![
                        Parameter {
                            name: "selector".to_string(),
                            param_type: "string".to_string(),
                            required: true,
                            description: "CSS selector".to_string(),
                            default: None,
                        }
                    ],
                },
                Action {
                    name: "extract".to_string(),
                    description: "Extract data from page".to_string(),
                    parameters: vec![
                        Parameter {
                            name: "selectors".to_string(),
                            param_type: "object".to_string(),
                            required: true,
                            description: "Map of field names to CSS selectors".to_string(),
                            default: None,
                        }
                    ],
                },
                Action {
                    name: "screenshot".to_string(),
                    description: "Take screenshot".to_string(),
                    parameters: vec![],
                },
            ],
        })
    }

    fn capabilities_sync(&self) -> Vec<Capability> {
        vec![Capability::Execute]
    }

    async fn execute(
        &self,
        action: &str,
        params: serde_json::Value,
    ) -> Result<ToolResult, ExecutionError> {
        match action {
            "navigate" => self.navigate(params).await,
            "click" => self.click(params).await,
            "extract" => self.extract(params).await,
            "screenshot" => self.screenshot(params).await,
            _ => Err(ExecutionError::UnknownAction(action.to_string())),
        }
    }

    async fn get_resource(&self, _uri: &str) -> Result<ResourceContent, ExecutionError> {
        Err(ExecutionError::NotSupported("Browser tool doesn't expose resources".to_string()))
    }

    async fn health_check(&self) -> HealthStatus {
        HealthStatus::Healthy
    }
}

impl BrowserTool {
    async fn navigate(&self, params: serde_json::Value) -> Result<ToolResult, ExecutionError> {
        let url = params["url"].as_str()
            .ok_or(ExecutionError::InvalidParams("Missing 'url'".to_string()))?;

        self.cdp_client.navigate(url).await?;

        Ok(ToolResult {
            success: true,
            data: serde_json::json!({ "url": url }),
            metadata: ResultMetadata {
                execution_time_ms: 0,
                cached: false,
                source: "browser".to_string(),
            },
            bytes_transferred: 0,
        })
    }

    async fn extract(&self, params: serde_json::Value) -> Result<ToolResult, ExecutionError> {
        let selectors = params["selectors"].as_object()
            .ok_or(ExecutionError::InvalidParams("Missing 'selectors'".to_string()))?;

        let mut extracted = serde_json::Map::new();

        for (key, selector_value) in selectors {
            let selector = selector_value.as_str()
                .ok_or(ExecutionError::InvalidParams("Selector must be string".to_string()))?;

            let text = self.cdp_client.extract_text(selector).await?;
            extracted.insert(key.clone(), serde_json::Value::String(text));
        }

        Ok(ToolResult {
            success: true,
            data: serde_json::Value::Object(extracted),
            metadata: ResultMetadata {
                execution_time_ms: 0,
                cached: false,
                source: "browser".to_string(),
            },
            bytes_transferred: 0,
        })
    }

    // ... other methods
}
```

---

## 4. AI Agent Orchestration

### 4.1 Agent Architecture

```rust
// src-platform/src/core/agent.rs

pub struct Agent {
    tool_registry: Arc<ToolRegistry>,
    llm_client: Arc<LLMClient>,
    context_manager: Arc<ContextManager>,
    planner: Arc<TaskPlanner>,
}

impl Agent {
    pub async fn handle_request(
        &mut self,
        user_input: &str,
        context: &ExecutionContext,
    ) -> Result<AgentResponse, AgentError> {
        // 1. Update context with user input
        self.context_manager.add_user_message(user_input).await?;

        // 2. Understand intent
        let intent = self.parse_intent(user_input, context).await?;

        // 3. Select tools
        let tools = self.select_tools(&intent, context).await?;

        // 4. Create execution plan
        let plan = self.planner.create_plan(&intent, &tools).await?;

        // 5. Execute plan
        let results = self.execute_plan(&plan, context).await?;

        // 6. Generate response
        let response = self.generate_response(&intent, &results).await?;

        // 7. Update context
        self.context_manager.add_assistant_message(&response).await?;

        Ok(AgentResponse {
            message: response.message,
            tool_calls: response.tool_calls,
            data: results,
            suggestions: response.suggestions,
        })
    }

    async fn parse_intent(
        &self,
        user_input: &str,
        context: &ExecutionContext,
    ) -> Result<Intent, AgentError> {
        let conversation_history = self.context_manager.get_recent_messages(10).await?;

        let prompt = format!(
            r#"Parse the user's intent and extract:
1. What they want to do (action)
2. What data/resources they need
3. Any filters or conditions
4. Expected output format

User input: "{}"

Recent context: {:?}

Return JSON:
{{
  "action": "query" | "create" | "update" | "delete" | "analyze",
  "entities": ["customer", "order", "payment"],
  "filters": [{{"field": "status", "operator": "eq", "value": "active"}}],
  "output_format": "table" | "chart" | "text",
  "required_capabilities": ["read", "execute"],
  "confidence": 0.95
}}"#,
            user_input, conversation_history
        );

        let response = self.llm_client.complete(prompt).await?;
        let intent: Intent = serde_json::from_str(&response.text)?;

        Ok(intent)
    }

    async fn select_tools(
        &self,
        intent: &Intent,
        context: &ExecutionContext,
    ) -> Result<Vec<ToolInfo>, AgentError> {
        // Get available tools for this user/org
        let available_tools = self.tool_registry.list(ToolFilter {
            capabilities: Some(intent.required_capabilities.clone()),
            organization_id: Some(context.organization_id.clone()),
            status: Some(vec![ToolStatus::Active]),
            ..Default::default()
        }).await;

        if available_tools.is_empty() {
            return Err(AgentError::NoToolsAvailable);
        }

        // Use LLM to rank and select best tools
        let tool_descriptions: Vec<_> = available_tools
            .iter()
            .map(|t| format!("{}: {}", t.name, t.description))
            .collect();

        let prompt = format!(
            r#"Select the best tools for this intent:
Intent: {:?}

Available tools:
{}

Return JSON array of tool IDs in order of relevance:
["tool-id-1", "tool-id-2"]"#,
            intent,
            tool_descriptions.join("\n")
        );

        let response = self.llm_client.complete(prompt).await?;
        let selected_ids: Vec<String> = serde_json::from_str(&response.text)?;

        let selected_tools: Vec<_> = selected_ids
            .iter()
            .filter_map(|id| {
                available_tools.iter().find(|t| &t.id == id).cloned()
            })
            .collect();

        Ok(selected_tools)
    }

    async fn execute_plan(
        &self,
        plan: &ExecutionPlan,
        context: &ExecutionContext,
    ) -> Result<Vec<ToolResult>, AgentError> {
        let mut results = Vec::new();

        for step in &plan.steps {
            match step.execution_type {
                ExecutionType::Sequential => {
                    // Execute steps one by one
                    let result = self.tool_registry
                        .execute(
                            &step.tool_id,
                            &step.action,
                            step.params.clone(),
                            context,
                        )
                        .await?;

                    results.push(result);
                }

                ExecutionType::Parallel => {
                    // Execute steps in parallel
                    let futures: Vec<_> = step.parallel_actions
                        .iter()
                        .map(|action| {
                            self.tool_registry.execute(
                                &action.tool_id,
                                &action.action,
                                action.params.clone(),
                                context,
                            )
                        })
                        .collect();

                    let parallel_results = futures::future::join_all(futures).await;

                    for result in parallel_results {
                        match result {
                            Ok(r) => results.push(r),
                            Err(e) => {
                                // Handle error based on plan error strategy
                                if step.error_strategy == ErrorStrategy::FailFast {
                                    return Err(AgentError::ExecutionError(e.to_string()));
                                }
                                // Continue on error
                                warn!("Step failed but continuing: {}", e);
                            }
                        }
                    }
                }
            }
        }

        Ok(results)
    }
}

#[derive(Debug, Clone)]
pub struct Intent {
    pub action: ActionType,
    pub entities: Vec<String>,
    pub filters: Vec<Filter>,
    pub output_format: OutputFormat,
    pub required_capabilities: Vec<Capability>,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub enum ActionType {
    Query,
    Create,
    Update,
    Delete,
    Analyze,
}

#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub steps: Vec<ExecutionStep>,
    pub estimated_time_ms: u64,
    pub estimated_cost: f64,
}

#[derive(Debug, Clone)]
pub struct ExecutionStep {
    pub tool_id: String,
    pub action: String,
    pub params: serde_json::Value,
    pub execution_type: ExecutionType,
    pub parallel_actions: Vec<ParallelAction>,
    pub error_strategy: ErrorStrategy,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionType {
    Sequential,
    Parallel,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrorStrategy {
    FailFast,
    Continue,
    Retry,
}
```

---

## 5. Multi-Tenant Infrastructure

### 5.1 Database Schema

```sql
-- Organizations (Tenants)
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255),
    plan VARCHAR(20) NOT NULL DEFAULT 'free', -- free, team, enterprise
    status VARCHAR(20) NOT NULL DEFAULT 'active', -- active, suspended, deleted

    -- Limits
    max_users INT,
    max_mcps INT,
    max_api_calls_per_month BIGINT,

    -- Billing
    stripe_customer_id VARCHAR(255),
    billing_email VARCHAR(255),

    -- Settings
    settings JSONB DEFAULT '{}',

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMP
);

CREATE INDEX idx_organizations_slug ON organizations(slug);
CREATE INDEX idx_organizations_domain ON organizations(domain);

-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    avatar_url VARCHAR(500),

    -- Auth
    password_hash VARCHAR(255), -- NULL if SSO-only
    email_verified BOOLEAN DEFAULT FALSE,

    -- SSO
    sso_provider VARCHAR(50), -- 'google', 'github', 'saml'
    sso_id VARCHAR(255),

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'active', -- active, suspended, deleted
    last_login_at TIMESTAMP,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_sso ON users(sso_provider, sso_id);

-- Organization Members (Many-to-Many)
CREATE TABLE organization_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL DEFAULT 'member', -- owner, admin, developer, member, viewer

    -- Permissions
    permissions JSONB DEFAULT '[]', -- Array of permission strings

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    UNIQUE(organization_id, user_id)
);

CREATE INDEX idx_org_members_org ON organization_members(organization_id);
CREATE INDEX idx_org_members_user ON organization_members(user_id);

-- Tools (MCP Servers, Built-in, Custom)
CREATE TABLE tools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,

    -- Identity
    slug VARCHAR(100) NOT NULL, -- 'stripe', 'database', 'browser'
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',

    -- Type
    tool_type VARCHAR(20) NOT NULL, -- 'mcp', 'builtin', 'custom'
    provider VARCHAR(100), -- 'stripe', 'postgres', etc.
    category VARCHAR(50), -- 'payment', 'database', etc.

    -- Configuration
    config JSONB NOT NULL DEFAULT '{}',
    secrets_encrypted TEXT, -- Encrypted JSON

    -- For MCP tools
    mcp_server_url VARCHAR(500),
    mcp_version VARCHAR(20),

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'active', -- active, inactive, error, deploying
    health_status VARCHAR(20), -- healthy, unhealthy, unknown
    last_health_check TIMESTAMP,

    -- Ownership
    created_by UUID REFERENCES users(id),

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    UNIQUE(organization_id, slug)
);

CREATE INDEX idx_tools_org ON tools(organization_id);
CREATE INDEX idx_tools_type ON tools(tool_type);
CREATE INDEX idx_tools_status ON tools(status);

-- Tool Permissions
CREATE TABLE tool_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tool_id UUID NOT NULL REFERENCES tools(id) ON DELETE CASCADE,
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,

    -- Grantee (either user or role)
    user_id UUID REFERENCES users(id),
    role VARCHAR(50), -- 'developer', 'analyst', etc.

    -- Permissions
    can_read BOOLEAN DEFAULT TRUE,
    can_write BOOLEAN DEFAULT FALSE,
    can_execute BOOLEAN DEFAULT TRUE,
    can_delete BOOLEAN DEFAULT FALSE,
    can_share BOOLEAN DEFAULT FALSE,

    -- Constraints
    rate_limit_per_minute INT,
    rate_limit_per_day INT,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CHECK (user_id IS NOT NULL OR role IS NOT NULL)
);

CREATE INDEX idx_tool_perms_tool ON tool_permissions(tool_id);
CREATE INDEX idx_tool_perms_user ON tool_permissions(user_id);

-- Audit Logs
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id),
    user_id UUID REFERENCES users(id),

    -- Action
    action VARCHAR(100) NOT NULL, -- 'tool.execute', 'tool.create', 'user.login'
    resource_type VARCHAR(50), -- 'tool', 'user', 'workflow'
    resource_id UUID,

    -- Details
    details JSONB,

    -- Request info
    ip_address INET,
    user_agent TEXT,
    session_id VARCHAR(255),

    -- Result
    success BOOLEAN NOT NULL,
    error_message TEXT,

    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_logs_org ON audit_logs(organization_id, created_at DESC);
CREATE INDEX idx_audit_logs_user ON audit_logs(user_id, created_at DESC);
CREATE INDEX idx_audit_logs_action ON audit_logs(action, created_at DESC);

-- Usage Metrics (for billing)
CREATE TABLE usage_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id),
    tool_id UUID REFERENCES tools(id),

    -- Metrics
    metric_type VARCHAR(50) NOT NULL, -- 'api_calls', 'compute_time', 'data_transfer'
    value BIGINT NOT NULL,
    unit VARCHAR(20), -- 'count', 'milliseconds', 'bytes'

    -- Time
    period DATE NOT NULL, -- Daily aggregation
    hour INT, -- Hour of day (0-23) for hourly breakdowns

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    UNIQUE(organization_id, tool_id, metric_type, period, hour)
);

CREATE INDEX idx_usage_metrics_org ON usage_metrics(organization_id, period DESC);
CREATE INDEX idx_usage_metrics_tool ON usage_metrics(tool_id, period DESC);

-- Secrets (stored encrypted in Vault, this is just metadata)
CREATE TABLE secrets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id),
    tool_id UUID REFERENCES tools(id),

    -- Identity
    key VARCHAR(100) NOT NULL, -- 'DATABASE_PASSWORD', 'API_KEY'
    description TEXT,

    -- Vault reference
    vault_path VARCHAR(500) NOT NULL,

    -- Rotation
    expires_at TIMESTAMP,
    last_rotated_at TIMESTAMP,
    rotation_policy VARCHAR(50), -- 'manual', 'auto_30d', 'auto_90d'

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    UNIQUE(organization_id, tool_id, key)
);

CREATE INDEX idx_secrets_org ON secrets(organization_id);
CREATE INDEX idx_secrets_tool ON secrets(tool_id);
```

### 5.2 Tenant Isolation

```rust
// src-platform/src/core/tenant_isolation.rs

/// Ensures all queries are scoped to the tenant
pub struct TenantContext {
    pub organization_id: String,
    pub user_id: String,
}

impl TenantContext {
    /// Add WHERE organization_id = ? to all queries
    pub fn scope_query(&self, mut query: sea_query::SelectStatement) -> sea_query::SelectStatement {
        query.and_where(Expr::col("organization_id").eq(&self.organization_id));
        query
    }
}

/// Row-Level Security (RLS) policies in PostgreSQL
pub fn setup_rls_policies(db: &Database) -> Result<()> {
    db.execute(r#"
        ALTER TABLE tools ENABLE ROW LEVEL SECURITY;

        CREATE POLICY tools_isolation ON tools
            USING (organization_id = current_setting('app.current_organization_id')::UUID);
    "#)?;

    // Apply to all tenant-scoped tables
    Ok(())
}
```

---

## 6. MCP Runtime (Kubernetes)

### 6.1 MCP Server Deployment

```yaml
# Kubernetes Deployment for MCP Server
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-{org_id}-{tool_slug}
  namespace: mcp-servers
  labels:
    app: mcp-server
    org: { org_id }
    tool: { tool_slug }
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mcp-server
      org: { org_id }
      tool: { tool_slug }
  template:
    metadata:
      labels:
        app: mcp-server
        org: { org_id }
        tool: { tool_slug }
    spec:
      containers:
        - name: mcp-server
          image: yantra/mcp-runtime:latest
          ports:
            - containerPort: 3000
          env:
            - name: TOOL_ID
              value: '{tool_id}'
            - name: ORG_ID
              value: '{org_id}'
            - name: MCP_SERVER_CODE
              valueFrom:
                configMapKeyRef:
                  name: mcp-{tool_id}-code
                  key: server.js
          # Secrets injected from Vault
          envFrom:
            - secretRef:
                name: mcp-{tool_id}-secrets
          resources:
            requests:
              memory: '128Mi'
              cpu: '100m'
            limits:
              memory: '512Mi'
              cpu: '500m'
          livenessProbe:
            httpGet:
              path: /health
              port: 3000
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: 3000
            initialDelaySeconds: 5
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-{org_id}-{tool_slug}
  namespace: mcp-servers
spec:
  selector:
    app: mcp-server
    org: { org_id }
    tool: { tool_slug }
  ports:
    - protocol: TCP
      port: 80
      targetPort: 3000
  type: ClusterIP
```

### 6.2 Auto-Scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcp-{org_id}-{tool_slug}
  namespace: mcp-servers
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mcp-{org_id}-{tool_slug}
  minReplicas: 1
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

---

## 7. Security & Compliance

### 7.1 Secrets Management (HashiCorp Vault)

```rust
// src-platform/src/core/secrets.rs

pub struct SecretsManager {
    vault_client: Arc<VaultClient>,
}

impl SecretsManager {
    pub async fn store_secret(
        &self,
        organization_id: &str,
        tool_id: &str,
        key: &str,
        value: &str,
    ) -> Result<(), SecretsError> {
        // Encrypt with org-specific key
        let encryption_key = self.get_org_encryption_key(organization_id).await?;
        let encrypted = self.encrypt(value, &encryption_key)?;

        // Store in Vault
        let vault_path = format!("tenants/{}/tools/{}/secrets/{}", organization_id, tool_id, key);
        self.vault_client.write(&vault_path, encrypted).await?;

        // Store metadata in DB
        self.store_secret_metadata(organization_id, tool_id, key, &vault_path).await?;

        Ok(())
    }

    pub async fn get_secret(
        &self,
        organization_id: &str,
        tool_id: &str,
        key: &str,
    ) -> Result<String, SecretsError> {
        // Get vault path from metadata
        let vault_path = self.get_vault_path(organization_id, tool_id, key).await?;

        // Read from Vault
        let encrypted = self.vault_client.read(&vault_path).await?;

        // Decrypt with org key
        let encryption_key = self.get_org_encryption_key(organization_id).await?;
        let decrypted = self.decrypt(&encrypted, &encryption_key)?;

        Ok(decrypted)
    }

    async fn get_org_encryption_key(&self, organization_id: &str) -> Result<Vec<u8>, SecretsError> {
        // Each organization has its own encryption key stored in Vault
        let key_path = format!("encryption-keys/orgs/{}", organization_id);
        let key_data = self.vault_client.read(&key_path).await?;
        Ok(key_data)
    }
}
```

### 7.2 Audit Logging

```rust
// src-platform/src/core/audit.rs

pub struct AuditLogger {
    db: Arc<Database>,
}

impl AuditLogger {
    pub async fn log_tool_execution(
        &self,
        user_id: &str,
        organization_id: &str,
        tool_id: &str,
        action: &str,
        params: &serde_json::Value,
    ) -> Result<(), AuditError> {
        sqlx::query!(
            r#"
            INSERT INTO audit_logs (
                organization_id, user_id, action, resource_type, resource_id, details, success
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7
            )
            "#,
            organization_id.parse::<Uuid>()?,
            user_id.parse::<Uuid>()?,
            format!("tool.execute.{}", action),
            "tool",
            tool_id.parse::<Uuid>()?,
            params,
            true // Will update after execution
        )
        .execute(&*self.db)
        .await?;

        Ok(())
    }

    pub async fn generate_compliance_report(
        &self,
        organization_id: &str,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<ComplianceReport, AuditError> {
        // Query audit logs for compliance
        let logs = sqlx::query!(
            r#"
            SELECT action, COUNT(*) as count, success
            FROM audit_logs
            WHERE organization_id = $1
              AND created_at >= $2
              AND created_at <= $3
            GROUP BY action, success
            "#,
            organization_id.parse::<Uuid>()?,
            start_date.naive_utc(),
            end_date.naive_utc()
        )
        .fetch_all(&*self.db)
        .await?;

        // Generate report
        Ok(ComplianceReport {
            organization_id: organization_id.to_string(),
            period_start: start_date,
            period_end: end_date,
            total_actions: logs.iter().map(|l| l.count.unwrap_or(0)).sum(),
            actions_by_type: logs.into_iter().map(|l| (l.action, l.count.unwrap_or(0))).collect(),
        })
    }
}
```

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)

**Goal:** Core platform infrastructure

- ✅ Unified Tools Interface (UTI) definition
- ✅ Tool Registry implementation
- ✅ Basic MCP Tool wrapper
- ✅ Built-in Browser Tool
- ✅ Multi-tenant database schema
- ✅ Authentication (JWT + OAuth)
- ✅ Basic RBAC

**Deliverables:**

- Tool Registry API
- MCP Tool can be registered and executed
- Browser Tool working
- Users can login and belong to organizations

### Phase 2: Agent & Orchestration (Weeks 5-8)

**Goal:** AI agent that can use tools

- ✅ Agent implementation
- ✅ Intent parsing (LLM integration)
- ✅ Tool selection logic
- ✅ Task planning
- ✅ Execution engine
- ✅ Context management

**Deliverables:**

- Agent can parse natural language requests
- Agent can select appropriate tools
- Agent can execute multi-step plans
- Conversation context maintained

### Phase 3: MCP Runtime (Weeks 9-12)

**Goal:** Deploy and host MCP servers

- ✅ Kubernetes cluster setup
- ✅ MCP server deployment automation
- ✅ Auto-scaling configuration
- ✅ Health monitoring
- ✅ Secrets injection from Vault

**Deliverables:**

- MCP servers deploy to Kubernetes
- Auto-scaling based on load
- Health checks and auto-restart
- Secrets securely injected

### Phase 4: Security & Compliance (Weeks 13-16)

**Goal:** Enterprise-grade security

- ✅ HashiCorp Vault integration
- ✅ Secrets encryption
- ✅ Audit logging (all actions)
- ✅ Compliance reports (SOC2, HIPAA, GDPR)
- ✅ SSO/SAML integration

**Deliverables:**

- All secrets stored in Vault
- Audit logs for all actions
- Compliance reports generated
- SSO working

### Phase 5: Product Integration (Weeks 17-20)

**Goal:** Integrate with Develop, Flow, Data

- ✅ Yantra Develop integration
- ✅ Yantra Flow integration
- ✅ Yantra Data integration
- ✅ Cross-product tool sharing
- ✅ Unified billing

**Deliverables:**

- Tools created in Develop available in Flow/Data
- Workflows in Flow can use any tool
- Data queries can access any tool
- Usage tracked for billing

---

## 9. Success Metrics

### 9.1 Technical Metrics

**Performance:**

- Tool execution latency: <500ms (p95)
- Tool registry lookup: <50ms (p95)
- MCP server deploy time: <2 minutes
- Agent response time: <2 seconds (p95)

**Reliability:**

- Platform uptime: 99.9%
- MCP server uptime: 99.5%
- Failed tool executions: <1%
- Audit log completeness: 100%

**Scale:**

- Support 1000+ organizations
- Support 10,000+ tools
- Handle 1M+ tool executions/day
- Store 100M+ audit logs

### 9.2 Business Metrics

**Adoption:**

- Tools created per org: >5
- Tools used per workflow: >3
- Active users per org: >10
- Tool reuse rate: >50%

**Engagement:**

- Daily active users: >1000
- Tool executions per user/day: >10
- Cross-product usage: >70%

**Revenue:**

- Paid organizations: >100
- Average ACV: $24,000
- Churn rate: <5% monthly
- Expansion rate: >120%

---

## 10. Appendix

### 10.1 Error Codes

```rust
pub enum PlatformError {
    // Tool Registry
    ToolNotFound(String),              // 1001
    ToolAlreadyExists(String),         // 1002
    ToolRegistrationFailed(String),    // 1003

    // Execution
    ExecutionFailed(String),           // 2001
    InvalidParams(String),             // 2002
    PermissionDenied(String),          // 2003
    RateLimitExceeded(String),         // 2004

    // Agent
    IntentParsingFailed(String),       // 3001
    PlanningFailed(String),            // 3002
    NoToolsAvailable,                  // 3003

    // Tenant
    OrganizationNotFound(String),      // 4001
    UserNotFound(String),              // 4002
    InsufficientPermissions(String),   // 4003

    // MCP
    MCPServerDown(String),             // 5001
    MCPDeploymentFailed(String),       // 5002
    MCPHealthCheckFailed(String),      // 5003

    // Security
    SecretsAccessDenied(String),       // 6001
    EncryptionFailed(String),          // 6002
    AuditLogFailed(String),            // 6003
}
```

### 10.2 Configuration Example

```yaml
# platform-config.yaml
platform:
  name: 'Yantra Platform'
  version: '1.0.0'
  environment: 'production'

database:
  host: 'postgres.platform.yantra.internal'
  port: 5432
  database: 'yantra_platform'
  pool_size: 50

kubernetes:
  namespace: 'mcp-servers'
  registry: 'gcr.io/yantra-platform'
  max_pods_per_org: 50

vault:
  address: 'https://vault.platform.yantra.internal'
  role: 'yantra-platform'
  mount: 'tenants'

llm:
  provider: 'anthropic'
  model: 'claude-3-5-sonnet-20241022'
  api_key_env: 'ANTHROPIC_API_KEY'
  max_tokens: 4096

auth:
  jwt_secret_env: 'JWT_SECRET'
  session_duration: '24h'
  refresh_token_duration: '30d'

  sso:
    enabled: true
    providers:
      - google
      - github
      - saml

monitoring:
  prometheus:
    enabled: true
    port: 9090

  logging:
    level: 'info'
    format: 'json'
    output: 'elk'

limits:
  free_tier:
    max_tools: 5
    max_api_calls_per_month: 10000

  team_tier:
    max_tools: 50
    max_api_calls_per_month: 1000000

  enterprise_tier:
    max_tools: unlimited
    max_api_calls_per_month: unlimited
```

---

**Document Status:** Complete  
**Ready for:** Implementation  
**Next Steps:**

1. Review and approve specification
2. Set up development environment
3. Begin Phase 1 implementation
4. Weekly progress reviews

**Estimated Timeline:** 20 weeks to full platform
**Team Size:** 4-6 engineers
**Budget:** Determined by organization

---

_This specification defines the core platform that powers all Yantra products. It is the foundation upon which Develop, Flow, and Data are built._
