# Yantra Flow - Technical Specification

**Version:** 1.0  
**Date:** December 9, 2025  
**Status:** Specification - Ready for Implementation  
**Phase:** Cloud Product #1 - Workflow Automation & App Builder

---

## 1. Executive Summary

### 1.1 Vision

Yantra Flow is a cloud-based no-code workflow automation and dynamic application builder platform. It enables business users to create powerful automations and applications using MCP connectors created by developers in Yantra Develop, without writing any code.

### 1.2 Problem Statement

**Current Challenges:**
- Business users depend on developers for every automation
- Traditional workflow tools (Zapier) have limited connectors
- RPA tools (UiPath) are expensive and complex
- Building custom apps requires months of development
- Each tool operates in isolation (workflows ≠ apps ≠ data)

**Yantra Flow Solution:**
- **Workflow Automation:** No-code workflow builder using MCP connectors
- **Dynamic App Builder:** AI generates full-stack apps in minutes
- **Unified Platform:** Uses same MCPs as Yantra Develop and Data
- **Enterprise-Grade:** SSO, RBAC, audit logs, compliance

### 1.3 Success Criteria

**MVP Success Metrics:**
- ✅ Create workflow in <5 minutes (no code)
- ✅ Generate app in <60 seconds from natural language
- ✅ Support 100+ workflow executions/second
- ✅ 99.9% workflow execution success rate
- ✅ Deploy app with custom domain in <2 minutes

---

## 2. Product Overview

### 2.1 Core Features

```
┌──────────────────────────────────────────────────────────┐
│                    YANTRA FLOW                            │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  1. Workflow Automation                             │ │
│  │     • Visual workflow builder (drag-drop)           │ │
│  │     • MCP connector catalog                         │ │
│  │     • Triggers (webhook, schedule, event)           │ │
│  │     • Actions (MCP tools)                           │ │
│  │     • Conditions & branching                        │ │
│  │     • Error handling & retries                      │ │
│  │     • Monitoring & debugging                        │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  2. Dynamic App Builder                             │ │
│  │     • Natural language to full-stack app            │ │
│  │     • Uses MCP connectors as backend                │ │
│  │     • Generates React frontend                      │ │
│  │     • Instant deployment                            │ │
│  │     • Custom domains                                │ │
│  │     • Authentication (SSO from org)                 │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  3. MCP Connector Management                        │ │
│  │     • Browse available connectors                   │ │
│  │     • Configure credentials (OAuth, API keys)       │ │
│  │     • Test connections                              │ │
│  │     • View usage & costs                            │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  4. Enterprise Features                             │ │
│  │     • Workflow versioning & rollback                │ │
│  │     • Approval workflows                            │ │
│  │     • Team collaboration                            │ │
│  │     • Audit logs                                    │ │
│  │     • Compliance reports                            │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Workflow Automation

### 3.1 Workflow Builder UI

```
┌──────────────────────────────────────────────────────────┐
│  Yantra Flow - Workflow Builder                          │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐  ┌──────────────────────────────────┐  │
│  │ Connectors  │  │  Canvas                          │  │
│  │             │  │                                  │  │
│  │ 🔌 Stripe   │  │  ┌─────────────┐               │  │
│  │ 💾 Database │  │  │   Trigger   │               │  │
│  │ 📧 Email    │  │  │  Webhook    │               │  │
│  │ 🎫 Zendesk  │  │  └──────┬──────┘               │  │
│  │ 🔧 Salesfor │  │         │                       │  │
│  │             │  │         ▼                       │  │
│  │ [Search]    │  │  ┌─────────────┐               │  │
│  │             │  │  │   Action    │               │  │
│  │             │  │  │ Get Customer│               │  │
│  │             │  │  │ (Database)  │               │  │
│  │             │  │  └──────┬──────┘               │  │
│  │             │  │         │                       │  │
│  │             │  │         ▼                       │  │
│  │             │  │  ┌─────────────┐               │  │
│  │             │  │  │  Condition  │               │  │
│  │             │  │  │ If Premium? │               │  │
│  │             │  │  └──┬───────┬──┘               │  │
│  │             │  │     │  Yes  │  No              │  │
│  │             │  │     ▼       ▼                  │  │
│  │             │  │ ┌──────┐ ┌──────┐             │  │
│  │             │  │ │Stripe│ │Email │             │  │
│  │             │  │ │ VIP  │ │Basic │             │  │
│  │             │  │ └──────┘ └──────┘             │  │
│  │             │  │                                  │  │
│  └─────────────┘  └──────────────────────────────────┘  │
│                                                           │
│  [Save Draft]  [Test Run]  [Deploy]                     │
└──────────────────────────────────────────────────────────┘
```

### 3.2 Workflow Data Model

```typescript
// Workflow definition
interface Workflow {
  id: string;
  organization_id: string;
  name: string;
  description: string;
  version: number;
  
  // Flow definition
  trigger: Trigger;
  steps: Step[];
  
  // Configuration
  config: WorkflowConfig;
  
  // State
  status: 'draft' | 'active' | 'paused' | 'archived';
  deployed_at?: string;
  
  // Metadata
  created_by: string;
  created_at: string;
  updated_at: string;
}

// Triggers
interface Trigger {
  type: 'webhook' | 'schedule' | 'event' | 'manual';
  config: TriggerConfig;
}

interface WebhookTrigger extends Trigger {
  type: 'webhook';
  config: {
    url: string;         // Generated webhook URL
    method: 'POST' | 'GET';
    auth?: 'none' | 'basic' | 'bearer' | 'signature';
    secret?: string;     // For signature verification
  };
}

interface ScheduleTrigger extends Trigger {
  type: 'schedule';
  config: {
    cron: string;        // "0 9 * * *" (9am daily)
    timezone: string;    // "America/New_York"
  };
}

interface EventTrigger extends Trigger {
  type: 'event';
  config: {
    tool_id: string;     // MCP connector
    event_name: string;  // "customer.created"
    filter?: object;     // Event filter
  };
}

// Workflow Steps
interface Step {
  id: string;
  name: string;
  type: 'action' | 'condition' | 'loop' | 'parallel';
  config: StepConfig;
  
  // Flow control
  on_success?: string;  // Next step ID
  on_failure?: string;  // Error handler step ID
  retry_config?: RetryConfig;
}

// Action Step (call MCP tool)
interface ActionStep extends Step {
  type: 'action';
  config: {
    tool_id: string;       // e.g., "mcp-stripe"
    action: string;        // e.g., "create_payment"
    parameters: Record<string, ParameterValue>;
    timeout_ms?: number;
  };
}

// Parameter can reference previous step output
type ParameterValue = 
  | string                           // Static: "value"
  | { $ref: string }                 // Reference: { $ref: "trigger.body.customer_id" }
  | { $expr: string };               // Expression: { $expr: "trigger.amount * 1.1" }

// Condition Step (branching)
interface ConditionStep extends Step {
  type: 'condition';
  config: {
    condition: Condition;
    on_true: string;      // Step ID if true
    on_false: string;     // Step ID if false
  };
}

interface Condition {
  operator: '==' | '!=' | '>' | '<' | '>=' | '<=' | 'contains' | 'in';
  left: ParameterValue;
  right: ParameterValue;
}

// Loop Step
interface LoopStep extends Step {
  type: 'loop';
  config: {
    items: ParameterValue;  // Array to iterate
    steps: Step[];          // Steps to execute per item
    max_iterations?: number;
    parallel?: boolean;     // Execute in parallel
  };
}

// Parallel Step
interface ParallelStep extends Step {
  type: 'parallel';
  config: {
    steps: Step[];          // Execute all in parallel
    wait_for: 'all' | 'any' | 'none';
  };
}

// Retry Configuration
interface RetryConfig {
  max_attempts: number;
  backoff_type: 'fixed' | 'exponential';
  initial_delay_ms: number;
  max_delay_ms?: number;
}

// Workflow Configuration
interface WorkflowConfig {
  // Execution
  timeout_ms: number;           // Default: 300000 (5 min)
  max_concurrent_executions: number;  // Default: 10
  
  // Error Handling
  on_error: 'fail' | 'continue' | 'retry';
  error_notification?: {
    email?: string[];
    slack?: string;             // Slack webhook URL
  };
  
  // Rate Limiting
  rate_limit?: {
    max_per_minute: number;
    max_per_hour: number;
  };
}
```

### 3.3 Workflow Execution Engine

```rust
// src-flow/src/engine/executor.rs

pub struct WorkflowExecutor {
    tool_registry: Arc<ToolRegistry>,
    execution_store: Arc<ExecutionStore>,
    event_bus: Arc<EventBus>,
}

impl WorkflowExecutor {
    pub async fn execute(
        &self,
        workflow: &Workflow,
        trigger_data: serde_json::Value,
        context: &ExecutionContext,
    ) -> Result<ExecutionResult, ExecutionError> {
        // Create execution record
        let execution_id = Uuid::new_v4().to_string();
        let execution = Execution {
            id: execution_id.clone(),
            workflow_id: workflow.id.clone(),
            status: ExecutionStatus::Running,
            started_at: Utc::now(),
            trigger_data: trigger_data.clone(),
            steps: vec![],
        };
        
        self.execution_store.create(&execution).await?;
        
        // Execute workflow
        let result = self.execute_workflow_internal(
            workflow,
            trigger_data,
            context,
            &execution_id,
        ).await;
        
        // Update execution record
        let final_status = match result {
            Ok(_) => ExecutionStatus::Completed,
            Err(_) => ExecutionStatus::Failed,
        };
        
        self.execution_store.update_status(&execution_id, final_status).await?;
        
        result
    }
    
    async fn execute_workflow_internal(
        &self,
        workflow: &Workflow,
        trigger_data: serde_json::Value,
        context: &ExecutionContext,
        execution_id: &str,
    ) -> Result<ExecutionResult, ExecutionError> {
        let mut step_outputs = HashMap::new();
        step_outputs.insert("trigger".to_string(), trigger_data);
        
        let mut current_step_id = workflow.steps.first().map(|s| s.id.clone());
        
        while let Some(step_id) = current_step_id {
            let step = workflow.steps.iter().find(|s| s.id == step_id)
                .ok_or(ExecutionError::StepNotFound(step_id.clone()))?;
            
            // Execute step
            let step_result = self.execute_step(
                step,
                &step_outputs,
                context,
                execution_id,
            ).await;
            
            // Store step output
            match step_result {
                Ok(output) => {
                    step_outputs.insert(step_id.clone(), output.data.clone());
                    
                    // Record step execution
                    self.execution_store.add_step_execution(
                        execution_id,
                        &step_id,
                        StepExecutionStatus::Completed,
                        Some(&output.data),
                        None,
                    ).await?;
                    
                    // Determine next step
                    current_step_id = step.on_success.clone();
                }
                Err(e) => {
                    // Record step failure
                    self.execution_store.add_step_execution(
                        execution_id,
                        &step_id,
                        StepExecutionStatus::Failed,
                        None,
                        Some(&e.to_string()),
                    ).await?;
                    
                    // Handle error
                    match workflow.config.on_error {
                        OnError::Fail => return Err(e),
                        OnError::Continue => {
                            current_step_id = step.on_success.clone();
                        }
                        OnError::Retry => {
                            // Retry logic
                            if let Some(retry_config) = &step.retry_config {
                                let retry_result = self.retry_step(
                                    step,
                                    &step_outputs,
                                    context,
                                    execution_id,
                                    retry_config,
                                ).await;
                                
                                match retry_result {
                                    Ok(output) => {
                                        step_outputs.insert(step_id.clone(), output.data.clone());
                                        current_step_id = step.on_success.clone();
                                    }
                                    Err(retry_err) => {
                                        if let Some(error_step) = &step.on_failure {
                                            current_step_id = Some(error_step.clone());
                                        } else {
                                            return Err(retry_err);
                                        }
                                    }
                                }
                            } else if let Some(error_step) = &step.on_failure {
                                current_step_id = Some(error_step.clone());
                            } else {
                                return Err(e);
                            }
                        }
                    }
                }
            }
        }
        
        Ok(ExecutionResult {
            execution_id: execution_id.to_string(),
            status: ExecutionStatus::Completed,
            outputs: step_outputs,
        })
    }
    
    async fn execute_step(
        &self,
        step: &Step,
        step_outputs: &HashMap<String, serde_json::Value>,
        context: &ExecutionContext,
        execution_id: &str,
    ) -> Result<StepOutput, ExecutionError> {
        match step.type_field {
            StepType::Action => {
                self.execute_action_step(step, step_outputs, context).await
            }
            StepType::Condition => {
                self.execute_condition_step(step, step_outputs).await
            }
            StepType::Loop => {
                self.execute_loop_step(step, step_outputs, context, execution_id).await
            }
            StepType::Parallel => {
                self.execute_parallel_step(step, step_outputs, context, execution_id).await
            }
        }
    }
    
    async fn execute_action_step(
        &self,
        step: &Step,
        step_outputs: &HashMap<String, serde_json::Value>,
        context: &ExecutionContext,
    ) -> Result<StepOutput, ExecutionError> {
        let config = step.config.as_action()
            .ok_or(ExecutionError::InvalidStepConfig)?;
        
        // Resolve parameters (handle $ref and $expr)
        let resolved_params = self.resolve_parameters(
            &config.parameters,
            step_outputs,
        )?;
        
        // Execute tool action
        let result = self.tool_registry.execute(
            &config.tool_id,
            &config.action,
            resolved_params,
            context,
        ).await?;
        
        Ok(StepOutput {
            data: result.data,
            metadata: result.metadata,
        })
    }
    
    fn resolve_parameters(
        &self,
        parameters: &HashMap<String, ParameterValue>,
        step_outputs: &HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value, ExecutionError> {
        let mut resolved = serde_json::Map::new();
        
        for (key, value) in parameters {
            let resolved_value = match value {
                ParameterValue::Static(s) => serde_json::Value::String(s.clone()),
                ParameterValue::Reference(ref_path) => {
                    self.resolve_reference(ref_path, step_outputs)?
                }
                ParameterValue::Expression(expr) => {
                    self.evaluate_expression(expr, step_outputs)?
                }
            };
            
            resolved.insert(key.clone(), resolved_value);
        }
        
        Ok(serde_json::Value::Object(resolved))
    }
    
    fn resolve_reference(
        &self,
        ref_path: &str,
        step_outputs: &HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value, ExecutionError> {
        // Parse reference: "trigger.body.customer_id" or "step1.data.email"
        let parts: Vec<&str> = ref_path.split('.').collect();
        
        if parts.is_empty() {
            return Err(ExecutionError::InvalidReference(ref_path.to_string()));
        }
        
        let step_id = parts[0];
        let output = step_outputs.get(step_id)
            .ok_or(ExecutionError::ReferenceNotFound(ref_path.to_string()))?;
        
        // Navigate path
        let mut current = output;
        for part in &parts[1..] {
            current = current.get(part)
                .ok_or(ExecutionError::ReferenceNotFound(ref_path.to_string()))?;
        }
        
        Ok(current.clone())
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub execution_id: String,
    pub status: ExecutionStatus,
    pub outputs: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub enum ExecutionStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}
```

### 3.4 Webhook Triggers

```rust
// src-flow/src/triggers/webhook.rs

pub struct WebhookHandler {
    workflow_store: Arc<WorkflowStore>,
    executor: Arc<WorkflowExecutor>,
}

impl WebhookHandler {
    pub async fn handle_webhook(
        &self,
        webhook_id: &str,
        request: HttpRequest,
    ) -> Result<HttpResponse, WebhookError> {
        // Find workflow by webhook ID
        let workflow = self.workflow_store
            .find_by_webhook_id(webhook_id)
            .await?
            .ok_or(WebhookError::WebhookNotFound)?;
        
        // Verify webhook authentication
        self.verify_webhook_auth(&workflow, &request)?;
        
        // Extract trigger data
        let trigger_data = self.extract_trigger_data(&request).await?;
        
        // Execute workflow asynchronously
        let execution_context = ExecutionContext {
            user_id: workflow.created_by.clone(),
            organization_id: workflow.organization_id.clone(),
            session_id: Uuid::new_v4().to_string(),
            ip_address: request.remote_addr().to_string(),
            user_agent: request.headers().get("User-Agent")
                .and_then(|h| h.to_str().ok())
                .unwrap_or("unknown")
                .to_string(),
            product: Product::Flow,
        };
        
        // Spawn async execution (don't wait)
        let executor = self.executor.clone();
        let workflow_clone = workflow.clone();
        tokio::spawn(async move {
            let result = executor.execute(
                &workflow_clone,
                trigger_data,
                &execution_context,
            ).await;
            
            if let Err(e) = result {
                error!("Workflow execution failed: {}", e);
            }
        });
        
        // Return immediately
        Ok(HttpResponse::Accepted().json(json!({
            "message": "Workflow triggered",
            "workflow_id": workflow.id
        })))
    }
}
```

---

## 4. Dynamic App Builder

### 4.1 App Generation Flow

```
User: "Create customer portal where customers can view their 
       orders and update payment methods"

↓

Yantra Flow AI:
1. Analyze intent
2. Identify required MCPs:
   - Database MCP (orders)
   - Stripe MCP (payment methods)
3. Generate React app:
   - Authentication (SSO from org)
   - Dashboard page
   - Orders list page
   - Payment settings page
   - API routes (use MCPs)
4. Deploy to: portal.acme.yantra.app
5. Return link to user

Total time: 60 seconds
```

### 4.2 Generated App Structure

```
generated-app-{id}/
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Orders.tsx
│   │   │   └── PaymentSettings.tsx
│   │   ├── components/
│   │   │   ├── OrderCard.tsx
│   │   │   └── PaymentForm.tsx
│   │   ├── api/
│   │   │   └── client.ts
│   │   └── auth/
│   │       └── AuthProvider.tsx
│   ├── package.json
│   └── vite.config.ts
│
├── backend/
│   ├── src/
│   │   ├── routes/
│   │   │   ├── orders.ts
│   │   │   └── payments.ts
│   │   ├── middleware/
│   │   │   └── auth.ts
│   │   └── server.ts
│   └── package.json
│
├── deploy/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── kubernetes.yaml
│
└── yantra-app.yaml  # App configuration
```

### 4.3 App Configuration

```yaml
# yantra-app.yaml
app:
  id: "app-123"
  name: "Customer Portal"
  organization_id: "acme-123"
  
  # Deployment
  domain: "portal.acme.yantra.app"
  custom_domain: "portal.acme.com"  # Optional
  
  # Authentication
  auth:
    type: "sso"  # Use organization SSO
    allowed_roles: ["customer"]
    
  # MCPs used by this app
  mcps:
    - id: "mcp-database"
      permissions: ["read"]
    - id: "mcp-stripe"
      permissions: ["read", "write"]
  
  # Environment
  environment:
    NODE_ENV: "production"
    API_URL: "https://api.portal.acme.yantra.app"
  
  # Resources
  resources:
    cpu: "500m"
    memory: "512Mi"
    replicas: 2
```

### 4.4 App Generator

```typescript
// src-flow/src/apps/generator.ts

export class AppGenerator {
  constructor(
    private llmClient: LLMClient,
    private toolRegistry: ToolRegistry,
  ) {}
  
  async generateApp(
    description: string,
    organization: Organization,
  ): Promise<GeneratedApp> {
    // 1. Analyze requirements
    const requirements = await this.analyzeRequirements(description);
    
    // 2. Select MCPs
    const mcps = await this.selectMCPs(requirements, organization);
    
    // 3. Generate frontend
    const frontend = await this.generateFrontend(requirements, mcps);
    
    // 4. Generate backend
    const backend = await this.generateBackend(requirements, mcps);
    
    // 5. Generate deployment config
    const deployment = this.generateDeploymentConfig(requirements);
    
    // 6. Bundle everything
    const app = {
      id: generateId(),
      name: requirements.name,
      organization_id: organization.id,
      frontend,
      backend,
      deployment,
      mcps: mcps.map(m => ({ id: m.id, permissions: m.required_permissions })),
    };
    
    return app;
  }
  
  private async generateFrontend(
    requirements: AppRequirements,
    mcps: MCP[],
  ): Promise<FrontendCode> {
    const prompt = `
Generate a React + TypeScript + Vite application with:

Requirements:
${JSON.stringify(requirements, null, 2)}

Available MCPs (for data):
${mcps.map(m => `- ${m.name}: ${m.description}`).join('\n')}

Generate:
1. App.tsx (main app with routing)
2. Pages for each feature
3. Components for data display
4. API client (calls backend which uses MCPs)
5. Auth provider (SSO)
6. Tailwind CSS for styling

Code must be production-ready, well-structured, and follow best practices.
`;
    
    const response = await this.llmClient.complete(prompt);
    
    return this.parseFrontendCode(response.text);
  }
  
  private async generateBackend(
    requirements: AppRequirements,
    mcps: MCP[],
  ): Promise<BackendCode> {
    const prompt = `
Generate a Node.js + Express + TypeScript backend with:

Requirements:
${JSON.stringify(requirements, null, 2)}

Available MCPs (use these for data access):
${mcps.map(m => \`
- \${m.id}: \${m.name}
  Resources: \${m.resources.map(r => r.name).join(', ')}
  Actions: \${m.actions.map(a => a.name).join(', ')}
\`).join('\n')}

Generate:
1. Express server setup
2. Routes for each feature
3. Middleware (auth, error handling)
4. MCP client (use Yantra Platform API)
5. Request validation

Backend should:
- Use MCPs for all data operations (no direct DB access)
- Handle auth via JWT (from organization SSO)
- Have proper error handling
- Include rate limiting
`;
    
    const response = await this.llmClient.complete(prompt);
    
    return this.parseBackendCode(response.text);
  }
}
```

---

## 5. MCP Connector Management

### 5.1 Connector Catalog UI

```
┌──────────────────────────────────────────────────────────┐
│  Connectors                                               │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  [Search connectors...]                  [+ Add Custom]  │
│                                                           │
│  Connected (8):                                          │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ 💳 Stripe    │  │ 💾 Database  │  │ 📧 Gmail     │  │
│  │ Connected    │  │ Connected    │  │ Connected    │  │
│  │ ✓ Active     │  │ ✓ Active     │  │ ✓ Active     │  │
│  │              │  │              │  │              │  │
│  │ 1.2k calls   │  │ 3.4k calls   │  │ 234 calls    │  │
│  │ this month   │  │ this month   │  │ this month   │  │
│  │              │  │              │  │              │  │
│  │ [Configure]  │  │ [Configure]  │  │ [Configure]  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                           │
│  Available (23):                                         │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ 🎫 Zendesk   │  │ 📊 Mixpanel  │  │ 🔧 Salesfor  │  │
│  │ Not connect  │  │ Not connect  │  │ Not connect  │  │
│  │              │  │              │  │              │  │
│  │ Support      │  │ Analytics    │  │ CRM          │  │
│  │ tickets      │  │ events       │  │              │  │
│  │              │  │              │  │              │  │
│  │ [Connect]    │  │ [Connect]    │  │ [Connect]    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 5.2 Connector Configuration

```typescript
// When user clicks "Connect" on a connector

interface ConnectorConfig {
  tool_id: string;
  name: string;
  auth_type: 'oauth' | 'api_key' | 'basic' | 'custom';
  
  // OAuth
  oauth?: {
    provider: string;
    client_id: string;
    scopes: string[];
    redirect_uri: string;
  };
  
  // API Key
  api_key?: {
    key_name: string;       // "API_KEY" or "X-API-Key"
    key_location: 'header' | 'query';
  };
  
  // Custom fields
  custom_fields?: {
    name: string;
    type: 'string' | 'number' | 'boolean' | 'secret';
    required: boolean;
    description: string;
    default?: any;
  }[];
}

// Example: Stripe connector configuration
{
  tool_id: "mcp-stripe",
  name: "Stripe",
  auth_type: "api_key",
  api_key: {
    key_name: "STRIPE_API_KEY",
    key_location: "header"
  }
}

// Example: Salesforce connector configuration
{
  tool_id: "mcp-salesforce",
  name: "Salesforce",
  auth_type: "oauth",
  oauth: {
    provider: "salesforce",
    client_id: "...",
    scopes: ["api", "refresh_token"],
    redirect_uri: "https://flow.yantra.app/oauth/callback"
  }
}
```

---

## 6. Enterprise Features

### 6.1 Workflow Versioning

```rust
// Workflow versions stored in database
CREATE TABLE workflow_versions (
    id UUID PRIMARY KEY,
    workflow_id UUID NOT NULL REFERENCES workflows(id),
    version INT NOT NULL,
    definition JSONB NOT NULL,
    changes TEXT,  -- Description of what changed
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP NOT NULL,
    
    UNIQUE(workflow_id, version)
);

// Rollback to previous version
pub async fn rollback_workflow(
    workflow_id: &str,
    target_version: i32,
) -> Result<(), Error> {
    let old_version = get_workflow_version(workflow_id, target_version).await?;
    
    // Create new version from old definition
    create_workflow_version(workflow_id, old_version.definition).await?;
    
    // Update active workflow
    update_workflow(workflow_id, old_version.definition).await?;
    
    Ok(())
}
```

### 6.2 Approval Workflows

```typescript
// Approval workflow for sensitive operations
interface ApprovalWorkflow {
  id: string;
  workflow_id: string;
  action: 'deploy' | 'update' | 'delete';
  requested_by: string;
  requested_at: string;
  
  // Approval chain
  approvers: Approver[];
  approval_policy: 'any' | 'all' | 'majority';
  
  // State
  status: 'pending' | 'approved' | 'rejected';
  approvals: Approval[];
}

interface Approver {
  user_id?: string;
  role?: string;  // e.g., "admin"
}

interface Approval {
  approver_id: string;
  decision: 'approved' | 'rejected';
  comment?: string;
  timestamp: string;
}

// Example: Deployment requires admin approval
{
  workflow_id: "wf-123",
  action: "deploy",
  requested_by: "user-456",
  approvers: [
    { role: "admin" }
  ],
  approval_policy: "any",  // Any admin can approve
  status: "pending"
}
```

---

## 7. User Interface

### 7.1 Main Dashboard

```
┌──────────────────────────────────────────────────────────┐
│  Yantra Flow                                  [User ▼]   │
├──────────────────────────────────────────────────────────┤
│  [Workflows]  [Apps]  [Connectors]  [Executions]        │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Recent Workflows                        [+ New Workflow]│
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Process Orders                            Active   │  │
│  │ Triggered 234 times today                         │  │
│  │ Success rate: 99.1%                               │  │
│  │ Avg duration: 1.2s                                │  │
│  │ [View] [Edit] [Pause]                             │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Daily Customer Report                     Active   │  │
│  │ Runs every day at 9am                             │  │
│  │ Last run: 2 hours ago                             │  │
│  │ Next run: in 22 hours                             │  │
│  │ [View] [Edit] [Run Now]                           │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  Recent Apps                                 [+ New App] │
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Customer Portal                          🟢 Live   │  │
│  │ https://portal.acme.yantra.app                    │  │
│  │ 45 active users                                   │  │
│  │ [Open] [Edit] [Analytics]                         │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  Activity Feed                                           │
│                                                           │
│  • Workflow "Process Orders" executed successfully       │
│    2 minutes ago                                         │
│                                                           │
│  • Connector "Stripe" rate limit warning                 │
│    15 minutes ago                                        │
│                                                           │
│  • App "Customer Portal" deployed                        │
│    1 hour ago                                            │
└──────────────────────────────────────────────────────────┘
```

---

## 8. Pricing & Packaging

```
┌─────────────────────────────────────────────────────────┐
│  Yantra Flow Pricing                                    │
│                                                          │
│  Starter: $49/month                                     │
│  • 10 workflows                                         │
│  • 1 dynamic app                                        │
│  • 10,000 workflow executions/month                     │
│  • 5 users                                              │
│  • Email support                                        │
│  • Basic connectors (10+)                               │
│                                                          │
│  Professional: $199/month                               │
│  • 100 workflows                                        │
│  • 10 dynamic apps                                      │
│  • 100,000 workflow executions/month                    │
│  • 25 users                                             │
│  • Priority support                                     │
│  • All connectors (100+)                                │
│  • Workflow versioning                                  │
│  • Custom domains for apps                              │
│                                                          │
│  Enterprise: $999/month                                 │
│  • Unlimited workflows                                  │
│  • Unlimited apps                                       │
│  • Unlimited executions                                 │
│  • Unlimited users                                      │
│  • Dedicated support + SLA                              │
│  • All connectors + custom                              │
│  • SSO/SAML                                             │
│  • Approval workflows                                   │
│  • Audit logs & compliance                              │
│  • On-premise agents                                    │
│  • White-label apps                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 9. Success Metrics

### 9.1 Technical Metrics
- Workflow creation time: <5 minutes
- Workflow execution latency: <2 seconds (p95)
- App generation time: <60 seconds
- App deployment time: <2 minutes
- Workflow success rate: >99%
- Platform uptime: 99.9%

### 9.2 Business Metrics
- Active workflows per org: >10
- Workflow executions per day: >100
- Apps created per org: >2
- User adoption rate: >70%
- Monthly recurring revenue: >$100K

---

**Document Status:** Complete  
**Next Steps:**
1. Review and approve specification
2. Begin implementation (workflow engine first)
3. Beta testing with 10 customers
4. Public launch

**Estimated Timeline:** 12 weeks
**Team Size:** 3-4 engineers

---

*Yantra Flow - Workflows and apps for everyone, powered by MCPs from developers.*
