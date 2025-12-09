# Yantra Data - Technical Specification

**Version:** 1.0  
**Date:** December 9, 2025  
**Status:** Specification - Ready for Implementation  
**Phase:** Cloud Product #2 - AI-Powered Data Platform

---

## 1. Executive Summary

### 1.1 Vision

Yantra Data is an AI-powered data platform that enables users to query, analyze, and visualize data from any source without ETL, SQL, or data warehouses. It creates a virtual unified schema across all connected MCPs and uses AI agents to orchestrate complex cross-source queries.

### 1.2 Problem Statement

**Current Challenges:**

- Data scattered across multiple systems (databases, SaaS apps, APIs)
- Traditional BI tools require expensive data warehouses
- ETL pipelines are fragile and slow
- SQL knowledge required for analysis
- Each tool operates independently (can't join Stripe + Salesforce)
- Freshness issues (data is stale in warehouse)

**Yantra Data Solution:**

- **Virtual Schema:** Automatically discovers and maps data across all MCPs
- **No ETL:** Queries sources directly in real-time
- **AI Orchestration:** Natural language queries, AI handles complexity
- **Smart Joins:** Automatically joins data from different sources
- **Always Fresh:** Real-time queries, no stale data
- **Unified Platform:** Uses same MCPs as Yantra Develop and Flow

### 1.3 Success Criteria

**MVP Success Metrics:**

- ✅ Answer natural language query in <5 seconds
- ✅ Auto-discover schema from 10+ different MCPs
- ✅ Successfully join data across 3+ sources
- ✅ Generate dashboard from description in <30 seconds
- ✅ 95%+ query accuracy (correct data, correct joins)

---

## 2. Product Overview

### 2.1 Core Features

```
┌──────────────────────────────────────────────────────────┐
│                    YANTRA DATA                            │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  1. Virtual Schema Engine                           │ │
│  │     • Auto-discover schema from all MCPs            │ │
│  │     • Infer relationships across sources            │ │
│  │     • Semantic understanding (customer, revenue)    │ │
│  │     • Type normalization (dates, currency)          │ │
│  │     • Join key detection                            │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  2. AI Query Engine                                 │ │
│  │     • Natural language to query                     │ │
│  │     • Intelligent tool selection                    │ │
│  │     • Query planning & optimization                 │ │
│  │     • Parallel execution                            │ │
│  │     • Smart joins (in-memory, pushed-down)          │ │
│  │     • Result caching                                │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  3. Visualization & Dashboards                      │ │
│  │     • Auto-generate charts from data                │ │
│  │     • Interactive dashboards                        │ │
│  │     • Real-time updates                             │ │
│  │     • Custom metrics & KPIs                         │ │
│  │     • Export (CSV, Excel, PDF)                      │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  4. Scheduled Reports & Alerts                      │ │
│  │     • Schedule queries (daily, weekly)              │ │
│  │     • Email/Slack delivery                          │ │
│  │     • Threshold alerts                              │ │
│  │     • Anomaly detection                             │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Virtual Schema Engine

### 3.1 Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Virtual Schema Engine                                   │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Schema Discovery                                   │ │
│  │  ───────────────────                                │ │
│  │  When MCP is connected:                             │ │
│  │  1. Query MCP for schema                            │ │
│  │  2. Extract tables/collections                      │ │
│  │  3. Extract fields/columns with types               │ │
│  │  4. Extract sample data (100 rows)                  │ │
│  │  5. Detect relationships (FKs, common fields)       │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Entity Recognition                                 │ │
│  │  ─────────────────────                              │ │
│  │  AI analyzes schemas to identify:                   │ │
│  │  • Entities (Customer, Order, Product, etc.)        │ │
│  │  • Common fields (email, customer_id, user_id)      │ │
│  │  • Semantic meaning (revenue vs amount vs price)    │ │
│  │  • Temporal fields (created_at, updated_at)         │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Relationship Inference                             │ │
│  │  ──────────────────────                             │ │
│  │  AI infers relationships:                           │ │
│  │  • Same entity across MCPs (by field similarity)    │ │
│  │  • One-to-Many (orders → customer)                  │ │
│  │  • Many-to-Many (products ↔ orders via junction)    │ │
│  │  • Join keys (email, IDs, foreign keys)             │ │
│  │  • Confidence score (0.0-1.0)                       │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Unified Schema                                     │ │
│  │  ──────────────                                     │ │
│  │  Virtual entities that span multiple MCPs:          │ │
│  │                                                      │ │
│  │  Customer (virtual entity):                         │ │
│  │  - id (from DB.users.id)                            │ │
│  │  - email (from DB.users.email)                      │ │
│  │  - stripe_id (from Stripe.customers.id)             │ │
│  │  - salesforce_id (from SF.contacts.Id)              │ │
│  │  - total_revenue (computed from Stripe payments)    │ │
│  │  - support_tickets (count from Zendesk)             │ │
│  │                                                      │ │
│  │  User queries "Customer" → System knows how to      │ │
│  │  query multiple MCPs and join results               │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### 3.2 Data Model

```rust
// src-data/src/schema/mod.rs

/// Virtual entity that spans multiple data sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualEntity {
    pub name: String,                  // "Customer"
    pub description: String,
    pub fields: Vec<VirtualField>,
    pub source_mappings: Vec<SourceMapping>,
    pub relationships: Vec<EntityRelationship>,
}

/// Field in virtual entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualField {
    pub name: String,                  // "email"
    pub field_type: DataType,          // String, Int, Float, Boolean, Date, etc.
    pub semantic_type: SemanticType,   // Email, Currency, Percentage, ID, etc.
    pub sources: Vec<FieldSource>,     // Where this field comes from
    pub computed: Option<Computation>, // For computed fields
    pub nullable: bool,
}

/// Source of a field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSource {
    pub tool_id: String,               // "mcp-database"
    pub table: String,                 // "users"
    pub column: String,                // "email"
    pub transformation: Option<String>, // "lowercase", "trim", etc.
    pub confidence: f32,               // 0.0-1.0 (how sure we are about this mapping)
}

/// Computed field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Computation {
    pub expression: String,            // "SUM(payments.amount)"
    pub dependencies: Vec<String>,     // ["payments"]
    pub cache_duration_seconds: Option<u64>,
}

/// Relationship between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRelationship {
    pub from_entity: String,           // "Customer"
    pub to_entity: String,             // "Order"
    pub relationship_type: RelationshipType,
    pub join_keys: Vec<JoinKey>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    OneToOne,
    OneToMany,
    ManyToMany,
}

/// Join key between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinKey {
    pub from_field: String,            // "id"
    pub to_field: String,              // "customer_id"
    pub join_type: JoinType,           // Inner, Left, Right, Full
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
}

/// Semantic types (for AI understanding)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SemanticType {
    Identifier,        // Primary key, foreign key
    Email,
    Phone,
    Currency,
    Percentage,
    URL,
    IPAddress,
    Timestamp,
    Duration,
    Count,
    Other(String),
}

/// Data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    String,
    Integer,
    Float,
    Boolean,
    Date,
    DateTime,
    JSON,
    Array(Box<DataType>),
}

/// Mapping between MCP source and virtual entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMapping {
    pub tool_id: String,
    pub resource_uri: String,          // "database://users"
    pub entity_name: String,           // "Customer"
    pub field_mappings: HashMap<String, String>, // virtual_field -> source_field
}
```

### 3.3 Schema Discovery Process

```rust
// src-data/src/schema/discovery.rs

pub struct SchemaDiscovery {
    tool_registry: Arc<ToolRegistry>,
    llm_client: Arc<LLMClient>,
    schema_store: Arc<SchemaStore>,
}

impl SchemaDiscovery {
    /// Discover schema from a newly connected MCP
    pub async fn discover_schema(
        &self,
        tool_id: &str,
        organization_id: &str,
    ) -> Result<Vec<VirtualEntity>, DiscoveryError> {
        // 1. Get tool
        let tool = self.tool_registry.get(tool_id).await
            .ok_or(DiscoveryError::ToolNotFound)?;

        // 2. Query tool for resources
        let resources = tool.capabilities().await?.resources;

        // 3. For each resource, get schema and sample data
        let mut discovered_tables = Vec::new();

        for resource in resources {
            let schema = self.get_resource_schema(tool, &resource).await?;
            let sample_data = self.get_sample_data(tool, &resource, 100).await?;

            discovered_tables.push(DiscoveredTable {
                tool_id: tool_id.to_string(),
                resource_uri: resource.uri.clone(),
                name: resource.name.clone(),
                schema,
                sample_data,
            });
        }

        // 4. Use AI to identify entities and relationships
        let entities = self.identify_entities(&discovered_tables).await?;

        // 5. Infer relationships across all tables
        let relationships = self.infer_relationships(&entities, &discovered_tables).await?;

        // 6. Create/update virtual entities
        for entity in &entities {
            self.schema_store.upsert_entity(organization_id, entity).await?;
        }

        Ok(entities)
    }

    async fn identify_entities(
        &self,
        tables: &[DiscoveredTable],
    ) -> Result<Vec<VirtualEntity>, DiscoveryError> {
        let prompt = format!(
            r#"Analyze these database tables and identify logical entities.

Tables:
{}

For each table, identify:
1. What real-world entity it represents (Customer, Order, Product, etc.)
2. Key fields (ID, email, name, etc.)
3. Semantic types (email, currency, timestamp, etc.)

Return JSON array of entities:
[{{
  "name": "Customer",
  "description": "...",
  "tables": [{{
    "tool_id": "mcp-database",
    "table": "users",
    "key_fields": ["id", "email", "name"]
  }}],
  "semantic_fields": {{
    "email": "email_address",
    "created_at": "timestamp"
  }}
}}]
"#,
            tables.iter().map(|t| format!("{}: {}", t.name, serde_json::to_string_pretty(&t.schema).unwrap())).collect::<Vec<_>>().join("\n\n")
        );

        let response = self.llm_client.complete(prompt).await?;
        let entities: Vec<EntityDefinition> = serde_json::from_str(&response.text)?;

        // Convert to VirtualEntity
        let virtual_entities = entities.into_iter().map(|e| self.create_virtual_entity(e, tables)).collect();

        Ok(virtual_entities)
    }

    async fn infer_relationships(
        &self,
        entities: &[VirtualEntity],
        tables: &[DiscoveredTable],
    ) -> Result<Vec<EntityRelationship>, DiscoveryError> {
        let prompt = format!(
            r#"Analyze these entities and infer relationships between them.

Entities:
{}

Tables (with FKs and common fields):
{}

Identify:
1. Relationships (one-to-one, one-to-many, many-to-many)
2. Join keys (which fields can be used to join entities)
3. Confidence (how sure you are about each relationship)

Look for:
- Foreign key constraints
- Field name similarities (user_id, customer_id, etc.)
- Common field values (same emails, IDs in multiple tables)

Return JSON array:
[{{
  "from_entity": "Order",
  "to_entity": "Customer",
  "type": "many_to_one",
  "join_keys": [{{
    "from_field": "customer_id",
    "to_field": "id"
  }}],
  "confidence": 0.95
}}]
"#,
            serde_json::to_string_pretty(&entities).unwrap(),
            serde_json::to_string_pretty(&tables).unwrap()
        );

        let response = self.llm_client.complete(prompt).await?;
        let relationships: Vec<EntityRelationship> = serde_json::from_str(&response.text)?;

        Ok(relationships)
    }
}

struct DiscoveredTable {
    tool_id: String,
    resource_uri: String,
    name: String,
    schema: serde_json::Value,
    sample_data: Vec<serde_json::Value>,
}
```

---

## 4. AI Query Engine

### 4.1 Query Flow

```
User: "Show me high-value customers who opened support
       tickets last month"

↓

1. Intent Understanding
   - Entities: Customer
   - Filters: high-value (>$10k LTV), support tickets (last 30 days)
   - Output: Table

↓

2. Tool Selection
   - Need customer data: Database MCP or Salesforce MCP?
   - Need revenue data: Stripe MCP
   - Need ticket data: Zendesk MCP

↓

3. Query Planning
   Step 1: Get high-value customers from Stripe
           Query: SUM(payments) GROUP BY customer_id HAVING total > 10000
   Step 2: Get recent tickets from Zendesk
           Query: COUNT(tickets) WHERE created > 30d ago GROUP BY user_email
   Step 3: Join on email (Stripe customer.email ↔ Zendesk user.email)
   Step 4: Enrich with customer details from Database

↓

4. Parallel Execution
   [Stripe Query] + [Zendesk Query] + [Database Query]
   (All run simultaneously)

↓

5. Smart Join
   Join results in memory using virtual schema mappings

↓

6. Result + Insights
   - Table: Name, Email, Revenue, Tickets, Last Ticket Date
   - Insight: "23 high-value customers with high support volume"
   - Suggestion: "Consider dedicated account manager"
```

### 4.2 Query Engine Implementation

```rust
// src-data/src/query/engine.rs

pub struct QueryEngine {
    tool_registry: Arc<ToolRegistry>,
    virtual_schema: Arc<VirtualSchema>,
    llm_client: Arc<LLMClient>,
    cache: Arc<QueryCache>,
    planner: Arc<QueryPlanner>,
}

impl QueryEngine {
    pub async fn execute_natural_language_query(
        &self,
        query: &str,
        context: &ExecutionContext,
    ) -> Result<QueryResult, QueryError> {
        // 1. Parse intent
        let intent = self.parse_query_intent(query, context).await?;

        // 2. Create query plan
        let plan = self.planner.create_plan(&intent, &self.virtual_schema).await?;

        // 3. Optimize plan
        let optimized_plan = self.optimize_plan(plan)?;

        // 4. Execute plan
        let raw_results = self.execute_plan(&optimized_plan, context).await?;

        // 5. Join results (if multiple sources)
        let joined_data = self.join_results(&raw_results, &optimized_plan)?;

        // 6. Generate insights
        let insights = self.generate_insights(&joined_data, &intent).await?;

        // 7. Choose visualization
        let visualization = self.suggest_visualization(&joined_data, &intent)?;

        Ok(QueryResult {
            data: joined_data,
            insights,
            visualization,
            execution_stats: ExecutionStats {
                total_time_ms: 0,
                sources_queried: raw_results.len(),
                rows_returned: joined_data.len(),
                cached: false,
            },
        })
    }

    async fn parse_query_intent(
        &self,
        query: &str,
        context: &ExecutionContext,
    ) -> Result<QueryIntent, QueryError> {
        // Get available entities from virtual schema
        let entities = self.virtual_schema.list_entities(context.organization_id).await?;

        let prompt = format!(
            r#"Parse this data query and extract structured intent.

User query: "{}"

Available entities: {}

Extract:
1. Entities involved (Customer, Order, Product, etc.)
2. Filters (date ranges, comparisons, etc.)
3. Aggregations (SUM, COUNT, AVG, etc.)
4. Grouping
5. Sorting
6. Output format (table, chart, single value)

Return JSON:
{{
  "entities": ["Customer"],
  "filters": [
    {{ "field": "lifetime_value", "operator": ">", "value": 10000 }},
    {{ "field": "support_tickets.created_at", "operator": ">", "value": "30d ago" }}
  ],
  "fields": ["name", "email", "lifetime_value", "ticket_count"],
  "group_by": [],
  "order_by": [{{ "field": "lifetime_value", "direction": "desc" }}],
  "limit": 100,
  "output_format": "table"
}}
"#,
            query,
            entities.iter().map(|e| e.name.clone()).collect::<Vec<_>>().join(", ")
        );

        let response = self.llm_client.complete(prompt).await?;
        let intent: QueryIntent = serde_json::from_str(&response.text)?;

        Ok(intent)
    }

    async fn execute_plan(
        &self,
        plan: &QueryPlan,
        context: &ExecutionContext,
    ) -> Result<Vec<ToolQueryResult>, QueryError> {
        let mut results = Vec::new();

        // Execute tool queries in parallel
        let futures: Vec<_> = plan.tool_queries.iter().map(|tq| {
            self.execute_tool_query(tq, context)
        }).collect();

        let query_results = futures::future::join_all(futures).await;

        for result in query_results {
            match result {
                Ok(r) => results.push(r),
                Err(e) => {
                    // Log error but continue (partial results better than none)
                    warn!("Tool query failed: {}", e);
                }
            }
        }

        Ok(results)
    }

    async fn execute_tool_query(
        &self,
        query: &ToolQuery,
        context: &ExecutionContext,
    ) -> Result<ToolQueryResult, QueryError> {
        // Check cache
        let cache_key = self.cache.generate_key(query);
        if let Some(cached) = self.cache.get(&cache_key).await? {
            return Ok(ToolQueryResult {
                tool_id: query.tool_id.clone(),
                data: cached,
                cached: true,
            });
        }

        // Execute query via tool
        let result = self.tool_registry.execute(
            &query.tool_id,
            &query.action,
            query.parameters.clone(),
            context,
        ).await?;

        // Cache result
        self.cache.set(&cache_key, &result.data, Duration::from_secs(300)).await?;

        Ok(ToolQueryResult {
            tool_id: query.tool_id.clone(),
            data: result.data,
            cached: false,
        })
    }

    fn join_results(
        &self,
        results: &[ToolQueryResult],
        plan: &QueryPlan,
    ) -> Result<Vec<serde_json::Value>, QueryError> {
        if results.len() == 1 {
            // Single source, no join needed
            return Ok(results[0].data.as_array()
                .ok_or(QueryError::InvalidResultFormat)?
                .clone());
        }

        // Multi-source, need to join
        let mut joined = Vec::new();

        // Start with first result
        let mut current_data = results[0].data.as_array()
            .ok_or(QueryError::InvalidResultFormat)?
            .clone();

        // Join with each subsequent result
        for (i, result) in results.iter().skip(1).enumerate() {
            let join_spec = &plan.joins[i];

            let right_data = result.data.as_array()
                .ok_or(QueryError::InvalidResultFormat)?;

            current_data = self.perform_join(
                &current_data,
                right_data,
                join_spec,
            )?;
        }

        Ok(current_data)
    }

    fn perform_join(
        &self,
        left: &[serde_json::Value],
        right: &[serde_json::Value],
        join_spec: &JoinSpec,
    ) -> Result<Vec<serde_json::Value>, QueryError> {
        let mut result = Vec::new();

        // Build index on right side for faster lookup
        let mut right_index: HashMap<String, Vec<&serde_json::Value>> = HashMap::new();

        for row in right {
            let key = self.extract_join_key(row, &join_spec.right_field)?;
            right_index.entry(key).or_default().push(row);
        }

        // Join
        for left_row in left {
            let left_key = self.extract_join_key(left_row, &join_spec.left_field)?;

            if let Some(matching_right_rows) = right_index.get(&left_key) {
                for right_row in matching_right_rows {
                    // Merge rows
                    let mut merged = left_row.as_object().unwrap().clone();

                    for (k, v) in right_row.as_object().unwrap() {
                        merged.insert(k.clone(), v.clone());
                    }

                    result.push(serde_json::Value::Object(merged));
                }
            } else if join_spec.join_type == JoinType::Left {
                // Left join: include left row even without match
                result.push(left_row.clone());
            }
        }

        Ok(result)
    }

    async fn generate_insights(
        &self,
        data: &[serde_json::Value],
        intent: &QueryIntent,
    ) -> Result<Vec<String>, QueryError> {
        let prompt = format!(
            r#"Analyze this query result and generate insights.

User query intent: {:?}

Result (sample of first 10 rows):
{}

Generate 3-5 insights such as:
- Key findings
- Trends
- Anomalies
- Recommendations

Return JSON array of insight strings.
"#,
            intent,
            serde_json::to_string_pretty(&data.iter().take(10).collect::<Vec<_>>()).unwrap()
        );

        let response = self.llm_client.complete(prompt).await?;
        let insights: Vec<String> = serde_json::from_str(&response.text)?;

        Ok(insights)
    }
}

#[derive(Debug, Clone)]
pub struct QueryIntent {
    pub entities: Vec<String>,
    pub filters: Vec<Filter>,
    pub fields: Vec<String>,
    pub group_by: Vec<String>,
    pub order_by: Vec<OrderBy>,
    pub limit: Option<usize>,
    pub output_format: OutputFormat,
}

#[derive(Debug, Clone)]
pub struct Filter {
    pub field: String,
    pub operator: FilterOperator,
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FilterOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Contains,
    In,
}

#[derive(Debug, Clone)]
pub struct OrderBy {
    pub field: String,
    pub direction: SortDirection,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SortDirection {
    Asc,
    Desc,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OutputFormat {
    Table,
    Chart,
    SingleValue,
}

#[derive(Debug, Clone)]
pub struct QueryPlan {
    pub tool_queries: Vec<ToolQuery>,
    pub joins: Vec<JoinSpec>,
    pub post_processing: Vec<PostProcessStep>,
}

#[derive(Debug, Clone)]
pub struct ToolQuery {
    pub tool_id: String,
    pub action: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct JoinSpec {
    pub left_field: String,
    pub right_field: String,
    pub join_type: JoinType,
}

#[derive(Debug, Clone)]
pub enum PostProcessStep {
    Filter(Filter),
    Sort(OrderBy),
    Limit(usize),
    Aggregate { field: String, function: AggregateFunction },
}

#[derive(Debug, Clone)]
pub enum AggregateFunction {
    Sum,
    Avg,
    Count,
    Min,
    Max,
}

#[derive(Debug)]
pub struct QueryResult {
    pub data: Vec<serde_json::Value>,
    pub insights: Vec<String>,
    pub visualization: VisualizationType,
    pub execution_stats: ExecutionStats,
}

#[derive(Debug)]
pub struct ExecutionStats {
    pub total_time_ms: u64,
    pub sources_queried: usize,
    pub rows_returned: usize,
    pub cached: bool,
}

#[derive(Debug, Clone)]
pub enum VisualizationType {
    Table,
    LineChart,
    BarChart,
    PieChart,
    SingleValue,
}
```

---

## 5. User Interface

### 5.1 Query Interface

```
┌──────────────────────────────────────────────────────────┐
│  Yantra Data                              [User ▼]       │
├──────────────────────────────────────────────────────────┤
│  [Query]  [Dashboards]  [Reports]  [Sources]            │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Ask anything about your data...                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Show me revenue by customer segment                  │ │
│  └─────────────────────────────────────────────────────┘ │
│  [Ask] or press Enter                                    │
│                                                           │
│  💡 Suggestions:                                         │
│  • Show me top 10 customers by revenue                   │
│  • What's our churn rate this month?                     │
│  • Compare sales this quarter vs last quarter            │
│                                                           │
│  ────────────────────────────────────────────────────────│
│                                                           │
│  Results:                                                │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Revenue by Customer Segment                         │ │
│  │                                                       │ │
│  │  Segment      Revenue        Customers    Avg/Cust  │ │
│  │  ─────────────────────────────────────────────────── │ │
│  │  Enterprise   $1,234,567     45           $27,435   │ │
│  │  SMB          $456,789       123          $3,713    │ │
│  │  Startup      $123,456       234          $528      │ │
│  │                                                       │ │
│  │  [📊 Chart View]  [📥 Export CSV]  [⭐ Save]       │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  💡 Insights:                                            │
│  • Enterprise segment generates 65% of total revenue     │
│  • Startup segment has lowest average but growing 15%   │
│  • Consider upselling SMB customers to Enterprise        │
│                                                           │
│  📊 Suggested Visualizations:                           │
│  [Bar Chart]  [Pie Chart]  [Trend Line]                 │
└──────────────────────────────────────────────────────────┘
```

### 5.2 Dashboard Builder

```
┌──────────────────────────────────────────────────────────┐
│  Create Dashboard                                         │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Describe your dashboard:                                │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Executive dashboard with revenue, customer growth,   │ │
│  │ churn rate, and top products                         │ │
│  └─────────────────────────────────────────────────────┘ │
│  [Generate Dashboard]                                    │
│                                                           │
│  ────────────────────────────────────────────────────────│
│                                                           │
│  Generated: Executive Dashboard                          │
│                                                           │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │ 💰 Revenue      │  │ 👥 Customers    │              │
│  │ $1.2M           │  │ 402             │              │
│  │ +15% ↗          │  │ +23 this month  │              │
│  └─────────────────┘  └─────────────────┘              │
│                                                           │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │ 📉 Churn Rate   │  │ ⭐ Top Products  │              │
│  │ 3.2%            │  │ 1. Pro Plan     │              │
│  │ -0.5% ↘         │  │ 2. Enterprise   │              │
│  └─────────────────┘  │ 3. Starter      │              │
│                        └─────────────────┘              │
│                                                           │
│  ┌──────────────────────────────────────────────────────┐│
│  │  Revenue Trend (Last 12 Months)                      ││
│  │  [Line chart showing revenue growth]                 ││
│  └──────────────────────────────────────────────────────┘│
│                                                           │
│  [Edit Panels]  [Save Dashboard]  [Schedule Report]     │
└──────────────────────────────────────────────────────────┘
```

### 5.3 Data Sources View

```
┌──────────────────────────────────────────────────────────┐
│  Data Sources                                             │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Connected (6):                                          │
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 💾 Production Database                              │  │
│  │ PostgreSQL • mcp-database                           │  │
│  │                                                      │  │
│  │ Tables: users, orders, products, invoices           │  │
│  │ Last synced: 5 minutes ago                          │  │
│  │ Rows: ~2.3M                                         │  │
│  │                                                      │  │
│  │ [View Schema]  [Refresh]  [Settings]                │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 💳 Stripe                                           │  │
│  │ Payment Processing • mcp-stripe                     │  │
│  │                                                      │  │
│  │ Resources: customers, payments, subscriptions       │  │
│  │ Last synced: 2 minutes ago                          │  │
│  │ Payments this month: 1,234                          │  │
│  │                                                      │  │
│  │ [View Schema]  [Refresh]  [Settings]                │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  Virtual Entities (Auto-discovered):                     │
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 👤 Customer                                         │  │
│  │ Unified view across 3 sources                       │  │
│  │                                                      │  │
│  │ Fields: id, email, name, created_at, segment,       │  │
│  │         lifetime_value, support_tickets              │  │
│  │                                                      │  │
│  │ Sources:                                            │  │
│  │ • Database (users table)                            │  │
│  │ • Stripe (customers)                                │  │
│  │ • Salesforce (contacts)                             │  │
│  │                                                      │  │
│  │ [View Details]  [Edit Mappings]                     │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## 6. Dashboards & Reports

### 6.1 Dashboard Definition

```typescript
interface Dashboard {
  id: string;
  organization_id: string;
  name: string;
  description: string;

  // Layout
  panels: Panel[];
  layout: Layout;

  // Refresh
  auto_refresh: boolean;
  refresh_interval_seconds: number;

  // Access
  visibility: 'private' | 'team' | 'organization';
  shared_with: string[];

  created_by: string;
  created_at: string;
  updated_at: string;
}

interface Panel {
  id: string;
  title: string;
  type: PanelType;
  query: string; // Natural language query
  visualization: VisualizationType;

  // Position in grid
  x: number;
  y: number;
  width: number;
  height: number;

  // Refresh
  refresh_interval_seconds?: number;
}

type PanelType = 'metric' | 'chart' | 'table' | 'text';

interface Layout {
  columns: number; // Grid columns (e.g., 12)
  row_height: number; // Height of one row (e.g., 50px)
}
```

### 6.2 Scheduled Reports

```rust
// src-data/src/reports/scheduler.rs

pub struct ReportScheduler {
    dashboard_store: Arc<DashboardStore>,
    query_engine: Arc<QueryEngine>,
    notification_service: Arc<NotificationService>,
}

impl ReportScheduler {
    pub async fn schedule_report(
        &self,
        config: ReportScheduleConfig,
    ) -> Result<String, SchedulerError> {
        // Create cron job
        let job_id = Uuid::new_v4().to_string();

        let schedule = Schedule::from_str(&config.cron_expression)?;

        // Register job
        let dashboard_id = config.dashboard_id.clone();
        let recipients = config.recipients.clone();
        let format = config.format.clone();

        let query_engine = self.query_engine.clone();
        let notification_service = self.notification_service.clone();

        tokio::spawn(async move {
            for datetime in schedule.upcoming(Utc) {
                let now = Utc::now();
                let duration_until = datetime.signed_duration_since(now);

                if duration_until.num_seconds() < 0 {
                    continue;
                }

                tokio::time::sleep(duration_until.to_std().unwrap()).await;

                // Execute report
                let result = Self::execute_report(
                    &query_engine,
                    &dashboard_id,
                ).await;

                match result {
                    Ok(report_data) => {
                        // Send report
                        let _ = notification_service.send_report(
                            &recipients,
                            &dashboard_id,
                            &report_data,
                            format,
                        ).await;
                    }
                    Err(e) => {
                        error!("Failed to generate report: {}", e);
                    }
                }
            }
        });

        Ok(job_id)
    }

    async fn execute_report(
        query_engine: &QueryEngine,
        dashboard_id: &str,
    ) -> Result<ReportData, SchedulerError> {
        // Execute all panel queries
        // Generate PDF/CSV/etc.
        // Return report data
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct ReportScheduleConfig {
    pub dashboard_id: String,
    pub cron_expression: String,  // "0 9 * * MON" (9am every Monday)
    pub timezone: String,
    pub recipients: Vec<Recipient>,
    pub format: ReportFormat,
}

#[derive(Debug, Clone)]
pub enum Recipient {
    Email(String),
    Slack(String),  // Slack channel
}

#[derive(Debug, Clone, Copy)]
pub enum ReportFormat {
    PDF,
    CSV,
    Excel,
    HTML,
}
```

---

## 7. Pricing & Packaging

```
┌─────────────────────────────────────────────────────────┐
│  Yantra Data Pricing                                    │
│                                                          │
│  Starter: $49/month                                     │
│  • 5 data sources                                       │
│  • 10 dashboards                                        │
│  • 1,000 queries/month                                  │
│  • 5 users                                              │
│  • Email support                                        │
│  • Export to CSV                                        │
│                                                          │
│  Professional: $199/month                               │
│  • 20 data sources                                      │
│  • 50 dashboards                                        │
│  • 10,000 queries/month                                 │
│  • 25 users                                             │
│  • Priority support                                     │
│  • Scheduled reports                                    │
│  • Export to Excel/PDF                                  │
│  • API access                                           │
│                                                          │
│  Enterprise: $999/month                                 │
│  • Unlimited data sources                               │
│  • Unlimited dashboards                                 │
│  • Unlimited queries                                    │
│  • Unlimited users                                      │
│  • Dedicated support + SLA                              │
│  • White-label dashboards                               │
│  • Custom integrations                                  │
│  • Audit logs & compliance                              │
│  • On-premise option                                    │
└─────────────────────────────────────────────────────────┘
```

---

## 8. Success Metrics

### 8.1 Technical Metrics

- Query response time: <5 seconds (p95)
- Schema discovery time: <2 minutes per MCP
- Dashboard generation time: <30 seconds
- Join accuracy: >95%
- Cache hit rate: >60%
- Platform uptime: 99.9%

### 8.2 Business Metrics

- Data sources connected per org: >5
- Queries per user/day: >10
- Dashboards created per org: >3
- Scheduled reports per org: >5
- User satisfaction: >4.5/5

---

**Document Status:** Complete  
**Next Steps:**

1. Review and approve specification
2. Begin implementation (virtual schema engine first)
3. Beta testing with 10 customers
4. Public launch

**Estimated Timeline:** 16 weeks
**Team Size:** 3-4 engineers

---

_Yantra Data - Ask anything, get answers from all your data, powered by AI and MCPs._
