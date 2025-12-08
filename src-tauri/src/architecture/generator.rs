// Architecture Generator - AI generation from user intent
// Purpose: Generate architecture diagrams from natural language descriptions
// Created: November 28, 2025

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;
use crate::llm::orchestrator::LLMOrchestrator;
use crate::llm::CodeGenerationRequest;
use super::types::{Architecture, Component, ComponentType, Connection, ConnectionType, Position};

/// Architecture generator that creates architecture from natural language
pub struct ArchitectureGenerator {
    llm: Arc<tokio::sync::Mutex<LLMOrchestrator>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ArchitectureSpec {
    components: Vec<ComponentSpec>,
    connections: Vec<ConnectionSpec>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ComponentSpec {
    name: String,
    component_type: String,
    layer: String,
    description: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ConnectionSpec {
    from: String,
    to: String,
    connection_type: String,
    label: Option<String>,
}

impl ArchitectureGenerator {
    /// Create new generator
    pub fn new(llm: Arc<tokio::sync::Mutex<LLMOrchestrator>>) -> Self {
        Self { llm }
    }

    /// Generate architecture from user intent
    pub async fn generate_from_intent(&self, user_intent: &str) -> Result<Architecture, String> {
        // Create prompt for LLM
        let prompt = self.create_generation_prompt(user_intent);

        // Call LLM
        let response = {
            let llm = self.llm.lock().await;
            let request = CodeGenerationRequest {
                intent: prompt.clone(),
                context: vec![],
                file_path: None,
                dependencies: vec![],
            };
            llm.generate_code(&request).await
                .map_err(|e| format!("LLM call failed: {}", e))?
        };

        // Parse JSON response
        let spec: ArchitectureSpec = serde_json::from_str(&response.code)
            .map_err(|e| format!("Failed to parse LLM response: {}", e))?;

        // Convert to Architecture
        let architecture = self.spec_to_architecture(spec, user_intent)?;

        Ok(architecture)
    }

    /// Create prompt for architecture generation
    fn create_generation_prompt(&self, user_intent: &str) -> String {
        format!(
            r#"Generate a software architecture diagram from the following user description.

User Intent: "{}"

Analyze the requirements and generate a complete architecture with components and connections.

Rules:
1. Identify all major components (services, modules, databases, external systems, UI layers)
2. Determine appropriate component types: service, module, layer, database, external, ui_component
3. Assign layers: frontend, backend, database, external, infrastructure
4. Infer all connections between components
5. Specify connection types: data_flow, api_call, event, dependency

Output as JSON in this exact format:
{{
  "components": [
    {{
      "name": "Component Name",
      "component_type": "service|module|layer|database|external|ui_component",
      "layer": "frontend|backend|database|external|infrastructure",
      "description": "Brief description of component's purpose"
    }}
  ],
  "connections": [
    {{
      "from": "Source Component Name",
      "to": "Target Component Name",
      "connection_type": "data_flow|api_call|event|dependency",
      "label": "Optional connection label"
    }}
  ]
}}

Example for "Build a REST API with JWT authentication":
{{
  "components": [
    {{
      "name": "API Gateway",
      "component_type": "service",
      "layer": "backend",
      "description": "Main entry point for HTTP requests, routes to services"
    }},
    {{
      "name": "Auth Service",
      "component_type": "service",
      "layer": "backend",
      "description": "Handles JWT token generation and validation"
    }},
    {{
      "name": "User Service",
      "component_type": "service",
      "layer": "backend",
      "description": "Manages user CRUD operations"
    }},
    {{
      "name": "PostgreSQL",
      "component_type": "database",
      "layer": "database",
      "description": "Persistent data storage"
    }}
  ],
  "connections": [
    {{
      "from": "API Gateway",
      "to": "Auth Service",
      "connection_type": "api_call",
      "label": "Verify JWT"
    }},
    {{
      "from": "API Gateway",
      "to": "User Service",
      "connection_type": "api_call",
      "label": "Route requests"
    }},
    {{
      "from": "Auth Service",
      "to": "PostgreSQL",
      "connection_type": "data_flow",
      "label": "Store tokens"
    }},
    {{
      "from": "User Service",
      "to": "PostgreSQL",
      "connection_type": "data_flow",
      "label": "Store users"
    }}
  ]
}}

Now generate the architecture JSON for the user's intent:"#,
            user_intent
        )
    }

    /// Convert specification to Architecture
    fn spec_to_architecture(
        &self,
        spec: ArchitectureSpec,
        user_intent: &str,
    ) -> Result<Architecture, String> {
        let id = uuid::Uuid::new_v4().to_string();
        let timestamp = chrono::Utc::now().timestamp();

        // Convert components
        let mut components = Vec::new();
        let mut component_map = std::collections::HashMap::new();
        
        for (index, comp_spec) in spec.components.iter().enumerate() {
            let component_id = format!("comp_{}", index + 1);
            component_map.insert(comp_spec.name.clone(), component_id.clone());

            let component = Component {
                id: component_id,
                name: comp_spec.name.clone(),
                description: comp_spec.description.clone(),
                component_type: ComponentType::Planned,
                category: comp_spec.layer.clone(),  // layer maps to category
                position: Position {
                    x: (index % 3) as f64 * 250.0 + 50.0,
                    y: (index / 3) as f64 * 150.0 + 50.0,
                },
                files: Vec::new(),
                metadata: HashMap::new(),
                created_at: timestamp,
                updated_at: timestamp,
            };
            components.push(component);
        }

        // Convert connections
        let mut connections = Vec::new();
        for (index, conn_spec) in spec.connections.iter().enumerate() {
            let source_id = component_map
                .get(&conn_spec.from)
                .ok_or_else(|| format!("Unknown source component: {}", conn_spec.from))?
                .clone();
            
            let target_id = component_map
                .get(&conn_spec.to)
                .ok_or_else(|| format!("Unknown target component: {}", conn_spec.to))?
                .clone();

            let connection = Connection {
                id: format!("conn_{}", index + 1),
                source_id,
                target_id,
                connection_type: Self::parse_connection_type(&conn_spec.connection_type)?,
                description: conn_spec.label.clone().unwrap_or_default(),  // label maps to description
                metadata: HashMap::new(),
                created_at: timestamp,
                updated_at: timestamp,
            };
            connections.push(connection);
        }

        Ok(Architecture {
            id,
            name: "Generated Architecture".to_string(),
            description: user_intent.to_string(),
            components,
            connections,
            metadata: {
                let mut map = HashMap::new();
                map.insert("generation_method".to_string(), "llm_intent".to_string());
                map.insert("user_intent".to_string(), user_intent.to_string());
                map
            },
            created_at: timestamp,
            updated_at: timestamp,
        })
    }

    /// Parse connection type from string
    fn parse_connection_type(s: &str) -> Result<ConnectionType, String> {
        match s.to_lowercase().as_str() {
            "data_flow" => Ok(ConnectionType::DataFlow),
            "api_call" => Ok(ConnectionType::ApiCall),
            "event" => Ok(ConnectionType::Event),
            "dependency" => Ok(ConnectionType::Dependency),
            _ => Err(format!("Unknown connection type: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{LLMConfig, LLMProvider};

    #[tokio::test]
    async fn test_prompt_generation() {
        let config = LLMConfig {
            claude_api_key: None,
            openai_api_key: None,
            openrouter_api_key: None,
            groq_api_key: None,
            gemini_api_key: None,
            primary_provider: LLMProvider::Claude,
            secondary_provider: None,
            max_retries: 3,
            timeout_seconds: 30,
            selected_models: vec![],
        };
        let llm = Arc::new(tokio::sync::Mutex::new(LLMOrchestrator::new(config)));
        let generator = ArchitectureGenerator::new(llm);
        
        let prompt = generator.create_generation_prompt("Build a REST API");
        assert!(prompt.contains("REST API"));
        assert!(prompt.contains("component_type"));
        assert!(prompt.contains("connections"));
    }

    #[test]
    #[ignore = "parse_component_type method needs to be implemented"]
    fn test_parse_component_type() {
        // Component type parsing will be implemented when needed
        // assert!(matches!(
        //     ArchitectureGenerator::parse_component_type("planned"),
        //     Ok(ComponentType::Planned)
        // ));
    }

    #[test]
    #[ignore = "parse_layer method needs to be implemented"]
    fn test_parse_layer() {
        // assert!(matches!(
        //     ArchitectureGenerator::parse_layer("frontend"),
        //     Ok(Layer::Frontend)
        // ));
        // assert!(matches!(
        //     ArchitectureGenerator::parse_layer("backend"),
        //     Ok(Layer::Backend)
        // ));
    }

    #[test]
    fn test_parse_connection_type() {
        assert!(matches!(
            ArchitectureGenerator::parse_connection_type("api_call"),
            Ok(ConnectionType::ApiCall)
        ));
        assert!(matches!(
            ArchitectureGenerator::parse_connection_type("data_flow"),
            Ok(ConnectionType::DataFlow)
        ));
    }
}
