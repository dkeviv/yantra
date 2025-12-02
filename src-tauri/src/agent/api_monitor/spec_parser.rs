use serde_json::Value;
use chrono::Utc;

use super::{ApiContract, SpecFormat, Endpoint};

pub struct SpecParser {}

impl SpecParser {
    pub fn new() -> Self {
        Self {}
    }

    /// Parse an OpenAPI/Swagger spec (JSON or YAML) into an ApiContract.
    /// This is a lightweight, tolerant parser: it extracts paths, methods,
    /// simple request/response JSON schemas and basic metadata.
    pub async fn parse(&self, name: &str, spec_content: &str) -> Result<ApiContract, String> {
        // Try JSON first
        let v: Value = match serde_json::from_str(spec_content) {
            Ok(v) => v,
            Err(_) => {
                // Try YAML (serde_yaml)
                match serde_yaml::from_str::<Value>(spec_content) {
                    Ok(v) => v,
                    Err(e) => return Err(format!("Failed to parse spec as JSON or YAML: {}", e)),
                }
            }
        };

        // Detect format
        let spec_format = if v.get("openapi").is_some() {
            SpecFormat::OpenApiV3
        } else if v.get("swagger").is_some() {
            SpecFormat::Swagger
        } else if v.get("swaggerVersion").is_some() {
            SpecFormat::OpenApiV2
        } else {
            // default to OpenApiV3 if unknown
            SpecFormat::OpenApiV3
        };

        // Extract version
        let version = v.get("info").and_then(|i| i.get("version")).and_then(|s| s.as_str()).unwrap_or("0.0.0").to_string();

        // Extract base_url: prefer servers[0].url (OpenAPI v3), else host/basePath (swagger)
        let base_url = if let Some(servers) = v.get("servers").and_then(|s| s.as_array()) {
            if let Some(s0) = servers.get(0) {
                s0.get("url").and_then(|u| u.as_str()).unwrap_or("").to_string()
            } else { String::new() }
        } else if let Some(host) = v.get("host").and_then(|h| h.as_str()) {
            let base_path = v.get("basePath").and_then(|b| b.as_str()).unwrap_or("");
            format!("{}{}", host, base_path)
        } else {
            // fallback: try servers in vendor extensions
            String::new()
        };

        // Parse paths
        let mut endpoints: std::collections::HashMap<String, Endpoint> = std::collections::HashMap::new();

        if let Some(paths) = v.get("paths").and_then(|p| p.as_object()) {
            for (path, methods) in paths {
                if let Some(methods_obj) = methods.as_object() {
                    for (method, def) in methods_obj {
                        // method could be parameters at path level, skip non-method keys
                        let m = method.to_uppercase();
                        if !["GET","POST","PUT","DELETE","PATCH","HEAD","OPTIONS"].contains(&m.as_str()) {
                            continue;
                        }

                        let mut request_schema: Option<Value> = None;
                        let mut response_schema: Option<Value> = None;
                        let mut required_params: Vec<String> = Vec::new();
                        let mut optional_params: Vec<String> = Vec::new();

                        // Parameters (OpenAPI v2/v3) can be an array
                        if let Some(params) = def.get("parameters").and_then(|p| p.as_array()) {
                            for param in params {
                                if let Some(name) = param.get("name").and_then(|n| n.as_str()) {
                                    let required = param.get("required").and_then(|r| r.as_bool()).unwrap_or(false);
                                    let param_in = param.get("in").and_then(|i| i.as_str()).unwrap_or("");
                                    if required {
                                        required_params.push(name.to_string());
                                    } else {
                                        optional_params.push(name.to_string());
                                    }
                                    // If body param, capture its schema
                                    if param_in == "body" || param.get("schema").is_some() {
                                        if let Some(schema) = param.get("schema") {
                                            request_schema = Some(schema.clone());
                                        }
                                    }
                                }
                            }
                        }

                        // requestBody (OpenAPI v3)
                        if request_schema.is_none() {
                            if let Some(request_body) = def.get("requestBody") {
                                if let Some(content) = request_body.get("content") {
                                    if let Some(app_json) = content.get("application/json") {
                                        if let Some(schema) = app_json.get("schema") {
                                            request_schema = Some(schema.clone());
                                        }
                                    }
                                }
                            }
                        }

                        // responses: try 200, default, or first response with a schema
                        if let Some(responses) = def.get("responses") {
                            if let Some(r200) = responses.get("200").or_else(|| responses.get("201")).or_else(|| responses.get("default")) {
                                // v3: content.application/json.schema
                                if let Some(content) = r200.get("content") {
                                    if let Some(app_json) = content.get("application/json") {
                                        if let Some(schema) = app_json.get("schema") {
                                            response_schema = Some(schema.clone());
                                        }
                                    }
                                }

                                // v2: schema directly under response
                                if response_schema.is_none() {
                                    if let Some(schema) = r200.get("schema") {
                                        response_schema = Some(schema.clone());
                                    }
                                }
                            } else if let Some(map) = responses.as_object() {
                                // pick first response that has a schema
                                for (_code, rv) in map {
                                    if let Some(schema) = rv.get("schema") {
                                        response_schema = Some(schema.clone());
                                        break;
                                    }
                                }
                            }
                        }

                        let endpoint_def = Endpoint {
                            path: path.clone(),
                            method: m.clone(),
                            request_schema,
                            response_schema,
                            required_params,
                            optional_params,
                        };

                        let key = format!("{}:{}", m, path);
                        endpoints.insert(key, endpoint_def);
                    }
                }
            }
        }

        let contract = ApiContract {
            name: name.to_string(),
            base_url,
            version,
            spec_format,
            spec_content: spec_content.to_string(),
            endpoints,
            rate_limit: None,
            last_validated: Utc::now(),
            health_check_url: None,
        };

        Ok(contract)
    }
}

impl Default for SpecParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parse_simple_openapi_json() {
        let parser = SpecParser::new();
        let spec = r#"
        {
            "openapi": "3.0.0",
            "info": { "version": "1.2.3", "title": "Test API" },
            "servers": [ { "url": "https://api.example.com" } ],
            "paths": {
                "/users": {
                    "get": {
                        "responses": {
                            "200": {
                                "description": "OK",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": { "id": { "type": "string" } },
                                            "required": ["id"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        "#;

        let contract = parser.parse("test_api", spec).await.unwrap();
        assert_eq!(contract.version, "1.2.3");
        assert_eq!(contract.base_url, "https://api.example.com");
        let key = "GET:/users".to_string();
        assert!(contract.endpoints.contains_key(&key));
        let ep = contract.endpoints.get(&key).unwrap();
        assert!(ep.response_schema.is_some());
    }
}
