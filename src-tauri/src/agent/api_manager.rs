// API Management: OpenAPI spec import, contract validation, health monitoring
// Purpose: Track external APIs used by generated code, validate contracts, monitor health

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// OpenAPI specification version
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OpenAPIVersion {
    V2, // Swagger 2.0
    V3, // OpenAPI 3.0+
}

/// HTTP method
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    PATCH,
    DELETE,
    HEAD,
    OPTIONS,
}

/// API endpoint definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiEndpoint {
    pub path: String,
    pub method: HttpMethod,
    pub operation_id: Option<String>,
    pub summary: Option<String>,
    pub description: Option<String>,
    pub parameters: Vec<ApiParameter>,
    pub request_body: Option<ApiRequestBody>,
    pub responses: HashMap<String, ApiResponse>,
    pub security: Vec<String>,
}

/// API parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiParameter {
    pub name: String,
    pub location: ParameterLocation,
    pub required: bool,
    pub schema_type: String,
    pub description: Option<String>,
}

/// Parameter location
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ParameterLocation {
    Query,
    Header,
    Path,
    Cookie,
}

/// Request body definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiRequestBody {
    pub required: bool,
    pub content_type: String,
    pub schema: serde_json::Value,
}

/// Response definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse {
    pub description: String,
    pub content_type: Option<String>,
    pub schema: Option<serde_json::Value>,
}

/// Parsed OpenAPI specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiSpec {
    pub version: OpenAPIVersion,
    pub title: String,
    pub description: Option<String>,
    pub base_url: String,
    pub endpoints: Vec<ApiEndpoint>,
    pub security_schemes: HashMap<String, SecurityScheme>,
}

/// Security scheme definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityScheme {
    pub scheme_type: String,
    pub description: Option<String>,
    pub name: Option<String>,
    pub location: Option<String>,
}

/// Contract validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<String>,
}

/// Validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub path: String,
    pub method: HttpMethod,
    pub error_type: String,
    pub message: String,
}

/// API Manager for parsing and validating specs
pub struct ApiManager;

impl ApiManager {
    /// Import OpenAPI spec from file
    pub fn import_spec(file_path: &str) -> Result<ApiSpec, String> {
        let path = Path::new(file_path);
        
        if !path.exists() {
            return Err(format!("File not found: {}", file_path));
        }
        
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        
        // Determine format (JSON or YAML)
        let spec_json: serde_json::Value = if file_path.ends_with(".json") {
            serde_json::from_str(&content)
                .map_err(|e| format!("Invalid JSON: {}", e))?
        } else if file_path.ends_with(".yaml") || file_path.ends_with(".yml") {
            serde_yaml::from_str(&content)
                .map_err(|e| format!("Invalid YAML: {}", e))?
        } else {
            return Err("Unsupported file format. Use .json, .yaml, or .yml".to_string());
        };
        
        Self::parse_spec(spec_json)
    }
    
    /// Parse OpenAPI spec JSON
    fn parse_spec(spec_json: serde_json::Value) -> Result<ApiSpec, String> {
        // Detect version
        let version = if spec_json.get("swagger").is_some() {
            OpenAPIVersion::V2
        } else if spec_json.get("openapi").is_some() {
            OpenAPIVersion::V3
        } else {
            return Err("Invalid OpenAPI spec: missing version field".to_string());
        };
        
        // Extract basic info
        let info = spec_json.get("info")
            .ok_or_else(|| "Missing 'info' field".to_string())?;
        
        let title = info.get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Untitled API")
            .to_string();
        
        let description = info.get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        
        // Extract base URL
        let base_url = match version {
            OpenAPIVersion::V2 => {
                let host = spec_json.get("host")
                    .and_then(|v| v.as_str())
                    .unwrap_or("localhost");
                let base_path = spec_json.get("basePath")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let scheme = spec_json.get("schemes")
                    .and_then(|v| v.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|v| v.as_str())
                    .unwrap_or("https");
                format!("{}://{}{}", scheme, host, base_path)
            }
            OpenAPIVersion::V3 => {
                spec_json.get("servers")
                    .and_then(|v| v.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|v| v.get("url"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("http://localhost")
                    .to_string()
            }
        };
        
        // Parse endpoints
        let paths = spec_json.get("paths")
            .and_then(|v| v.as_object())
            .ok_or_else(|| "Missing 'paths' field".to_string())?;
        
        let mut endpoints = Vec::new();
        
        for (path, path_item) in paths {
            if let Some(operations) = path_item.as_object() {
                for (method_str, operation) in operations {
                    if method_str.starts_with('x') || method_str == "parameters" {
                        continue; // Skip extensions and common parameters
                    }
                    
                    let method = match method_str.to_uppercase().as_str() {
                        "GET" => HttpMethod::GET,
                        "POST" => HttpMethod::POST,
                        "PUT" => HttpMethod::PUT,
                        "PATCH" => HttpMethod::PATCH,
                        "DELETE" => HttpMethod::DELETE,
                        "HEAD" => HttpMethod::HEAD,
                        "OPTIONS" => HttpMethod::OPTIONS,
                        _ => continue,
                    };
                    
                    let endpoint = Self::parse_operation(path, method, operation)?;
                    endpoints.push(endpoint);
                }
            }
        }
        
        // Parse security schemes
        let security_schemes = Self::parse_security_schemes(&spec_json, &version);
        
        Ok(ApiSpec {
            version,
            title,
            description,
            base_url,
            endpoints,
            security_schemes,
        })
    }
    
    /// Parse individual operation
    fn parse_operation(
        path: &str,
        method: HttpMethod,
        operation: &serde_json::Value,
    ) -> Result<ApiEndpoint, String> {
        let operation_id = operation.get("operationId")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        
        let summary = operation.get("summary")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        
        let description = operation.get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        
        // Parse parameters
        let mut parameters = Vec::new();
        if let Some(params) = operation.get("parameters").and_then(|v| v.as_array()) {
            for param in params {
                if let Some(param_obj) = param.as_object() {
                    parameters.push(ApiParameter {
                        name: param_obj.get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        location: match param_obj.get("in").and_then(|v| v.as_str()) {
                            Some("query") => ParameterLocation::Query,
                            Some("header") => ParameterLocation::Header,
                            Some("path") => ParameterLocation::Path,
                            Some("cookie") => ParameterLocation::Cookie,
                            _ => ParameterLocation::Query,
                        },
                        required: param_obj.get("required")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false),
                        schema_type: param_obj.get("type")
                            .or_else(|| param_obj.get("schema").and_then(|v| v.get("type")))
                            .and_then(|v| v.as_str())
                            .unwrap_or("string")
                            .to_string(),
                        description: param_obj.get("description")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                    });
                }
            }
        }
        
        // Parse request body (OpenAPI 3.0)
        let request_body = operation.get("requestBody")
            .and_then(|body| {
                let required = body.get("required")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                
                body.get("content")
                    .and_then(|content| content.as_object())
                    .and_then(|obj| obj.iter().next())
                    .map(|(content_type, schema_obj)| {
                        ApiRequestBody {
                            required,
                            content_type: content_type.to_string(),
                            schema: schema_obj.get("schema")
                                .cloned()
                                .unwrap_or(serde_json::Value::Null),
                        }
                    })
            });
        
        // Parse responses
        let mut responses = HashMap::new();
        if let Some(resp_obj) = operation.get("responses").and_then(|v| v.as_object()) {
            for (status_code, response) in resp_obj {
                let description = response.get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                
                let (content_type, schema) = response.get("content")
                    .and_then(|c| c.as_object())
                    .and_then(|obj| obj.iter().next())
                    .map(|(ct, schema_obj)| {
                        (
                            Some(ct.to_string()),
                            schema_obj.get("schema").cloned(),
                        )
                    })
                    .unwrap_or((None, None));
                
                responses.insert(
                    status_code.to_string(),
                    ApiResponse {
                        description,
                        content_type,
                        schema,
                    },
                );
            }
        }
        
        // Parse security
        let security = operation.get("security")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_object())
                    .flat_map(|obj| obj.keys())
                    .map(|k| k.to_string())
                    .collect()
            })
            .unwrap_or_default();
        
        Ok(ApiEndpoint {
            path: path.to_string(),
            method,
            operation_id,
            summary,
            description,
            parameters,
            request_body,
            responses,
            security,
        })
    }
    
    /// Parse security schemes
    fn parse_security_schemes(
        spec_json: &serde_json::Value,
        version: &OpenAPIVersion,
    ) -> HashMap<String, SecurityScheme> {
        let mut schemes = HashMap::new();
        
        let security_defs = match version {
            OpenAPIVersion::V2 => spec_json.get("securityDefinitions"),
            OpenAPIVersion::V3 => spec_json.get("components")
                .and_then(|v| v.get("securitySchemes")),
        };
        
        if let Some(defs) = security_defs.and_then(|v| v.as_object()) {
            for (name, scheme) in defs {
                if let Some(scheme_obj) = scheme.as_object() {
                    schemes.insert(
                        name.to_string(),
                        SecurityScheme {
                            scheme_type: scheme_obj.get("type")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown")
                                .to_string(),
                            description: scheme_obj.get("description")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string()),
                            name: scheme_obj.get("name")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string()),
                            location: scheme_obj.get("in")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string()),
                        },
                    );
                }
            }
        }
        
        schemes
    }
    
    /// Validate API contract against generated code usage
    pub fn validate_contract(
        spec: &ApiSpec,
        used_endpoints: Vec<(String, HttpMethod)>,
    ) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // Build endpoint lookup
        let mut spec_endpoints = HashMap::new();
        for endpoint in &spec.endpoints {
            spec_endpoints.insert((endpoint.path.clone(), endpoint.method.clone()), endpoint);
        }
        
        // Validate each used endpoint
        for (path, method) in used_endpoints {
            if let Some(endpoint) = spec_endpoints.get(&(path.clone(), method.clone())) {
                // Validate required parameters (would need actual request to validate fully)
                let required_params: Vec<_> = endpoint.parameters.iter()
                    .filter(|p| p.required)
                    .collect();
                
                if !required_params.is_empty() {
                    warnings.push(format!(
                        "{} {} requires {} parameter(s): {}",
                        format!("{:?}", method),
                        path,
                        required_params.len(),
                        required_params.iter()
                            .map(|p| p.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
            } else {
                errors.push(ValidationError {
                    path: path.clone(),
                    method: method.clone(),
                    error_type: "ENDPOINT_NOT_FOUND".to_string(),
                    message: format!("Endpoint {} {} not found in API spec", format!("{:?}", method), path),
                });
            }
        }
        
        ValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_parse_openapi_v3() {
        let spec_json = serde_json::json!({
            "openapi": "3.0.0",
            "info": {
                "title": "Test API",
                "description": "Test API for unit tests",
                "version": "1.0.0"
            },
            "servers": [
                {
                    "url": "https://api.example.com/v1"
                }
            ],
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "listUsers",
                        "summary": "List all users",
                        "parameters": [
                            {
                                "name": "limit",
                                "in": "query",
                                "required": false,
                                "schema": {
                                    "type": "integer"
                                }
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "Successful response",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
        
        let spec = ApiManager::parse_spec(spec_json).unwrap();
        
        assert_eq!(spec.version, OpenAPIVersion::V3);
        assert_eq!(spec.title, "Test API");
        assert_eq!(spec.base_url, "https://api.example.com/v1");
        assert_eq!(spec.endpoints.len(), 1);
        
        let endpoint = &spec.endpoints[0];
        assert_eq!(endpoint.path, "/users");
        assert_eq!(endpoint.method, HttpMethod::GET);
        assert_eq!(endpoint.operation_id, Some("listUsers".to_string()));
        assert_eq!(endpoint.parameters.len(), 1);
        assert_eq!(endpoint.parameters[0].name, "limit");
    }
    
    #[test]
    fn test_import_spec_json() {
        let spec_content = r#"{
            "openapi": "3.0.0",
            "info": {
                "title": "Test API",
                "version": "1.0.0"
            },
            "servers": [{"url": "https://api.test.com"}],
            "paths": {
                "/test": {
                    "get": {
                        "responses": {
                            "200": {"description": "OK"}
                        }
                    }
                }
            }
        }"#;
        
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(spec_content.as_bytes()).unwrap();
        let file_path = temp_file.path().to_str().unwrap().to_string() + ".json";
        
        // Rename to .json extension
        std::fs::copy(temp_file.path(), &file_path).unwrap();
        
        let spec = ApiManager::import_spec(&file_path).unwrap();
        
        assert_eq!(spec.title, "Test API");
        assert_eq!(spec.endpoints.len(), 1);
        
        std::fs::remove_file(&file_path).ok();
    }
    
    #[test]
    fn test_validate_contract() {
        let spec = ApiSpec {
            version: OpenAPIVersion::V3,
            title: "Test API".to_string(),
            description: None,
            base_url: "https://api.test.com".to_string(),
            endpoints: vec![
                ApiEndpoint {
                    path: "/users".to_string(),
                    method: HttpMethod::GET,
                    operation_id: None,
                    summary: None,
                    description: None,
                    parameters: vec![],
                    request_body: None,
                    responses: HashMap::new(),
                    security: vec![],
                },
            ],
            security_schemes: HashMap::new(),
        };
        
        // Test valid endpoint
        let used = vec![("/users".to_string(), HttpMethod::GET)];
        let result = ApiManager::validate_contract(&spec, used);
        assert!(result.valid);
        assert_eq!(result.errors.len(), 0);
        
        // Test invalid endpoint
        let used = vec![("/invalid".to_string(), HttpMethod::POST)];
        let result = ApiManager::validate_contract(&spec, used);
        assert!(!result.valid);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].error_type, "ENDPOINT_NOT_FOUND");
    }
    
    #[test]
    fn test_parse_parameters() {
        let spec_json = serde_json::json!({
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {
                "/users/{id}": {
                    "get": {
                        "parameters": [
                            {
                                "name": "id",
                                "in": "path",
                                "required": true,
                                "schema": {"type": "integer"}
                            },
                            {
                                "name": "Authorization",
                                "in": "header",
                                "required": true,
                                "schema": {"type": "string"}
                            }
                        ],
                        "responses": {
                            "200": {"description": "OK"}
                        }
                    }
                }
            }
        });
        
        let spec = ApiManager::parse_spec(spec_json).unwrap();
        let endpoint = &spec.endpoints[0];
        
        assert_eq!(endpoint.parameters.len(), 2);
        
        let path_param = &endpoint.parameters[0];
        assert_eq!(path_param.name, "id");
        assert_eq!(path_param.location, ParameterLocation::Path);
        assert!(path_param.required);
        assert_eq!(path_param.schema_type, "integer");
        
        let header_param = &endpoint.parameters[1];
        assert_eq!(header_param.name, "Authorization");
        assert_eq!(header_param.location, ParameterLocation::Header);
        assert!(header_param.required);
    }
}
