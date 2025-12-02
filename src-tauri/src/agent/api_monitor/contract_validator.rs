// Contract Validator - Validates API requests/responses against OpenAPI schemas
// Detects breaking changes between API versions

use super::{ApiContract, Endpoint, ValidationResult, ValidationError, BreakingChange, BreakingChangeType, Severity};
use std::collections::HashMap;

pub struct ContractValidator;

impl ContractValidator {
    pub fn new() -> Self {
        Self
    }

    /// Validate request parameters and body against contract
    pub fn validate_request(
        &self,
        contract: &ApiContract,
        endpoint: &str,
        method: &str,
        params: Option<&serde_json::Value>,
        body: Option<&serde_json::Value>,
    ) -> Result<ValidationResult, String> {
        let endpoint_key = format!("{}:{}", method.to_uppercase(), endpoint);
        
        let endpoint_def = contract.endpoints.get(&endpoint_key)
            .ok_or_else(|| format!("Endpoint not found: {} {}", method, endpoint))?;

        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate required parameters
        if let Some(params_value) = params {
            if let Some(params_obj) = params_value.as_object() {
                for required_param in &endpoint_def.required_params {
                    if !params_obj.contains_key(required_param) {
                        errors.push(ValidationError {
                            field: required_param.clone(),
                            expected: "present".to_string(),
                            actual: "missing".to_string(),
                            message: format!("Required parameter '{}' is missing", required_param),
                        });
                    }
                }

                // Check for unexpected parameters
                for param_key in params_obj.keys() {
                    if !endpoint_def.required_params.contains(param_key) 
                        && !endpoint_def.optional_params.contains(param_key) {
                        warnings.push(format!("Unexpected parameter: {}", param_key));
                    }
                }
            }
        } else if !endpoint_def.required_params.is_empty() {
            for required_param in &endpoint_def.required_params {
                errors.push(ValidationError {
                    field: required_param.clone(),
                    expected: "present".to_string(),
                    actual: "missing".to_string(),
                    message: format!("Required parameter '{}' is missing", required_param),
                });
            }
        }

        // Validate request body schema if present
        if let Some(body_value) = body {
            if let Some(request_schema) = &endpoint_def.request_schema {
                self.validate_against_schema(body_value, request_schema, "request body", &mut errors);
            }
        }

        Ok(ValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings,
        })
    }

    /// Validate response against contract
    pub fn validate_response(
        &self,
        contract: &ApiContract,
        endpoint: &str,
        method: &str,
        status_code: u16,
        body: &serde_json::Value,
    ) -> Result<ValidationResult, String> {
        let endpoint_key = format!("{}:{}", method.to_uppercase(), endpoint);
        
        let endpoint_def = contract.endpoints.get(&endpoint_key)
            .ok_or_else(|| format!("Endpoint not found: {} {}", method, endpoint))?;

        let mut errors = Vec::new();
        let warnings = Vec::new();

        // Validate response body schema if present
        if let Some(response_schema) = &endpoint_def.response_schema {
            self.validate_against_schema(body, response_schema, "response body", &mut errors);
        }

        Ok(ValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings,
        })
    }

    /// Validate JSON value against JSON schema
    fn validate_against_schema(
        &self,
        value: &serde_json::Value,
        schema: &serde_json::Value,
        context: &str,
        errors: &mut Vec<ValidationError>,
    ) {
        // Basic schema validation (can be extended with jsonschema crate)
        if let Some(schema_obj) = schema.as_object() {
            if let Some(required) = schema_obj.get("required") {
                if let Some(required_array) = required.as_array() {
                    if let Some(value_obj) = value.as_object() {
                        for req in required_array {
                            if let Some(req_str) = req.as_str() {
                                if !value_obj.contains_key(req_str) {
                                    errors.push(ValidationError {
                                        field: format!("{}.{}", context, req_str),
                                        expected: "present".to_string(),
                                        actual: "missing".to_string(),
                                        message: format!("Required field '{}' is missing in {}", req_str, context),
                                    });
                                }
                            }
                        }
                    }
                }
            }

            // Validate property types
            if let Some(properties) = schema_obj.get("properties") {
                if let Some(props_obj) = properties.as_object() {
                    if let Some(value_obj) = value.as_object() {
                        for (prop_name, prop_schema) in props_obj {
                            if let Some(prop_value) = value_obj.get(prop_name) {
                                if let Some(type_def) = prop_schema.get("type") {
                                    if let Some(expected_type) = type_def.as_str() {
                                        let actual_type = match prop_value {
                                            serde_json::Value::Null => "null",
                                            serde_json::Value::Bool(_) => "boolean",
                                            serde_json::Value::Number(_) => "number",
                                            serde_json::Value::String(_) => "string",
                                            serde_json::Value::Array(_) => "array",
                                            serde_json::Value::Object(_) => "object",
                                        };

                                        if expected_type != actual_type && expected_type != "integer" {
                                            errors.push(ValidationError {
                                                field: format!("{}.{}", context, prop_name),
                                                expected: expected_type.to_string(),
                                                actual: actual_type.to_string(),
                                                message: format!(
                                                    "Type mismatch for field '{}': expected {}, got {}",
                                                    prop_name, expected_type, actual_type
                                                ),
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Detect breaking changes between two API contracts
    pub fn detect_breaking_changes(
        &self,
        old_contract: &ApiContract,
        new_contract: &ApiContract,
    ) -> Result<Vec<BreakingChange>, String> {
        let mut breaking_changes = Vec::new();

        // Check for removed endpoints
        for (old_endpoint_key, old_endpoint) in &old_contract.endpoints {
            if !new_contract.endpoints.contains_key(old_endpoint_key) {
                breaking_changes.push(BreakingChange {
                    change_type: BreakingChangeType::EndpointRemoved,
                    endpoint: old_endpoint.path.clone(),
                    description: format!("Endpoint {} {} was removed", old_endpoint.method, old_endpoint.path),
                    severity: Severity::Critical,
                });
            }
        }

        // Check for changes in existing endpoints
        for (endpoint_key, old_endpoint) in &old_contract.endpoints {
            if let Some(new_endpoint) = new_contract.endpoints.get(endpoint_key) {
                // Check for added required parameters (breaking)
                for new_required in &new_endpoint.required_params {
                    if !old_endpoint.required_params.contains(new_required) {
                        breaking_changes.push(BreakingChange {
                            change_type: BreakingChangeType::RequiredParamAdded,
                            endpoint: old_endpoint.path.clone(),
                            description: format!(
                                "New required parameter '{}' added to {} {}",
                                new_required, old_endpoint.method, old_endpoint.path
                            ),
                            severity: Severity::High,
                        });
                    }
                }

                // Check for removed optional parameters (could be breaking)
                for old_optional in &old_endpoint.optional_params {
                    if !new_endpoint.optional_params.contains(old_optional) 
                        && !new_endpoint.required_params.contains(old_optional) {
                        breaking_changes.push(BreakingChange {
                            change_type: BreakingChangeType::ParamTypeChanged,
                            endpoint: old_endpoint.path.clone(),
                            description: format!(
                                "Optional parameter '{}' removed from {} {}",
                                old_optional, old_endpoint.method, old_endpoint.path
                            ),
                            severity: Severity::Medium,
                        });
                    }
                }

                // Check for response schema changes (simplified check)
                if old_endpoint.response_schema != new_endpoint.response_schema {
                    breaking_changes.push(BreakingChange {
                        change_type: BreakingChangeType::ResponseSchemaChanged,
                        endpoint: old_endpoint.path.clone(),
                        description: format!(
                            "Response schema changed for {} {}",
                            old_endpoint.method, old_endpoint.path
                        ),
                        severity: Severity::High,
                    });
                }
            }
        }

        Ok(breaking_changes)
    }
}

impl Default for ContractValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_validate_request_missing_required_param() {
        let validator = ContractValidator::new();
        
        let mut endpoints = HashMap::new();
        endpoints.insert(
            "GET:/users".to_string(),
            Endpoint {
                path: "/users".to_string(),
                method: "GET".to_string(),
                request_schema: None,
                response_schema: None,
                required_params: vec!["id".to_string()],
                optional_params: vec![],
            },
        );

        let contract = ApiContract {
            name: "test".to_string(),
            base_url: "https://api.example.com".to_string(),
            version: "1.0.0".to_string(),
            spec_format: super::super::SpecFormat::OpenApiV3,
            spec_content: String::new(),
            endpoints,
            rate_limit: None,
            last_validated: chrono::Utc::now(),
            health_check_url: None,
        };

        let result = validator.validate_request(
            &contract,
            "/users",
            "GET",
            Some(&json!({})),
            None,
        ).unwrap();

        assert!(!result.valid);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].field, "id");
    }

    #[test]
    fn test_detect_breaking_changes_removed_endpoint() {
        let validator = ContractValidator::new();
        
        let mut old_endpoints = HashMap::new();
        old_endpoints.insert(
            "GET:/users".to_string(),
            Endpoint {
                path: "/users".to_string(),
                method: "GET".to_string(),
                request_schema: None,
                response_schema: None,
                required_params: vec![],
                optional_params: vec![],
            },
        );

        let old_contract = ApiContract {
            name: "test".to_string(),
            base_url: "https://api.example.com".to_string(),
            version: "1.0.0".to_string(),
            spec_format: super::super::SpecFormat::OpenApiV3,
            spec_content: String::new(),
            endpoints: old_endpoints,
            rate_limit: None,
            last_validated: chrono::Utc::now(),
            health_check_url: None,
        };

        let new_contract = ApiContract {
            name: "test".to_string(),
            base_url: "https://api.example.com".to_string(),
            version: "2.0.0".to_string(),
            spec_format: super::super::SpecFormat::OpenApiV3,
            spec_content: String::new(),
            endpoints: HashMap::new(),
            rate_limit: None,
            last_validated: chrono::Utc::now(),
            health_check_url: None,
        };

        let changes = validator.detect_breaking_changes(&old_contract, &new_contract).unwrap();
        assert_eq!(changes.len(), 1);
        assert!(matches!(changes[0].change_type, BreakingChangeType::EndpointRemoved));
        assert!(matches!(changes[0].severity, Severity::Critical));
    }
}
