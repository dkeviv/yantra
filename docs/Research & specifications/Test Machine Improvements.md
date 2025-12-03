# Test Intelligence Layer: Complete State Machine Specification

**Version:** 1.0
**Date:** December 2024
**Purpose:** Address critical gaps in autonomous test generation and validation
**Integration:** Extends existing Testing State Machine (Section 3927-3986 of main spec)

---

## Overview

The Test Intelligence Layer (TIL) addresses the fundamental challenge of autonomous testing: **knowing what "correct" means without human verification** . It introduces structured approaches for:

1. **Test Oracle Generation** — Deriving expected behavior from intent
2. **Test Data Synthesis** — Generating meaningful inputs automatically
3. **Test Quality Assurance** — Verifying tests are effective
4. **Execution Intelligence** — Rich feedback for debugging
5. **Test Evolution** — Keeping tests synchronized with code

---

## Architecture Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           YANTRA PLATFORM                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    CODE GENERATION STATE MACHINE                        │ │
│  │  ArchGen → Plan → Execute → TestabilityCheck (NEW) → Complete          │ │
│  └──────────────────────────────────┬─────────────────────────────────────┘ │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                 TEST INTELLIGENCE STATE MACHINE (NEW)                   │ │
│  │                                                                          │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │ │
│  │  │   ORACLE    │───▶│  TEST DATA  │───▶│    TEST     │                  │ │
│  │  │  SYNTHESIS  │    │  GENERATION │    │  GENERATION │                  │ │
│  │  └─────────────┘    └─────────────┘    └─────────────┘                  │ │
│  │         │                  │                  │                          │ │
│  │         ▼                  ▼                  ▼                          │ │
│  │  ┌─────────────────────────────────────────────────────┐                │ │
│  │  │              TEST QUALITY GATE (NEW)                 │                │ │
│  │  │  MutationTesting → AssertionAnalysis → FlakeCheck   │                │ │
│  │  └─────────────────────────────────────────────────────┘                │ │
│  └──────────────────────────────────┬─────────────────────────────────────┘ │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    TEST EXECUTION STATE MACHINE                         │ │
│  │  (Enhanced with ExecutionTracing, SemanticVerification)                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## State Machine 1: Test Intelligence State Machine (NEW)

**Responsibility:** Transform user intent into verified, high-quality test suites

**Total States:** 12 states across 4 phases

### Phase 1: Oracle Synthesis (3 states)

#### State 1.1: IntentExtraction

**Purpose:** Parse user intent into formal behavioral expectations

**Entry Condition:** CodeGen machine completes successfully
**Exit Condition:** Structured expectations extracted

```rust
// src-tauri/src/testing/intelligence/intent_extraction.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentExtraction {
    pub state: IntentExtractionState,
    pub user_intent: String,
    pub generated_code: GeneratedCode,
    pub context: GNNContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntentExtractionState {
    Pending,
    ExtractingBehaviors,
    ExtractingConstraints,
    ExtractingEdgeCases,
    Complete,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralExpectation {
    pub id: String,
    pub description: String,
    pub expectation_type: ExpectationType,
    pub preconditions: Vec<Condition>,
    pub postconditions: Vec<Condition>,
    pub invariants: Vec<Invariant>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpectationType {
    // Functional expectations
    InputOutput {
        input_pattern: InputPattern,
        expected_output: OutputPattern,
    },
    StateTransition {
        initial_state: StatePattern,
        action: String,
        final_state: StatePattern,
    },
    // Non-functional expectations
    Performance {
        metric: PerformanceMetric,
        threshold: Threshold,
    },
    ErrorHandling {
        error_condition: String,
        expected_behavior: ErrorBehavior,
    },
    // Relational expectations
    Idempotent,
    Commutative,
    Associative,
    Monotonic { direction: MonotonicDirection },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    pub expression: String,           // "len(input) > 0"
    pub natural_language: String,     // "input must not be empty"
    pub checkable: bool,              // Can we verify this programmatically?
    pub check_code: Option<String>,   // Generated assertion code
}

impl IntentExtraction {
    pub async fn execute(&mut self, llm: &LLMOrchestrator) -> Result<Vec<BehavioralExpectation>> {
        self.state = IntentExtractionState::ExtractingBehaviors;

        // Step 1: Extract high-level behaviors from intent
        let behavior_prompt = self.build_behavior_extraction_prompt();
        let behaviors = llm.generate_structured::<Vec<BehaviorDescription>>(
            &behavior_prompt,
            "Extract distinct behavioral expectations from user intent"
        ).await?;

        self.state = IntentExtractionState::ExtractingConstraints;

        // Step 2: For each behavior, extract formal constraints
        let mut expectations = Vec::new();
        for behavior in behaviors {
            let constraint_prompt = self.build_constraint_prompt(&behavior);
            let expectation = llm.generate_structured::<BehavioralExpectation>(
                &constraint_prompt,
                "Convert behavior to formal preconditions/postconditions"
            ).await?;

            // Validate expectation is checkable
            let validated = self.validate_expectation(expectation)?;
            expectations.push(validated);
        }

        self.state = IntentExtractionState::ExtractingEdgeCases;

        // Step 3: Extract implicit edge cases from intent
        let edge_case_prompt = self.build_edge_case_prompt(&expectations);
        let edge_cases = llm.generate_structured::<Vec<BehavioralExpectation>>(
            &edge_case_prompt,
            "Identify edge cases and error conditions from intent"
        ).await?;

        expectations.extend(edge_cases);

        self.state = IntentExtractionState::Complete;
        Ok(expectations)
    }

    fn build_behavior_extraction_prompt(&self) -> String {
        format!(r#"
You are extracting behavioral expectations from a user's intent.

USER INTENT:
{}

GENERATED CODE:
```

{}

```

CODE CONTEXT (from dependency graph):
{}

Extract ALL distinct behaviors the code should exhibit. For each behavior:
1. What is the expected input-output relationship?
2. What state changes should occur?
3. What errors should be handled?
4. What properties should always hold (invariants)?

Output as JSON array of behaviors with:
- id: unique identifier
- description: natural language description
- category: "functional" | "error_handling" | "state_mutation" | "invariant"
- inputs: description of valid inputs
- outputs: description of expected outputs
- side_effects: any state changes

Be exhaustive. Include implicit expectations the user would assume.
"#, self.user_intent, self.generated_code.code, self.context.summary())
    }

    fn validate_expectation(&self, expectation: BehavioralExpectation) -> Result<BehavioralExpectation> {
        // Ensure preconditions and postconditions are programmatically checkable
        let mut validated = expectation.clone();

        for condition in &mut validated.preconditions {
            if condition.check_code.is_none() {
                // Generate assertion code for condition
                condition.check_code = Some(self.generate_assertion_code(&condition.expression)?);
            }
            condition.checkable = self.is_checkable(&condition)?;
        }

        for condition in &mut validated.postconditions {
            if condition.check_code.is_none() {
                condition.check_code = Some(self.generate_assertion_code(&condition.expression)?);
            }
            condition.checkable = self.is_checkable(&condition)?;
        }

        Ok(validated)
    }
}
```

**LLM Prompt Strategy:**

```rust
// Structured prompt for intent extraction
pub const INTENT_EXTRACTION_SYSTEM_PROMPT: &str = r#"
You are a test oracle generator. Your job is to extract VERIFIABLE behavioral
expectations from user intent.

CRITICAL RULES:
1. Every expectation must be PROGRAMMATICALLY VERIFIABLE
2. Extract IMPLICIT expectations (what user assumes but didn't say)
3. Include error conditions and edge cases
4. Be specific about input/output relationships
5. Identify invariants that should ALWAYS hold

For a function like "validate email", implicit expectations include:
- Empty string returns false (or raises error)
- None/null input is handled
- Very long strings don't crash
- Unicode is handled appropriately
- Result is deterministic (same input = same output)

DO NOT extract expectations that cannot be automatically checked.
"#;
```

**Performance Targets:**

- Intent extraction: <2s (single LLM call with structured output)
- Constraint formalization: <500ms per behavior
- Edge case extraction: <1s

---

#### State 1.2: ExpectationValidation

**Purpose:** Verify extracted expectations are consistent and complete

**Entry Condition:** IntentExtraction completes
**Exit Condition:** Validated, non-contradictory expectations

```rust
// src-tauri/src/testing/intelligence/expectation_validation.rs

#[derive(Debug, Clone)]
pub struct ExpectationValidator {
    expectations: Vec<BehavioralExpectation>,
    code_ast: AST,
    gnn: Arc<GNNEngine>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub contradictions: Vec<Contradiction>,
    pub coverage_gaps: Vec<CoverageGap>,
    pub ambiguities: Vec<Ambiguity>,
    pub refined_expectations: Vec<BehavioralExpectation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contradiction {
    pub expectation_a: String,
    pub expectation_b: String,
    pub reason: String,
    pub resolution: Option<Resolution>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageGap {
    pub code_element: CodeElement,  // Function, branch, etc.
    pub missing_expectation: String,
    pub suggested_expectation: Option<BehavioralExpectation>,
}

impl ExpectationValidator {
    pub async fn validate(&self) -> Result<ValidationResult> {
        let mut result = ValidationResult::default();

        // 1. Check for logical contradictions between expectations
        result.contradictions = self.find_contradictions().await?;

        // 2. Check code coverage - are there code paths without expectations?
        result.coverage_gaps = self.find_coverage_gaps().await?;

        // 3. Check for ambiguous expectations
        result.ambiguities = self.find_ambiguities().await?;

        // 4. Refine expectations based on findings
        if !result.contradictions.is_empty() || !result.coverage_gaps.is_empty() {
            result.refined_expectations = self.refine_expectations(&result).await?;
        } else {
            result.refined_expectations = self.expectations.clone();
        }

        result.is_valid = result.contradictions.is_empty()
            && result.ambiguities.iter().all(|a| a.resolved);

        Ok(result)
    }

    async fn find_coverage_gaps(&self) -> Result<Vec<CoverageGap>> {
        let mut gaps = Vec::new();

        // Get all code paths from AST
        let code_paths = self.extract_code_paths(&self.code_ast)?;

        // For each code path, check if there's a matching expectation
        for path in code_paths {
            let has_expectation = self.expectations.iter().any(|exp| {
                self.expectation_covers_path(exp, &path)
            });

            if !has_expectation {
                gaps.push(CoverageGap {
                    code_element: path.element.clone(),
                    missing_expectation: format!(
                        "No expectation covers code path: {}",
                        path.description
                    ),
                    suggested_expectation: self.suggest_expectation_for_path(&path).await?,
                });
            }
        }

        Ok(gaps)
    }

    fn extract_code_paths(&self, ast: &AST) -> Result<Vec<CodePath>> {
        let mut paths = Vec::new();

        // Extract branches (if/else, match, try/catch)
        for branch in ast.find_all_branches() {
            paths.push(CodePath {
                element: CodeElement::Branch(branch.clone()),
                description: format!("Branch at line {}: {}", branch.line, branch.condition),
                reachability: self.analyze_reachability(&branch)?,
            });
        }

        // Extract function entry/exit points
        for func in ast.find_all_functions() {
            paths.push(CodePath {
                element: CodeElement::FunctionEntry(func.clone()),
                description: format!("Function {} entry", func.name),
                reachability: Reachability::Always,
            });

            for return_point in func.return_points() {
                paths.push(CodePath {
                    element: CodeElement::FunctionExit(return_point.clone()),
                    description: format!("Function {} exit at line {}", func.name, return_point.line),
                    reachability: self.analyze_reachability(&return_point)?,
                });
            }
        }

        // Extract error handling paths
        for error_handler in ast.find_all_error_handlers() {
            paths.push(CodePath {
                element: CodeElement::ErrorHandler(error_handler.clone()),
                description: format!("Error handler at line {}", error_handler.line),
                reachability: Reachability::OnError,
            });
        }

        Ok(paths)
    }
}
```

---

#### State 1.3: OracleGeneration

**Purpose:** Generate executable oracle functions from validated expectations

**Entry Condition:** ExpectationValidation passes
**Exit Condition:** Executable oracle code generated

```rust
// src-tauri/src/testing/intelligence/oracle_generation.rs

#[derive(Debug, Clone)]
pub struct OracleGenerator {
    expectations: Vec<BehavioralExpectation>,
    language: ProgrammingLanguage,
    test_framework: TestFramework,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedOracle {
    pub expectation_id: String,
    pub oracle_code: String,
    pub oracle_type: OracleType,
    pub dependencies: Vec<String>,
    pub setup_code: Option<String>,
    pub teardown_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OracleType {
    // Direct comparison oracle
    ExactMatch {
        expected_value: String,
    },
    // Pattern-based oracle
    PatternMatch {
        pattern: String,
        pattern_type: PatternType, // Regex, JSON schema, etc.
    },
    // Property-based oracle
    PropertyCheck {
        properties: Vec<Property>,
    },
    // Differential oracle (compare with reference implementation)
    Differential {
        reference_impl: String,
    },
    // Metamorphic oracle (check relationships between outputs)
    Metamorphic {
        relation: MetamorphicRelation,
    },
    // Statistical oracle (for non-deterministic code)
    Statistical {
        metric: StatisticalMetric,
        threshold: f64,
        sample_size: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Property {
    pub name: String,
    pub check_code: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetamorphicRelation {
    // f(x) == f(f(x)) for idempotent functions
    Idempotent,
    // f(x, y) == f(y, x) for commutative functions
    Commutative,
    // f(a, f(b, c)) == f(f(a, b), c) for associative functions
    Associative,
    // f(x) <= f(y) when x <= y for monotonic functions
    Monotonic { direction: MonotonicDirection },
    // Custom relation
    Custom { relation_code: String },
}

impl OracleGenerator {
    pub async fn generate(&self, llm: &LLMOrchestrator) -> Result<Vec<GeneratedOracle>> {
        let mut oracles = Vec::new();

        for expectation in &self.expectations {
            let oracle = match &expectation.expectation_type {
                ExpectationType::InputOutput { input_pattern, expected_output } => {
                    self.generate_io_oracle(expectation, input_pattern, expected_output).await?
                }
                ExpectationType::StateTransition { initial_state, action, final_state } => {
                    self.generate_state_oracle(expectation, initial_state, action, final_state).await?
                }
                ExpectationType::ErrorHandling { error_condition, expected_behavior } => {
                    self.generate_error_oracle(expectation, error_condition, expected_behavior).await?
                }
                ExpectationType::Idempotent => {
                    self.generate_metamorphic_oracle(expectation, MetamorphicRelation::Idempotent).await?
                }
                ExpectationType::Monotonic { direction } => {
                    self.generate_metamorphic_oracle(
                        expectation,
                        MetamorphicRelation::Monotonic { direction: direction.clone() }
                    ).await?
                }
                _ => {
                    self.generate_property_oracle(expectation, llm).await?
                }
            };

            oracles.push(oracle);
        }

        Ok(oracles)
    }

    async fn generate_io_oracle(
        &self,
        expectation: &BehavioralExpectation,
        input_pattern: &InputPattern,
        expected_output: &OutputPattern,
    ) -> Result<GeneratedOracle> {
        let oracle_code = match (&self.language, &self.test_framework) {
            (ProgrammingLanguage::Python, TestFramework::Pytest) => {
                self.generate_python_io_oracle(expectation, input_pattern, expected_output)?
            }
            (ProgrammingLanguage::JavaScript, TestFramework::Jest) => {
                self.generate_js_io_oracle(expectation, input_pattern, expected_output)?
            }
            (ProgrammingLanguage::Rust, TestFramework::RustTest) => {
                self.generate_rust_io_oracle(expectation, input_pattern, expected_output)?
            }
            _ => return Err(anyhow!("Unsupported language/framework combination")),
        };

        Ok(GeneratedOracle {
            expectation_id: expectation.id.clone(),
            oracle_code,
            oracle_type: OracleType::ExactMatch {
                expected_value: expected_output.to_string(),
            },
            dependencies: self.extract_dependencies(&oracle_code)?,
            setup_code: self.generate_setup_code(expectation)?,
            teardown_code: self.generate_teardown_code(expectation)?,
        })
    }

    fn generate_python_io_oracle(
        &self,
        expectation: &BehavioralExpectation,
        input_pattern: &InputPattern,
        expected_output: &OutputPattern,
    ) -> Result<String> {
        let precondition_checks = expectation.preconditions.iter()
            .filter_map(|c| c.check_code.as_ref())
            .map(|code| format!("    assert {}, \"Precondition failed: {}\"", code, code))
            .collect::<Vec<_>>()
            .join("\n");

        let postcondition_checks = expectation.postconditions.iter()
            .filter_map(|c| c.check_code.as_ref())
            .map(|code| format!("    assert {}, \"Postcondition failed: {}\"", code, code))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(format!(r#"
def oracle_{expectation_id}(func, input_value):
    """
    Oracle for: {description}
    """
    # Precondition checks
{precondition_checks}

    # Execute function under test
    result = func(input_value)

    # Postcondition checks
{postcondition_checks}

    # Output validation
    {output_validation}

    return True  # All checks passed
"#,
            expectation_id = expectation.id,
            description = expectation.description,
            precondition_checks = precondition_checks,
            postcondition_checks = postcondition_checks,
            output_validation = self.generate_output_validation(expected_output)?,
        ))
    }

    async fn generate_metamorphic_oracle(
        &self,
        expectation: &BehavioralExpectation,
        relation: MetamorphicRelation,
    ) -> Result<GeneratedOracle> {
        let oracle_code = match relation {
            MetamorphicRelation::Idempotent => {
                self.generate_idempotent_oracle(expectation)?
            }
            MetamorphicRelation::Commutative => {
                self.generate_commutative_oracle(expectation)?
            }
            MetamorphicRelation::Monotonic { ref direction } => {
                self.generate_monotonic_oracle(expectation, direction)?
            }
            MetamorphicRelation::Custom { ref relation_code } => {
                relation_code.clone()
            }
            _ => return Err(anyhow!("Unsupported metamorphic relation")),
        };

        Ok(GeneratedOracle {
            expectation_id: expectation.id.clone(),
            oracle_code,
            oracle_type: OracleType::Metamorphic { relation },
            dependencies: vec![],
            setup_code: None,
            teardown_code: None,
        })
    }

    fn generate_idempotent_oracle(&self, expectation: &BehavioralExpectation) -> Result<String> {
        Ok(format!(r#"
def oracle_idempotent_{expectation_id}(func, input_value):
    """
    Metamorphic oracle: {description}
    Property: f(x) == f(f(x)) (idempotent)
    """
    result_once = func(input_value)
    result_twice = func(result_once)

    assert result_once == result_twice, (
        f"Idempotency violated: f(x)={{result_once}}, f(f(x))={{result_twice}}"
    )
    return True
"#,
            expectation_id = expectation.id,
            description = expectation.description,
        ))
    }
}
```

---

### Phase 2: Test Data Generation (3 states)

#### State 2.1: InputDomainAnalysis

**Purpose:** Analyze function signatures and constraints to understand valid input space

```rust
// src-tauri/src/testing/intelligence/input_domain.rs

#[derive(Debug, Clone)]
pub struct InputDomainAnalyzer {
    function_signature: FunctionSignature,
    type_info: TypeInfo,
    constraints: Vec<Condition>,
    gnn: Arc<GNNEngine>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputDomain {
    pub parameters: Vec<ParameterDomain>,
    pub constraints: Vec<DomainConstraint>,
    pub equivalence_classes: Vec<EquivalenceClass>,
    pub boundary_values: Vec<BoundaryValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDomain {
    pub name: String,
    pub param_type: ParameterType,
    pub valid_range: Option<Range>,
    pub special_values: Vec<SpecialValue>,
    pub nullable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    Integer { min: Option<i64>, max: Option<i64>, constraints: Vec<String> },
    Float { min: Option<f64>, max: Option<f64>, precision: Option<u32> },
    String { min_len: Option<usize>, max_len: Option<usize>, pattern: Option<String> },
    Boolean,
    Array { element_type: Box<ParameterType>, min_len: Option<usize>, max_len: Option<usize> },
    Object { fields: HashMap<String, ParameterType>, required: Vec<String> },
    Enum { variants: Vec<String> },
    Custom { type_name: String, constraints: Vec<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquivalenceClass {
    pub name: String,
    pub description: String,
    pub representative_values: Vec<TestValue>,
    pub constraints: Vec<String>,
    pub is_valid: bool,  // Valid or invalid equivalence class
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryValue {
    pub parameter: String,
    pub value: TestValue,
    pub boundary_type: BoundaryType,
    pub expected_behavior: ExpectedBehavior,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryType {
    Minimum,
    Maximum,
    JustBelowMinimum,
    JustAboveMaximum,
    Zero,
    NegativeOne,
    Empty,
    SingleElement,
    MaxCapacity,
}

impl InputDomainAnalyzer {
    pub async fn analyze(&self) -> Result<InputDomain> {
        let mut domain = InputDomain::default();

        // Step 1: Analyze each parameter's type and constraints
        for param in &self.function_signature.parameters {
            let param_domain = self.analyze_parameter(param).await?;
            domain.parameters.push(param_domain);
        }

        // Step 2: Extract equivalence classes
        domain.equivalence_classes = self.extract_equivalence_classes().await?;

        // Step 3: Identify boundary values
        domain.boundary_values = self.identify_boundaries(&domain.parameters).await?;

        // Step 4: Extract cross-parameter constraints
        domain.constraints = self.extract_cross_constraints().await?;

        Ok(domain)
    }

    async fn analyze_parameter(&self, param: &Parameter) -> Result<ParameterDomain> {
        let param_type = self.infer_parameter_type(param)?;

        // Extract constraints from docstrings, type hints, and code analysis
        let constraints = self.extract_parameter_constraints(param).await?;

        // Identify special values that should be tested
        let special_values = self.identify_special_values(&param_type, &constraints)?;

        Ok(ParameterDomain {
            name: param.name.clone(),
            param_type,
            valid_range: self.extract_valid_range(&constraints)?,
            special_values,
            nullable: self.is_nullable(param)?,
        })
    }

    fn identify_special_values(
        &self,
        param_type: &ParameterType,
        constraints: &[String]
    ) -> Result<Vec<SpecialValue>> {
        let mut special = Vec::new();

        match param_type {
            ParameterType::Integer { min, max, .. } => {
                special.push(SpecialValue::new("zero", TestValue::Integer(0)));
                special.push(SpecialValue::new("negative_one", TestValue::Integer(-1)));
                special.push(SpecialValue::new("one", TestValue::Integer(1)));
                if let Some(min_val) = min {
                    special.push(SpecialValue::new("min", TestValue::Integer(*min_val)));
                    special.push(SpecialValue::new("below_min", TestValue::Integer(min_val - 1)));
                }
                if let Some(max_val) = max {
                    special.push(SpecialValue::new("max", TestValue::Integer(*max_val)));
                    special.push(SpecialValue::new("above_max", TestValue::Integer(max_val + 1)));
                }
                special.push(SpecialValue::new("max_int", TestValue::Integer(i64::MAX)));
                special.push(SpecialValue::new("min_int", TestValue::Integer(i64::MIN)));
            }
            ParameterType::String { min_len, max_len, pattern } => {
                special.push(SpecialValue::new("empty", TestValue::String("".to_string())));
                special.push(SpecialValue::new("whitespace", TestValue::String("   ".to_string())));
                special.push(SpecialValue::new("unicode", TestValue::String("こんにちは🎉".to_string())));
                special.push(SpecialValue::new("special_chars", TestValue::String("<script>alert('xss')</script>".to_string())));
                if let Some(max) = max_len {
                    special.push(SpecialValue::new("max_length", TestValue::String("x".repeat(*max))));
                    special.push(SpecialValue::new("over_max", TestValue::String("x".repeat(max + 1))));
                }
            }
            ParameterType::Array { element_type, min_len, max_len } => {
                special.push(SpecialValue::new("empty_array", TestValue::Array(vec![])));
                special.push(SpecialValue::new("single_element", TestValue::Array(vec![self.default_value(element_type)?])));
                if let Some(max) = max_len {
                    // Generate array at max capacity
                    let max_array: Vec<_> = (0..*max).map(|_| self.default_value(element_type).unwrap()).collect();
                    special.push(SpecialValue::new("max_capacity", TestValue::Array(max_array)));
                }
            }
            _ => {}
        }

        Ok(special)
    }

    async fn extract_equivalence_classes(&self) -> Result<Vec<EquivalenceClass>> {
        let mut classes = Vec::new();

        // Valid equivalence classes
        for param in &self.function_signature.parameters {
            match &self.infer_parameter_type(param)? {
                ParameterType::Integer { min, max, .. } => {
                    if let (Some(min_val), Some(max_val)) = (min, max) {
                        // Partition integer range
                        classes.push(EquivalenceClass {
                            name: format!("{}_negative", param.name),
                            description: "Negative integers".to_string(),
                            representative_values: vec![TestValue::Integer(-1), TestValue::Integer(-100)],
                            constraints: vec![format!("{} < 0", param.name)],
                            is_valid: *min_val < 0,
                        });
                        classes.push(EquivalenceClass {
                            name: format!("{}_zero", param.name),
                            description: "Zero".to_string(),
                            representative_values: vec![TestValue::Integer(0)],
                            constraints: vec![format!("{} == 0", param.name)],
                            is_valid: *min_val <= 0 && *max_val >= 0,
                        });
                        classes.push(EquivalenceClass {
                            name: format!("{}_positive", param.name),
                            description: "Positive integers".to_string(),
                            representative_values: vec![TestValue::Integer(1), TestValue::Integer(100)],
                            constraints: vec![format!("{} > 0", param.name)],
                            is_valid: *max_val > 0,
                        });
                    }
                }
                ParameterType::String { pattern, .. } => {
                    if let Some(pat) = pattern {
                        classes.push(EquivalenceClass {
                            name: format!("{}_matching", param.name),
                            description: format!("Strings matching pattern: {}", pat),
                            representative_values: self.generate_matching_strings(pat)?,
                            constraints: vec![format!("{} matches {}", param.name, pat)],
                            is_valid: true,
                        });
                        classes.push(EquivalenceClass {
                            name: format!("{}_not_matching", param.name),
                            description: format!("Strings NOT matching pattern: {}", pat),
                            representative_values: self.generate_non_matching_strings(pat)?,
                            constraints: vec![format!("{} does not match {}", param.name, pat)],
                            is_valid: false,
                        });
                    }
                }
                _ => {}
            }
        }

        // Invalid equivalence classes (for error path testing)
        for param in &self.function_signature.parameters {
            if self.is_nullable(param)? {
                classes.push(EquivalenceClass {
                    name: format!("{}_null", param.name),
                    description: "Null/None value".to_string(),
                    representative_values: vec![TestValue::Null],
                    constraints: vec![format!("{} is null", param.name)],
                    is_valid: false,
                });
            }
        }

        Ok(classes)
    }
}
```

---

#### State 2.2: TestDataSynthesis

**Purpose:** Generate concrete test data from domain analysis

```rust
// src-tauri/src/testing/intelligence/test_data_synthesis.rs

#[derive(Debug, Clone)]
pub struct TestDataSynthesizer {
    domain: InputDomain,
    oracles: Vec<GeneratedOracle>,
    strategies: Vec<DataGenerationStrategy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataGenerationStrategy {
    BoundaryValues,
    EquivalencePartitioning,
    RandomWithinConstraints,
    PropertyBased { num_samples: usize },
    HistoricalFailures,
    MutationBased { seed_data: Vec<TestValue> },
    ConstraintSolver,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestDataSet {
    pub test_cases: Vec<TestCase>,
    pub coverage_metrics: CoverageMetrics,
    pub generation_strategy: DataGenerationStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub id: String,
    pub inputs: HashMap<String, TestValue>,
    pub expected_output: Option<TestValue>,
    pub oracle_id: String,
    pub category: TestCaseCategory,
    pub priority: TestPriority,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestCaseCategory {
    HappyPath,
    BoundaryCondition,
    ErrorCondition,
    EdgeCase,
    Performance,
    Security,
    Regression,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestPriority {
    Critical,  // Must pass for any release
    High,      // Should pass, blocks most releases
    Medium,    // Important but can be deferred
    Low,       // Nice to have
}

impl TestDataSynthesizer {
    pub async fn synthesize(&self) -> Result<TestDataSet> {
        let mut all_cases = Vec::new();

        for strategy in &self.strategies {
            let cases = match strategy {
                DataGenerationStrategy::BoundaryValues => {
                    self.generate_boundary_cases().await?
                }
                DataGenerationStrategy::EquivalencePartitioning => {
                    self.generate_equivalence_cases().await?
                }
                DataGenerationStrategy::PropertyBased { num_samples } => {
                    self.generate_property_based_cases(*num_samples).await?
                }
                DataGenerationStrategy::RandomWithinConstraints => {
                    self.generate_random_constrained_cases(100).await?
                }
                DataGenerationStrategy::HistoricalFailures => {
                    self.generate_historical_failure_cases().await?
                }
                DataGenerationStrategy::ConstraintSolver => {
                    self.generate_constraint_solver_cases().await?
                }
                DataGenerationStrategy::MutationBased { seed_data } => {
                    self.generate_mutation_cases(seed_data).await?
                }
            };
            all_cases.extend(cases);
        }

        // Deduplicate and prioritize
        let deduplicated = self.deduplicate_cases(all_cases)?;
        let prioritized = self.prioritize_cases(deduplicated)?;

        Ok(TestDataSet {
            test_cases: prioritized,
            coverage_metrics: self.calculate_coverage_metrics()?,
            generation_strategy: DataGenerationStrategy::BoundaryValues, // Primary strategy
        })
    }

    async fn generate_boundary_cases(&self) -> Result<Vec<TestCase>> {
        let mut cases = Vec::new();

        for boundary in &self.domain.boundary_values {
            let mut inputs = HashMap::new();

            // Set boundary value for target parameter
            inputs.insert(boundary.parameter.clone(), boundary.value.clone());

            // Fill other parameters with valid defaults
            for param in &self.domain.parameters {
                if param.name != boundary.parameter {
                    inputs.insert(param.name.clone(), self.get_valid_default(&param)?);
                }
            }

            let oracle = self.find_matching_oracle(&boundary.expected_behavior)?;

            cases.push(TestCase {
                id: format!("boundary_{}_{}", boundary.parameter, boundary.boundary_type.name()),
                inputs,
                expected_output: self.compute_expected_output(&boundary)?,
                oracle_id: oracle.expectation_id.clone(),
                category: TestCaseCategory::BoundaryCondition,
                priority: TestPriority::High,
                description: format!(
                    "Boundary test: {} at {} ({})",
                    boundary.parameter,
                    boundary.value,
                    boundary.boundary_type.name()
                ),
            });
        }

        Ok(cases)
    }

    async fn generate_property_based_cases(&self, num_samples: usize) -> Result<Vec<TestCase>> {
        let mut cases = Vec::new();

        // Use Hypothesis-style shrinking arbitrary generation
        for i in 0..num_samples {
            let mut inputs = HashMap::new();

            for param in &self.domain.parameters {
                let value = self.generate_arbitrary_value(&param.param_type, &self.domain.constraints)?;
                inputs.insert(param.name.clone(), value);
            }

            // For property-based tests, oracle verifies properties, not specific output
            let oracle = self.find_property_oracle()?;

            cases.push(TestCase {
                id: format!("property_based_{}", i),
                inputs,
                expected_output: None, // Properties checked by oracle
                oracle_id: oracle.expectation_id.clone(),
                category: TestCaseCategory::HappyPath,
                priority: TestPriority::Medium,
                description: format!("Property-based test sample {}", i),
            });
        }

        Ok(cases)
    }

    fn generate_arbitrary_value(
        &self,
        param_type: &ParameterType,
        constraints: &[DomainConstraint]
    ) -> Result<TestValue> {
        match param_type {
            ParameterType::Integer { min, max, .. } => {
                let min_val = min.unwrap_or(i64::MIN / 2);
                let max_val = max.unwrap_or(i64::MAX / 2);

                // Bias toward interesting values
                let choices = vec![
                    (0.1, TestValue::Integer(0)),
                    (0.1, TestValue::Integer(1)),
                    (0.1, TestValue::Integer(-1)),
                    (0.1, TestValue::Integer(min_val)),
                    (0.1, TestValue::Integer(max_val)),
                    (0.5, TestValue::Integer(rand::thread_rng().gen_range(min_val..=max_val))),
                ];

                self.weighted_choice(&choices)
            }
            ParameterType::String { min_len, max_len, pattern } => {
                let len = rand::thread_rng().gen_range(
                    min_len.unwrap_or(0)..=max_len.unwrap_or(1000)
                );

                if let Some(pat) = pattern {
                    // Generate string matching pattern
                    self.generate_string_from_pattern(pat, len)
                } else {
                    Ok(TestValue::String(self.random_string(len)))
                }
            }
            ParameterType::Array { element_type, min_len, max_len } => {
                let len = rand::thread_rng().gen_range(
                    min_len.unwrap_or(0)..=max_len.unwrap_or(100)
                );

                let elements: Result<Vec<_>> = (0..len)
                    .map(|_| self.generate_arbitrary_value(element_type, constraints))
                    .collect();

                Ok(TestValue::Array(elements?))
            }
            ParameterType::Boolean => {
                Ok(TestValue::Boolean(rand::random()))
            }
            ParameterType::Object { fields, required } => {
                let mut obj = HashMap::new();
                for (name, field_type) in fields {
                    // Always include required fields, randomly include optional
                    if required.contains(name) || rand::random() {
                        obj.insert(name.clone(), self.generate_arbitrary_value(field_type, constraints)?);
                    }
                }
                Ok(TestValue::Object(obj))
            }
            ParameterType::Enum { variants } => {
                let idx = rand::thread_rng().gen_range(0..variants.len());
                Ok(TestValue::String(variants[idx].clone()))
            }
            _ => Err(anyhow!("Cannot generate arbitrary value for type")),
        }
    }

    async fn generate_historical_failure_cases(&self) -> Result<Vec<TestCase>> {
        // Query known issues database for similar functions
        let similar_failures = self.query_known_failures().await?;

        let mut cases = Vec::new();
        for failure in similar_failures {
            // Adapt failure input to current function signature
            if let Some(adapted_input) = self.adapt_failure_input(&failure)? {
                cases.push(TestCase {
                    id: format!("historical_{}", failure.id),
                    inputs: adapted_input,
                    expected_output: None, // Unknown - this used to fail
                    oracle_id: self.find_error_oracle()?.expectation_id.clone(),
                    category: TestCaseCategory::Regression,
                    priority: TestPriority::High,
                    description: format!("Historical failure: {}", failure.description),
                });
            }
        }

        Ok(cases)
    }
}
```

---

#### State 2.3: TestDataValidation

**Purpose:** Validate generated test data is meaningful and achievable

```rust
// src-tauri/src/testing/intelligence/test_data_validation.rs

#[derive(Debug, Clone)]
pub struct TestDataValidator {
    test_cases: Vec<TestCase>,
    domain: InputDomain,
    code_ast: AST,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestDataValidationResult {
    pub valid_cases: Vec<TestCase>,
    pub invalid_cases: Vec<InvalidTestCase>,
    pub redundant_cases: Vec<RedundantTestCase>,
    pub coverage_analysis: TestCoverageAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvalidTestCase {
    pub test_case: TestCase,
    pub reason: InvalidReason,
    pub suggested_fix: Option<TestCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvalidReason {
    ViolatesConstraint { constraint: String },
    UnreachablePath { path: String },
    ImpossibleInput { explanation: String },
    InconsistentExpectation { explanation: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundantTestCase {
    pub test_case: TestCase,
    pub equivalent_to: String,  // ID of equivalent test case
    pub redundancy_type: RedundancyType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RedundancyType {
    SameEquivalenceClass,
    SameBranchCoverage,
    SubsumedByOther,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCoverageAnalysis {
    pub branch_coverage: f32,
    pub path_coverage: f32,
    pub equivalence_class_coverage: f32,
    pub boundary_coverage: f32,
    pub uncovered_branches: Vec<String>,
    pub uncovered_classes: Vec<String>,
}

impl TestDataValidator {
    pub async fn validate(&self) -> Result<TestDataValidationResult> {
        let mut result = TestDataValidationResult::default();

        // Step 1: Validate each test case against constraints
        for test_case in &self.test_cases {
            match self.validate_test_case(test_case).await? {
                TestCaseValidity::Valid => {
                    result.valid_cases.push(test_case.clone());
                }
                TestCaseValidity::Invalid(reason) => {
                    result.invalid_cases.push(InvalidTestCase {
                        test_case: test_case.clone(),
                        reason,
                        suggested_fix: self.suggest_fix(test_case).await?,
                    });
                }
            }
        }

        // Step 2: Identify redundant test cases
        result.redundant_cases = self.find_redundant_cases(&result.valid_cases)?;

        // Remove redundant cases from valid set
        let redundant_ids: HashSet<_> = result.redundant_cases.iter()
            .map(|r| &r.test_case.id)
            .collect();
        result.valid_cases.retain(|tc| !redundant_ids.contains(&tc.id));

        // Step 3: Analyze coverage
        result.coverage_analysis = self.analyze_coverage(&result.valid_cases)?;

        Ok(result)
    }

    async fn validate_test_case(&self, test_case: &TestCase) -> Result<TestCaseValidity> {
        // Check against domain constraints
        for constraint in &self.domain.constraints {
            if !self.satisfies_constraint(test_case, constraint)? {
                return Ok(TestCaseValidity::Invalid(InvalidReason::ViolatesConstraint {
                    constraint: constraint.expression.clone(),
                }));
            }
        }

        // Check if input can actually reach the code path it's meant to test
        let reachable_paths = self.compute_reachable_paths(&test_case.inputs)?;
        if test_case.category == TestCaseCategory::ErrorCondition {
            // Error test should reach error handling path
            if !reachable_paths.iter().any(|p| p.is_error_path) {
                return Ok(TestCaseValidity::Invalid(InvalidReason::UnreachablePath {
                    path: "error handling".to_string(),
                }));
            }
        }

        Ok(TestCaseValidity::Valid)
    }

    fn find_redundant_cases(&self, cases: &[TestCase]) -> Result<Vec<RedundantTestCase>> {
        let mut redundant = Vec::new();

        // Compute coverage signature for each test case
        let mut coverage_signatures: HashMap<String, &TestCase> = HashMap::new();

        for case in cases {
            let signature = self.compute_coverage_signature(case)?;

            if let Some(existing) = coverage_signatures.get(&signature) {
                // Same coverage - one is redundant
                redundant.push(RedundantTestCase {
                    test_case: case.clone(),
                    equivalent_to: existing.id.clone(),
                    redundancy_type: RedundancyType::SameBranchCoverage,
                });
            } else {
                coverage_signatures.insert(signature, case);
            }
        }

        // Also check for equivalence class redundancy
        let mut class_representatives: HashMap<String, &TestCase> = HashMap::new();

        for case in cases {
            if let Some(eq_class) = self.get_equivalence_class(case)? {
                if let Some(existing) = class_representatives.get(&eq_class) {
                    if !redundant.iter().any(|r| r.test_case.id == case.id) {
                        redundant.push(RedundantTestCase {
                            test_case: case.clone(),
                            equivalent_to: existing.id.clone(),
                            redundancy_type: RedundancyType::SameEquivalenceClass,
                        });
                    }
                } else {
                    class_representatives.insert(eq_class, case);
                }
            }
        }

        Ok(redundant)
    }

    fn analyze_coverage(&self, cases: &[TestCase]) -> Result<TestCoverageAnalysis> {
        // Extract all branches from AST
        let all_branches = self.extract_branches(&self.code_ast)?;
        let covered_branches: HashSet<_> = cases.iter()
            .flat_map(|c| self.get_covered_branches(c).unwrap_or_default())
            .collect();

        let uncovered: Vec<_> = all_branches.iter()
            .filter(|b| !covered_branches.contains(*b))
            .cloned()
            .collect();

        // Check equivalence class coverage
        let all_classes: Vec<_> = self.domain.equivalence_classes.iter()
            .map(|c| c.name.clone())
            .collect();
        let covered_classes: HashSet<_> = cases.iter()
            .filter_map(|c| self.get_equivalence_class(c).ok().flatten())
            .collect();

        let uncovered_classes: Vec<_> = all_classes.iter()
            .filter(|c| !covered_classes.contains(*c))
            .cloned()
            .collect();

        Ok(TestCoverageAnalysis {
            branch_coverage: covered_branches.len() as f32 / all_branches.len() as f32,
            path_coverage: self.compute_path_coverage(cases)?,
            equivalence_class_coverage: covered_classes.len() as f32 / all_classes.len() as f32,
            boundary_coverage: self.compute_boundary_coverage(cases)?,
            uncovered_branches: uncovered,
            uncovered_classes,
        })
    }
}
```

---

### Phase 3: Test Quality Assurance (4 states)

#### State 3.1: TestGeneration (Enhanced)

**Purpose:** Generate test code from oracles and test data

```rust
// src-tauri/src/testing/intelligence/test_generation.rs

#[derive(Debug, Clone)]
pub struct EnhancedTestGenerator {
    oracles: Vec<GeneratedOracle>,
    test_data: TestDataSet,
    language: ProgrammingLanguage,
    framework: TestFramework,
    quality_requirements: QualityRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRequirements {
    pub min_assertion_count: usize,
    pub require_error_message: bool,
    pub require_setup_teardown: bool,
    pub require_timeout: bool,
    pub timeout_seconds: u64,
    pub require_isolation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedTest {
    pub id: String,
    pub name: String,
    pub code: String,
    pub test_case_id: String,
    pub oracle_id: String,
    pub assertions: Vec<Assertion>,
    pub setup_code: Option<String>,
    pub teardown_code: Option<String>,
    pub metadata: TestMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assertion {
    pub assertion_type: AssertionType,
    pub code: String,
    pub message: String,
    pub strength: AssertionStrength,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssertionType {
    Equality,
    Inequality,
    TypeCheck,
    RangeCheck,
    ContainsCheck,
    ThrowsException,
    DoesNotThrow,
    PropertyHolds,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssertionStrength {
    Strong,   // Specific value check
    Medium,   // Type or range check
    Weak,     // Just checks no exception
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMetadata {
    pub category: TestCaseCategory,
    pub priority: TestPriority,
    pub expected_duration_ms: u64,
    pub flakiness_risk: FlakeRisk,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlakeRisk {
    None,
    Low,      // Minor timing sensitivity
    Medium,   // Network/file system dependency
    High,     // Significant non-determinism
}

impl EnhancedTestGenerator {
    pub async fn generate(&self) -> Result<Vec<GeneratedTest>> {
        let mut tests = Vec::new();

        for test_case in &self.test_data.test_cases {
            let oracle = self.oracles.iter()
                .find(|o| o.expectation_id == test_case.oracle_id)
                .ok_or_else(|| anyhow!("Oracle not found: {}", test_case.oracle_id))?;

            let test = self.generate_test(test_case, oracle).await?;

            // Validate test meets quality requirements
            self.validate_test_quality(&test)?;

            tests.push(test);
        }

        Ok(tests)
    }

    async fn generate_test(
        &self,
        test_case: &TestCase,
        oracle: &GeneratedOracle,
    ) -> Result<GeneratedTest> {
        let assertions = self.generate_assertions(test_case, oracle)?;

        // Ensure minimum assertion count
        if assertions.len() < self.quality_requirements.min_assertion_count {
            let additional = self.generate_additional_assertions(test_case, oracle)?;
            assertions.extend(additional);
        }

        let test_code = self.generate_test_code(test_case, oracle, &assertions)?;

        Ok(GeneratedTest {
            id: format!("test_{}", test_case.id),
            name: self.generate_test_name(test_case),
            code: test_code,
            test_case_id: test_case.id.clone(),
            oracle_id: oracle.expectation_id.clone(),
            assertions,
            setup_code: oracle.setup_code.clone(),
            teardown_code: oracle.teardown_code.clone(),
            metadata: TestMetadata {
                category: test_case.category.clone(),
                priority: test_case.priority.clone(),
                expected_duration_ms: self.estimate_duration(test_case)?,
                flakiness_risk: self.assess_flake_risk(test_case)?,
                dependencies: self.extract_test_dependencies(test_case)?,
            },
        })
    }

    fn generate_assertions(
        &self,
        test_case: &TestCase,
        oracle: &GeneratedOracle,
    ) -> Result<Vec<Assertion>> {
        let mut assertions = Vec::new();

        // Primary output assertion
        if let Some(expected) = &test_case.expected_output {
            assertions.push(Assertion {
                assertion_type: AssertionType::Equality,
                code: self.generate_equality_assertion(expected)?,
                message: format!("Expected output: {:?}", expected),
                strength: AssertionStrength::Strong,
            });
        }

        // Type assertion
        assertions.push(Assertion {
            assertion_type: AssertionType::TypeCheck,
            code: self.generate_type_assertion(&oracle)?,
            message: "Result should be of expected type".to_string(),
            strength: AssertionStrength::Medium,
        });

        // Add assertions from oracle postconditions
        match &oracle.oracle_type {
            OracleType::PropertyCheck { properties } => {
                for property in properties {
                    assertions.push(Assertion {
                        assertion_type: AssertionType::PropertyHolds,
                        code: property.check_code.clone(),
                        message: property.description.clone(),
                        strength: AssertionStrength::Strong,
                    });
                }
            }
            OracleType::Metamorphic { relation } => {
                assertions.push(Assertion {
                    assertion_type: AssertionType::Custom,
                    code: self.generate_metamorphic_assertion(relation)?,
                    message: format!("Metamorphic relation: {:?}", relation),
                    strength: AssertionStrength::Strong,
                });
            }
            _ => {}
        }

        Ok(assertions)
    }

    fn generate_test_code(
        &self,
        test_case: &TestCase,
        oracle: &GeneratedOracle,
        assertions: &[Assertion],
    ) -> Result<String> {
        match (&self.language, &self.framework) {
            (ProgrammingLanguage::Python, TestFramework::Pytest) => {
                self.generate_pytest_code(test_case, oracle, assertions)
            }
            (ProgrammingLanguage::JavaScript, TestFramework::Jest) => {
                self.generate_jest_code(test_case, oracle, assertions)
            }
            (ProgrammingLanguage::Rust, TestFramework::RustTest) => {
                self.generate_rust_test_code(test_case, oracle, assertions)
            }
            _ => Err(anyhow!("Unsupported language/framework")),
        }
    }

    fn generate_pytest_code(
        &self,
        test_case: &TestCase,
        oracle: &GeneratedOracle,
        assertions: &[Assertion],
    ) -> Result<String> {
        let setup = oracle.setup_code.as_deref().unwrap_or("");
        let teardown = oracle.teardown_code.as_deref().unwrap_or("");

        let input_setup = test_case.inputs.iter()
            .map(|(name, value)| format!("    {} = {}", name, self.value_to_python(value)?))
            .collect::<Result<Vec<_>>>()?
            .join("\n");

        let assertion_code = assertions.iter()
            .map(|a| format!("    {}", a.code))
            .collect::<Vec<_>>()
            .join("\n");

        let timeout_decorator = if self.quality_requirements.require_timeout {
            format!("@pytest.mark.timeout({})\n", self.quality_requirements.timeout_seconds)
        } else {
            String::new()
        };

        Ok(format!(r#"
{timeout_decorator}def test_{test_id}():
    """
    {description}
    Category: {category:?}
    Priority: {priority:?}
    """
{setup}
    # Arrange
{input_setup}

    # Act
    result = function_under_test({args})

    # Assert
{assertion_code}
{teardown}
"#,
            timeout_decorator = timeout_decorator,
            test_id = test_case.id,
            description = test_case.description,
            category = test_case.category,
            priority = test_case.priority,
            setup = setup,
            input_setup = input_setup,
            args = test_case.inputs.keys().cloned().collect::<Vec<_>>().join(", "),
            assertion_code = assertion_code,
            teardown = teardown,
        ))
    }

    fn assess_flake_risk(&self, test_case: &TestCase) -> Result<FlakeRisk> {
        let mut risk_score = 0;

        // Check for timing-sensitive operations
        if self.involves_timing(test_case)? {
            risk_score += 2;
        }

        // Check for network operations
        if self.involves_network(test_case)? {
            risk_score += 3;
        }

        // Check for file system operations
        if self.involves_filesystem(test_case)? {
            risk_score += 2;
        }

        // Check for random/non-deterministic elements
        if self.involves_randomness(test_case)? {
            risk_score += 3;
        }

        // Check for shared state
        if self.involves_shared_state(test_case)? {
            risk_score += 2;
        }

        Ok(match risk_score {
            0 => FlakeRisk::None,
            1..=2 => FlakeRisk::Low,
            3..=5 => FlakeRisk::Medium,
            _ => FlakeRisk::High,
        })
    }
}
```

---

#### State 3.2: MutationTesting

**Purpose:** Verify tests can detect bugs by injecting mutations

```rust
// src-tauri/src/testing/intelligence/mutation_testing.rs

#[derive(Debug, Clone)]
pub struct MutationTester {
    code: String,
    tests: Vec<GeneratedTest>,
    language: ProgrammingLanguage,
    mutation_operators: Vec<MutationOperator>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutationOperator {
    // Arithmetic operators
    ArithmeticReplacement,     // + → -, * → /, etc.
    // Relational operators
    RelationalReplacement,     // < → <=, == → !=, etc.
    // Logical operators
    LogicalReplacement,        // && → ||, ! removal
    // Constant mutations
    ConstantReplacement,       // 0 → 1, true → false
    // Statement mutations
    StatementDeletion,         // Remove statements
    ReturnValueMutation,       // Return different value
    // Boundary mutations
    OffByOne,                  // +1, -1 to indices/limits
    BoundarySwap,              // < → <=, > → >=
    // Null/exception mutations
    NullReturn,                // Return null instead of value
    ExceptionRemoval,          // Remove error handling
    // Domain-specific
    ApiCallRemoval,            // Remove API/network calls
    ValidationBypass,          // Skip validation logic
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mutant {
    pub id: String,
    pub operator: MutationOperator,
    pub location: CodeLocation,
    pub original_code: String,
    pub mutated_code: String,
    pub full_mutated_source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationTestResult {
    pub total_mutants: usize,
    pub killed_mutants: usize,
    pub survived_mutants: usize,
    pub equivalent_mutants: usize,
    pub timeout_mutants: usize,
    pub mutation_score: f32,  // killed / (total - equivalent)
    pub surviving_mutants: Vec<SurvivingMutant>,
    pub weak_tests: Vec<WeakTest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivingMutant {
    pub mutant: Mutant,
    pub tests_run: Vec<String>,
    pub likely_reason: SurvivalReason,
    pub suggested_test: Option<String>,  // Suggested test to catch this
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SurvivalReason {
    NoTestCoverage,           // No test exercises this code
    WeakAssertions,           // Test runs code but doesn't check result
    EquivalentMutation,       // Mutation doesn't change behavior
    TimeoutTooShort,          // Test timed out before detecting
    UndetectableByTests,      // Mutation affects non-testable behavior
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeakTest {
    pub test_id: String,
    pub mutants_not_detected: Vec<String>,
    pub weakness_type: TestWeakness,
    pub suggested_improvement: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestWeakness {
    NoAssertions,
    WeakAssertions,
    MissingEdgeCases,
    NoErrorHandlingTest,
}

impl MutationTester {
    pub async fn run(&self) -> Result<MutationTestResult> {
        // Step 1: Generate mutants
        let mutants = self.generate_mutants().await?;

        let mut result = MutationTestResult {
            total_mutants: mutants.len(),
            killed_mutants: 0,
            survived_mutants: 0,
            equivalent_mutants: 0,
            timeout_mutants: 0,
            mutation_score: 0.0,
            surviving_mutants: Vec::new(),
            weak_tests: Vec::new(),
        };

        // Step 2: Run tests against each mutant
        for mutant in mutants {
            let mutant_result = self.test_mutant(&mutant).await?;

            match mutant_result {
                MutantStatus::Killed { by_test } => {
                    result.killed_mutants += 1;
                }
                MutantStatus::Survived { tests_run } => {
                    result.survived_mutants += 1;

                    let reason = self.analyze_survival(&mutant, &tests_run)?;
                    let suggested_test = self.suggest_killing_test(&mutant, &reason)?;

                    result.surviving_mutants.push(SurvivingMutant {
                        mutant,
                        tests_run,
                        likely_reason: reason,
                        suggested_test,
                    });
                }
                MutantStatus::Equivalent => {
                    result.equivalent_mutants += 1;
                }
                MutantStatus::Timeout => {
                    result.timeout_mutants += 1;
                }
            }
        }

        // Step 3: Calculate mutation score
        let killable = result.total_mutants - result.equivalent_mutants;
        result.mutation_score = if killable > 0 {
            result.killed_mutants as f32 / killable as f32
        } else {
            1.0
        };

        // Step 4: Identify weak tests
        result.weak_tests = self.identify_weak_tests(&result)?;

        Ok(result)
    }

    async fn generate_mutants(&self) -> Result<Vec<Mutant>> {
        let mut mutants = Vec::new();
        let ast = self.parse_code(&self.code)?;

        for operator in &self.mutation_operators {
            let applicable_locations = self.find_applicable_locations(&ast, operator)?;

            for location in applicable_locations {
                let mutant = self.create_mutant(&location, operator)?;

                // Skip trivially equivalent mutants
                if !self.is_trivially_equivalent(&mutant)? {
                    mutants.push(mutant);
                }
            }
        }

        // Limit mutants to avoid explosion (prioritize by operator importance)
        self.prioritize_and_limit_mutants(mutants, 100)
    }

    fn find_applicable_locations(
        &self,
        ast: &AST,
        operator: &MutationOperator,
    ) -> Result<Vec<CodeLocation>> {
        let mut locations = Vec::new();

        match operator {
            MutationOperator::ArithmeticReplacement => {
                // Find all arithmetic operations: +, -, *, /, %
                for node in ast.find_all(NodeType::BinaryOp) {
                    if matches!(node.operator, "+" | "-" | "*" | "/" | "%") {
                        locations.push(node.location.clone());
                    }
                }
            }
            MutationOperator::RelationalReplacement => {
                // Find all comparisons: <, >, <=, >=, ==, !=
                for node in ast.find_all(NodeType::Comparison) {
                    locations.push(node.location.clone());
                }
            }
            MutationOperator::OffByOne => {
                // Find array accesses and loop bounds
                for node in ast.find_all(NodeType::Subscript) {
                    locations.push(node.location.clone());
                }
                for node in ast.find_all(NodeType::ForRange) {
                    locations.push(node.location.clone());
                }
            }
            MutationOperator::ReturnValueMutation => {
                // Find return statements
                for node in ast.find_all(NodeType::Return) {
                    locations.push(node.location.clone());
                }
            }
            MutationOperator::StatementDeletion => {
                // Find deletable statements (not control flow)
                for node in ast.find_all(NodeType::Statement) {
                    if self.is_deletable(&node)? {
                        locations.push(node.location.clone());
                    }
                }
            }
            _ => {
                // Generic handling for other operators
            }
        }

        Ok(locations)
    }

    fn create_mutant(&self, location: &CodeLocation, operator: &MutationOperator) -> Result<Mutant> {
        let original = self.extract_code_at(location)?;
        let mutated = self.apply_mutation(&original, operator)?;
        let full_source = self.replace_in_source(&self.code, location, &mutated)?;

        Ok(Mutant {
            id: format!("mutant_{}_{}", operator.name(), location.start),
            operator: operator.clone(),
            location: location.clone(),
            original_code: original,
            mutated_code: mutated,
            full_mutated_source: full_source,
        })
    }

    fn apply_mutation(&self, original: &str, operator: &MutationOperator) -> Result<String> {
        match operator {
            MutationOperator::ArithmeticReplacement => {
                // Replace + with -, - with +, * with /, / with *
                let mutated = match original {
                    "+" => "-",
                    "-" => "+",
                    "*" => "/",
                    "/" => "*",
                    "%" => "+",
                    _ => return Err(anyhow!("Unexpected arithmetic operator")),
                };
                Ok(mutated.to_string())
            }
            MutationOperator::RelationalReplacement => {
                let mutated = match original {
                    "<" => "<=",
                    "<=" => "<",
                    ">" => ">=",
                    ">=" => ">",
                    "==" => "!=",
                    "!=" => "==",
                    _ => return Err(anyhow!("Unexpected relational operator")),
                };
                Ok(mutated.to_string())
            }
            MutationOperator::OffByOne => {
                // Add +1 or -1 to the expression
                Ok(format!("({} + 1)", original))
            }
            MutationOperator::NullReturn => {
                Ok("None".to_string()) // Python
            }
            MutationOperator::StatementDeletion => {
                Ok("pass".to_string()) // Python: replace with no-op
            }
            _ => Err(anyhow!("Mutation operator not implemented")),
        }
    }

    async fn test_mutant(&self, mutant: &Mutant) -> Result<MutantStatus> {
        // Write mutated code to temp file
        let temp_path = self.write_temp_file(&mutant.full_mutated_source)?;

        // Run all tests against mutant
        let mut tests_run = Vec::new();

        for test in &self.tests {
            tests_run.push(test.id.clone());

            let result = self.run_single_test(&test, &temp_path).await?;

            match result {
                TestResult::Failed { .. } => {
                    // Mutant killed by this test
                    return Ok(MutantStatus::Killed { by_test: test.id.clone() });
                }
                TestResult::Passed => {
                    // Test didn't detect mutation, continue
                }
                TestResult::Timeout => {
                    return Ok(MutantStatus::Timeout);
                }
                TestResult::Error { .. } => {
                    // Syntax error = mutant killed
                    return Ok(MutantStatus::Killed { by_test: test.id.clone() });
                }
            }
        }

        // All tests passed - mutant survived
        Ok(MutantStatus::Survived { tests_run })
    }

    fn suggest_killing_test(
        &self,
        mutant: &Mutant,
        reason: &SurvivalReason,
    ) -> Result<Option<String>> {
        match reason {
            SurvivalReason::NoTestCoverage => {
                // Generate test that exercises this code
                Ok(Some(format!(r#"
def test_kill_mutant_{}():
    """Test to detect mutation at {}"""
    # Setup input that exercises line {}
    input_value = <GENERATE_INPUT>

    # Expected result with original code
    expected = <EXPECTED_VALUE>

    result = function_under_test(input_value)

    # This assertion should fail with mutant
    assert result == expected, f"Expected {{expected}}, got {{result}}"
"#,
                    mutant.id,
                    mutant.location,
                    mutant.location.line
                )))
            }
            SurvivalReason::WeakAssertions => {
                Ok(Some(format!(r#"
# Strengthen existing assertions to detect mutation:
# Original: {}
# Mutated:  {}

# Add more specific assertion:
assert result == EXACT_EXPECTED_VALUE, "Value should be exactly X"
# Or add property assertion:
assert validate_property(result), "Result should satisfy property"
"#,
                    mutant.original_code,
                    mutant.mutated_code
                )))
            }
            SurvivalReason::EquivalentMutation => {
                Ok(None) // No test can kill equivalent mutant
            }
            _ => Ok(None),
        }
    }

    fn identify_weak_tests(&self, result: &MutationTestResult) -> Result<Vec<WeakTest>> {
        let mut weak_tests = Vec::new();

        // Group surviving mutants by test that should have caught them
        let mut test_failures: HashMap<String, Vec<String>> = HashMap::new();

        for survivor in &result.surviving_mutants {
            for test_id in &survivor.tests_run {
                test_failures
                    .entry(test_id.clone())
                    .or_default()
                    .push(survivor.mutant.id.clone());
            }
        }

        // Identify tests that ran many mutants but killed few
        for (test_id, not_detected) in test_failures {
            if not_detected.len() > 3 {
                let test = self.tests.iter().find(|t| t.id == test_id);
                if let Some(test) = test {
                    let weakness = self.classify_test_weakness(test, &not_detected)?;
                    let suggestion = self.suggest_test_improvement(test, &weakness)?;

                    weak_tests.push(WeakTest {
                        test_id,
                        mutants_not_detected: not_detected,
                        weakness_type: weakness,
                        suggested_improvement: suggestion,
                    });
                }
            }
        }

        Ok(weak_tests)
    }
}

#[derive(Debug, Clone)]
pub enum MutantStatus {
    Killed { by_test: String },
    Survived { tests_run: Vec<String> },
    Equivalent,
    Timeout,
}
```

---

#### State 3.3: AssertionAnalysis

**Purpose:** Analyze and strengthen test assertions

```rust
// src-tauri/src/testing/intelligence/assertion_analysis.rs

#[derive(Debug, Clone)]
pub struct AssertionAnalyzer {
    tests: Vec<GeneratedTest>,
    code_ast: AST,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssertionAnalysisResult {
    pub test_scores: Vec<TestAssertionScore>,
    pub weak_assertions: Vec<WeakAssertionReport>,
    pub missing_assertions: Vec<MissingAssertionReport>,
    pub overall_score: f32,
    pub recommendations: Vec<AssertionRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestAssertionScore {
    pub test_id: String,
    pub total_assertions: usize,
    pub strong_assertions: usize,
    pub medium_assertions: usize,
    pub weak_assertions: usize,
    pub score: f32,  // 0.0 - 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeakAssertionReport {
    pub test_id: String,
    pub assertion: Assertion,
    pub weakness: AssertionWeakness,
    pub strengthened_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssertionWeakness {
    TooGeneric,           // assert result is not None
    MissingMessage,       // No error message
    ImplicitTypeCheck,    // Only checks truthy, not actual value
    NoRangeValidation,    // Numeric result not bounded
    NoStructureCheck,     // Object fields not validated
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingAssertionReport {
    pub test_id: String,
    pub code_element: String,          // What should be asserted
    pub suggested_assertion: String,
    pub importance: AssertionImportance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssertionImportance {
    Critical,   // Core behavior not verified
    Important,  // Significant behavior gap
    Optional,   // Nice to have
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssertionRecommendation {
    pub test_id: String,
    pub action: RecommendedAction,
    pub code: String,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendedAction {
    AddAssertion,
    StrengthenAssertion,
    AddErrorMessage,
    AddTypeCheck,
    AddRangeCheck,
}

impl AssertionAnalyzer {
    pub async fn analyze(&self) -> Result<AssertionAnalysisResult> {
        let mut result = AssertionAnalysisResult::default();

        for test in &self.tests {
            // Score this test's assertions
            let score = self.score_test_assertions(test)?;
            result.test_scores.push(score.clone());

            // Identify weak assertions
            for assertion in &test.assertions {
                if let Some(weakness) = self.identify_weakness(assertion)? {
                    let strengthened = self.strengthen_assertion(assertion, &weakness)?;
                    result.weak_assertions.push(WeakAssertionReport {
                        test_id: test.id.clone(),
                        assertion: assertion.clone(),
                        weakness,
                        strengthened_version: strengthened,
                    });
                }
            }

            // Identify missing assertions
            let missing = self.find_missing_assertions(test)?;
            result.missing_assertions.extend(missing);
        }

        // Calculate overall score
        result.overall_score = result.test_scores.iter()
            .map(|s| s.score)
            .sum::<f32>() / result.test_scores.len() as f32;

        // Generate recommendations
        result.recommendations = self.generate_recommendations(&result)?;

        Ok(result)
    }

    fn score_test_assertions(&self, test: &GeneratedTest) -> Result<TestAssertionScore> {
        let mut strong = 0;
        let mut medium = 0;
        let mut weak = 0;

        for assertion in &test.assertions {
            match assertion.strength {
                AssertionStrength::Strong => strong += 1,
                AssertionStrength::Medium => medium += 1,
                AssertionStrength::Weak => weak += 1,
            }
        }

        let total = strong + medium + weak;
        let score = if total > 0 {
            (strong as f32 * 1.0 + medium as f32 * 0.5 + weak as f32 * 0.1) / total as f32
        } else {
            0.0
        };

        Ok(TestAssertionScore {
            test_id: test.id.clone(),
            total_assertions: total,
            strong_assertions: strong,
            medium_assertions: medium,
            weak_assertions: weak,
            score,
        })
    }

    fn identify_weakness(&self, assertion: &Assertion) -> Result<Option<AssertionWeakness>> {
        // Check for generic assertions
        if self.is_too_generic(&assertion.code)? {
            return Ok(Some(AssertionWeakness::TooGeneric));
        }

        // Check for missing error message
        if assertion.message.is_empty() || assertion.message == "Assertion failed" {
            return Ok(Some(AssertionWeakness::MissingMessage));
        }

        // Check for implicit type checks (assert result instead of assert result == expected)
        if self.is_implicit_type_check(&assertion.code)? {
            return Ok(Some(AssertionWeakness::ImplicitTypeCheck));
        }

        Ok(None)
    }

    fn is_too_generic(&self, code: &str) -> Result<bool> {
        let generic_patterns = vec![
            r"assert\s+result\s*$",                    // Just assert result
            r"assert\s+result\s+is\s+not\s+None",     // Only checks not None
            r"assertTrue\(.*\)",                       // Generic truthy check
            r"expect\(.*\)\.toBeTruthy\(\)",          // Jest truthy
            r"assert!\(.*\.is_some\(\)\)",            // Rust just checks Some
        ];

        for pattern in generic_patterns {
            if regex::Regex::new(pattern)?.is_match(code) {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn strengthen_assertion(
        &self,
        assertion: &Assertion,
        weakness: &AssertionWeakness,
    ) -> Result<String> {
        match weakness {
            AssertionWeakness::TooGeneric => {
                // Convert generic assertion to specific value check
                Ok(format!(
                    "assert result == EXPECTED_VALUE, \"Expected EXPECTED_VALUE, got {{result}}\"",
                ))
            }
            AssertionWeakness::MissingMessage => {
                // Add descriptive error message
                Ok(format!(
                    "{}, \"{}\"",
                    assertion.code.trim_end_matches(')'),
                    self.generate_error_message(assertion)?
                ))
            }
            AssertionWeakness::ImplicitTypeCheck => {
                // Add explicit value check
                Ok(format!(
                    "assert isinstance(result, EXPECTED_TYPE) and result == EXPECTED_VALUE"
                ))
            }
            AssertionWeakness::NoRangeValidation => {
                // Add range bounds
                Ok(format!(
                    "assert MIN_VALUE <= result <= MAX_VALUE, f\"Result {{result}} outside valid range\""
                ))
            }
            AssertionWeakness::NoStructureCheck => {
                // Add field-level assertions
                Ok(format!(
                    "assert 'required_field' in result\nassert result['required_field'] == EXPECTED"
                ))
            }
        }
    }

    fn find_missing_assertions(&self, test: &GeneratedTest) -> Result<Vec<MissingAssertionReport>> {
        let mut missing = Vec::new();

        // Get expected behaviors from test's oracle
        // Compare against actual assertions

        // Check: Does test verify return type?
        if !test.assertions.iter().any(|a| matches!(a.assertion_type, AssertionType::TypeCheck)) {
            missing.push(MissingAssertionReport {
                test_id: test.id.clone(),
                code_element: "return type".to_string(),
                suggested_assertion: "assert isinstance(result, ExpectedType)".to_string(),
                importance: AssertionImportance::Important,
            });
        }

        // Check: Does test verify no exceptions on valid input?
        if test.metadata.category == TestCaseCategory::HappyPath {
            if !test.assertions.iter().any(|a| matches!(a.assertion_type, AssertionType::DoesNotThrow)) {
                // Implicit in most tests, but good to be explicit
            }
        }

        // Check: Does error path test verify correct exception type?
        if test.metadata.category == TestCaseCategory::ErrorCondition {
            if !test.assertions.iter().any(|a| matches!(a.assertion_type, AssertionType::ThrowsException)) {
                missing.push(MissingAssertionReport {
                    test_id: test.id.clone(),
                    code_element: "exception type".to_string(),
                    suggested_assertion: "with pytest.raises(ExpectedException):".to_string(),
                    importance: AssertionImportance::Critical,
                });
            }
        }

        Ok(missing)
    }
}
```

---

#### State 3.4: FlakeDetection

**Purpose:** Identify and quarantine flaky tests

```rust
// src-tauri/src/testing/intelligence/flake_detection.rs

#[derive(Debug, Clone)]
pub struct FlakeDetector {
    tests: Vec<GeneratedTest>,
    repeat_count: u32,
    variance_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlakeDetectionResult {
    pub stable_tests: Vec<StableTest>,
    pub flaky_tests: Vec<FlakyTest>,
    pub quarantined: Vec<QuarantinedTest>,
    pub flake_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StableTest {
    pub test_id: String,
    pub runs: usize,
    pub all_passed: bool,
    pub avg_duration_ms: u64,
    pub duration_variance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlakyTest {
    pub test_id: String,
    pub runs: usize,
    pub passes: usize,
    pub failures: usize,
    pub flake_rate: f32,
    pub failure_patterns: Vec<FailurePattern>,
    pub likely_causes: Vec<FlakeCause>,
    pub suggested_fixes: Vec<FlakeFix>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailurePattern {
    pub error_message: String,
    pub occurrence_count: usize,
    pub timing: FailureTiming,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureTiming {
    Random,                    // No pattern
    AfterLongRun,              // Fails after long duration
    EarlyInSequence,           // Fails when run early
    LateInSequence,            // Fails when run late
    ConcurrentWithOther,       // Fails with specific other tests
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlakeCause {
    TimingDependency { details: String },
    SharedState { state_name: String },
    NetworkDependency { endpoint: String },
    FileSystemRace { path: String },
    OrderDependency { depends_on: String },
    ResourceExhaustion { resource: String },
    NonDeterministicCode { location: String },
    AsyncTimingIssue { details: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlakeFix {
    pub cause: FlakeCause,
    pub fix_type: FlakeFixType,
    pub code_change: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlakeFixType {
    AddRetry { max_attempts: u32 },
    AddTimeout { seconds: u64 },
    AddWaitCondition { condition: String },
    IsolateState { isolation_code: String },
    MockDependency { mock_code: String },
    AddSetupTeardown { setup: String, teardown: String },
    MakeDetrministic { change: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantinedTest {
    pub test: FlakyTest,
    pub quarantine_reason: String,
    pub quarantine_until: Option<DateTime<Utc>>,
    pub auto_unquarantine_on_fix: bool,
}

impl FlakeDetector {
    pub async fn detect(&self) -> Result<FlakeDetectionResult> {
        let mut result = FlakeDetectionResult {
            stable_tests: Vec::new(),
            flaky_tests: Vec::new(),
            quarantined: Vec::new(),
            flake_rate: 0.0,
        };

        for test in &self.tests {
            let runs = self.run_test_multiple_times(test, self.repeat_count).await?;

            let passes = runs.iter().filter(|r| r.passed).count();
            let failures = runs.len() - passes;

            if failures == 0 {
                // All runs passed - stable test
                result.stable_tests.push(StableTest {
                    test_id: test.id.clone(),
                    runs: runs.len(),
                    all_passed: true,
                    avg_duration_ms: self.calculate_avg_duration(&runs),
                    duration_variance: self.calculate_variance(&runs),
                });
            } else if passes == 0 {
                // All runs failed - consistently failing, not flaky
                result.stable_tests.push(StableTest {
                    test_id: test.id.clone(),
                    runs: runs.len(),
                    all_passed: false,
                    avg_duration_ms: self.calculate_avg_duration(&runs),
                    duration_variance: self.calculate_variance(&runs),
                });
            } else {
                // Mixed results - flaky test
                let flake_rate = failures as f32 / runs.len() as f32;
                let failure_patterns = self.analyze_failure_patterns(&runs)?;
                let likely_causes = self.identify_flake_causes(test, &failure_patterns)?;
                let suggested_fixes = self.suggest_fixes(&likely_causes)?;

                let flaky = FlakyTest {
                    test_id: test.id.clone(),
                    runs: runs.len(),
                    passes,
                    failures,
                    flake_rate,
                    failure_patterns,
                    likely_causes: likely_causes.clone(),
                    suggested_fixes,
                };

                // Quarantine if flake rate is high
                if flake_rate > self.variance_threshold {
                    result.quarantined.push(QuarantinedTest {
                        test: flaky.clone(),
                        quarantine_reason: format!(
                            "Flake rate {}% exceeds threshold {}%",
                            flake_rate * 100.0,
                            self.variance_threshold * 100.0
                        ),
                        quarantine_until: None,
                        auto_unquarantine_on_fix: true,
                    });
                }

                result.flaky_tests.push(flaky);
            }
        }

        result.flake_rate = result.flaky_tests.len() as f32 / self.tests.len() as f32;

        Ok(result)
    }

    async fn run_test_multiple_times(
        &self,
        test: &GeneratedTest,
        count: u32,
    ) -> Result<Vec<TestRunResult>> {
        let mut runs = Vec::new();

        for i in 0..count {
            // Clean state between runs
            self.reset_test_environment().await?;

            // Run with timing
            let start = Instant::now();
            let result = self.run_single_test(test).await?;
            let duration = start.elapsed();

            runs.push(TestRunResult {
                run_number: i,
                passed: result.is_pass(),
                duration_ms: duration.as_millis() as u64,
                error_message: result.error_message(),
                stdout: result.stdout,
                stderr: result.stderr,
            });

            // Small delay to surface timing issues
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(runs)
    }

    fn identify_flake_causes(
        &self,
        test: &GeneratedTest,
        patterns: &[FailurePattern],
    ) -> Result<Vec<FlakeCause>> {
        let mut causes = Vec::new();

        // Analyze test code for common flake sources
        let test_code = &test.code;

        // Check for timing-related code
        if test_code.contains("sleep") || test_code.contains("timeout") || test_code.contains("wait") {
            causes.push(FlakeCause::TimingDependency {
                details: "Test contains explicit timing code".to_string(),
            });
        }

        // Check for async code without proper awaiting
        if test_code.contains("async") || test_code.contains("await") {
            causes.push(FlakeCause::AsyncTimingIssue {
                details: "Async operations may have race conditions".to_string(),
            });
        }

        // Check for network calls
        if test_code.contains("http") || test_code.contains("fetch") || test_code.contains("request") {
            causes.push(FlakeCause::NetworkDependency {
                endpoint: "External network call detected".to_string(),
            });
        }

        // Check for file system operations
        if test_code.contains("open(") || test_code.contains("write") || test_code.contains("read") {
            causes.push(FlakeCause::FileSystemRace {
                path: "File system operations detected".to_string(),
            });
        }

        // Analyze failure patterns for clues
        for pattern in patterns {
            if pattern.error_message.contains("timeout") {
                causes.push(FlakeCause::TimingDependency {
                    details: format!("Timeout error: {}", pattern.error_message),
                });
            }
            if pattern.error_message.contains("connection") || pattern.error_message.contains("refused") {
                causes.push(FlakeCause::NetworkDependency {
                    endpoint: pattern.error_message.clone(),
                });
            }
        }

        Ok(causes)
    }

    fn suggest_fixes(&self, causes: &[FlakeCause]) -> Result<Vec<FlakeFix>> {
        let mut fixes = Vec::new();

        for cause in causes {
            let fix = match cause {
                FlakeCause::TimingDependency { details } => {
                    FlakeFix {
                        cause: cause.clone(),
                        fix_type: FlakeFixType::AddWaitCondition {
                            condition: "wait_for_condition(lambda: condition_met(), timeout=10)".to_string(),
                        },
                        code_change: r#"
# Replace fixed sleep with condition-based wait
from tenacity import retry, wait_exponential, stop_after_delay

@retry(wait=wait_exponential(multiplier=0.1), stop=stop_after_delay(10))
def wait_for_ready():
    assert is_ready(), "Not ready yet"
"#.to_string(),
                        confidence: 0.8,
                    }
                }
                FlakeCause::NetworkDependency { endpoint } => {
                    FlakeFix {
                        cause: cause.clone(),
                        fix_type: FlakeFixType::MockDependency {
                            mock_code: format!("@mock.patch('requests.get')\ndef test_...(mock_get):\n    mock_get.return_value.json.return_value = {{}}")
                        },
                        code_change: r#"
# Mock external dependency
@pytest.fixture
def mock_external_api(mocker):
    return mocker.patch('module.external_api_call', return_value={'status': 'ok'})

def test_with_mock(mock_external_api):
    result = function_under_test()
    assert result == expected
"#.to_string(),
                        confidence: 0.9,
                    }
                }
                FlakeCause::SharedState { state_name } => {
                    FlakeFix {
                        cause: cause.clone(),
                        fix_type: FlakeFixType::IsolateState {
                            isolation_code: "Use fresh instance per test".to_string(),
                        },
                        code_change: format!(r#"
# Isolate shared state
@pytest.fixture(autouse=True)
def reset_{}():
    # Setup: fresh state
    original = {}.copy()
    yield
    # Teardown: restore state
    {}.clear()
    {}.update(original)
"#, state_name, state_name, state_name, state_name),
                        confidence: 0.85,
                    }
                }
                FlakeCause::AsyncTimingIssue { details } => {
                    FlakeFix {
                        cause: cause.clone(),
                        fix_type: FlakeFixType::AddRetry { max_attempts: 3 },
                        code_change: r#"
# Add retry for async timing issues
import asyncio
from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.1))
async def test_async_operation():
    result = await async_function()
    assert result == expected
"#.to_string(),
                        confidence: 0.7,
                    }
                }
                _ => continue,
            };

            fixes.push(fix);
        }

        Ok(fixes)
    }
}

#[derive(Debug, Clone)]
struct TestRunResult {
    run_number: u32,
    passed: bool,
    duration_ms: u64,
    error_message: Option<String>,
    stdout: String,
    stderr: String,
}
```

---

### Phase 4: Execution Intelligence (2 states)

#### State 4.1: ExecutionTracing

**Purpose:** Capture rich execution context for debugging

````rust
// src-tauri/src/testing/intelligence/execution_tracing.rs

#[derive(Debug, Clone)]
pub struct ExecutionTracer {
    code: String,
    language: ProgrammingLanguage,
    trace_depth: TraceDepth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraceDepth {
    Minimal,    // Just entry/exit and exceptions
    Standard,   // + variable values at key points
    Detailed,   // + all variable assignments
    Full,       // + all expressions evaluated
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    pub test_id: String,
    pub events: Vec<TraceEvent>,
    pub final_state: ProgramState,
    pub exception: Option<ExceptionInfo>,
    pub coverage: LineCoverage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraceEvent {
    FunctionEntry {
        function_name: String,
        arguments: HashMap<String, TraceValue>,
        timestamp: u64,
    },
    FunctionExit {
        function_name: String,
        return_value: Option<TraceValue>,
        duration_ns: u64,
    },
    VariableAssignment {
        variable_name: String,
        value: TraceValue,
        location: CodeLocation,
    },
    BranchTaken {
        condition: String,
        evaluated_to: bool,
        location: CodeLocation,
    },
    LoopIteration {
        loop_id: String,
        iteration: usize,
        state: HashMap<String, TraceValue>,
    },
    ExceptionRaised {
        exception_type: String,
        message: String,
        location: CodeLocation,
        stack_trace: Vec<StackFrame>,
    },
    AssertionEvaluated {
        assertion: String,
        passed: bool,
        actual_values: HashMap<String, TraceValue>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceValue {
    pub type_name: String,
    pub value_repr: String,       // String representation
    pub value_summary: String,    // Truncated for large values
    pub size: Option<usize>,      // For collections
    pub is_truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExceptionInfo {
    pub exception_type: String,
    pub message: String,
    pub location: CodeLocation,
    pub stack_trace: Vec<StackFrame>,
    pub local_variables: HashMap<String, TraceValue>,
    pub relevant_context: Vec<ContextItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackFrame {
    pub function_name: String,
    pub file_path: String,
    pub line_number: usize,
    pub local_variables: HashMap<String, TraceValue>,
    pub code_snippet: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextItem {
    pub description: String,
    pub value: String,
    pub relevance: f32,
}

impl ExecutionTracer {
    pub async fn trace_execution(
        &self,
        test: &GeneratedTest,
    ) -> Result<ExecutionTrace> {
        // Instrument code for tracing
        let instrumented = self.instrument_code(&self.code)?;

        // Execute with tracing enabled
        let trace_collector = TraceCollector::new();
        let result = self.execute_instrumented(&instrumented, &test, &trace_collector).await?;

        // Build execution trace
        let mut trace = ExecutionTrace {
            test_id: test.id.clone(),
            events: trace_collector.events(),
            final_state: trace_collector.final_state(),
            exception: None,
            coverage: trace_collector.coverage(),
        };

        // If execution failed, capture exception details
        if let ExecutionResult::Failed { error } = result {
            trace.exception = Some(self.capture_exception_info(&error, &trace_collector)?);
        }

        Ok(trace)
    }

    fn instrument_code(&self, code: &str) -> Result<String> {
        match self.language {
            ProgrammingLanguage::Python => {
                self.instrument_python(code)
            }
            ProgrammingLanguage::JavaScript => {
                self.instrument_javascript(code)
            }
            _ => Err(anyhow!("Tracing not supported for this language")),
        }
    }

    fn instrument_python(&self, code: &str) -> Result<String> {
        // Add sys.settrace for function entry/exit
        // Add variable capture at key points
        Ok(format!(r#"
import sys
import json
from functools import wraps

_trace_events = []

def _trace_function(frame, event, arg):
    if event == 'call':
        _trace_events.append({{
            'type': 'function_entry',
            'function': frame.f_code.co_name,
            'args': {{k: repr(v)[:100] for k, v in frame.f_locals.items()}},
            'line': frame.f_lineno,
        }})
    elif event == 'return':
        _trace_events.append({{
            'type': 'function_exit',
            'function': frame.f_code.co_name,
            'return_value': repr(arg)[:100] if arg is not None else None,
        }})
    elif event == 'exception':
        exc_type, exc_value, exc_tb = arg
        _trace_events.append({{
            'type': 'exception',
            'exception_type': exc_type.__name__,
            'message': str(exc_value),
            'line': frame.f_lineno,
            'locals': {{k: repr(v)[:100] for k, v in frame.f_locals.items()}},
        }})
    return _trace_function

sys.settrace(_trace_function)

# Original code
{}

sys.settrace(None)
print(json.dumps(_trace_events))
"#, code))
    }

    fn capture_exception_info(
        &self,
        error: &ExecutionError,
        collector: &TraceCollector,
    ) -> Result<ExceptionInfo> {
        let stack_frames = error.stack_trace.iter()
            .map(|frame| {
                let locals = collector.get_locals_at_frame(frame)?;
                Ok(StackFrame {
                    function_name: frame.function.clone(),
                    file_path: frame.file.clone(),
                    line_number: frame.line,
                    local_variables: locals,
                    code_snippet: self.get_code_snippet(frame.line, 3)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Get most recent variable values leading to exception
        let local_variables = collector.get_locals_at_exception()?;

        // Find relevant context (what data led to this state)
        let relevant_context = self.find_relevant_context(&local_variables, &stack_frames)?;

        Ok(ExceptionInfo {
            exception_type: error.exception_type.clone(),
            message: error.message.clone(),
            location: error.location.clone(),
            stack_trace: stack_frames,
            local_variables,
            relevant_context,
        })
    }

    fn find_relevant_context(
        &self,
        locals: &HashMap<String, TraceValue>,
        stack: &[StackFrame],
    ) -> Result<Vec<ContextItem>> {
        let mut context = Vec::new();

        // Find variables that appear in exception message
        for (name, value) in locals {
            if self.is_likely_relevant(name, value)? {
                context.push(ContextItem {
                    description: format!("Variable '{}' at failure point", name),
                    value: value.value_repr.clone(),
                    relevance: 0.9,
                });
            }
        }

        // Add array/collection sizes (common in index errors)
        for (name, value) in locals {
            if let Some(size) = value.size {
                context.push(ContextItem {
                    description: format!("Size of '{}'", name),
                    value: size.to_string(),
                    relevance: 0.8,
                });
            }
        }

        // Add loop iteration counts from trace
        // Add recent assignments that might be relevant

        Ok(context)
    }

    pub fn format_for_llm(&self, trace: &ExecutionTrace) -> String {
        let mut output = String::new();

        if let Some(exception) = &trace.exception {
            output.push_str(&format!("## Exception Details\n\n"));
            output.push_str(&format!("**Type:** {}\n", exception.exception_type));
            output.push_str(&format!("**Message:** {}\n", exception.message));
            output.push_str(&format!("**Location:** {}:{}\n\n",
                exception.location.file, exception.location.line));

            output.push_str("### Stack Trace\n\n");
            for (i, frame) in exception.stack_trace.iter().enumerate() {
                output.push_str(&format!("{}. `{}` at {}:{}\n",
                    i + 1, frame.function_name, frame.file_path, frame.line_number));
                output.push_str(&format!("   ```\n   {}\n   ```\n", frame.code_snippet));

                if !frame.local_variables.is_empty() {
                    output.push_str("   **Local variables:**\n");
                    for (name, value) in &frame.local_variables {
                        output.push_str(&format!("   - {}: {} = {}\n",
                            name, value.type_name, value.value_summary));
                    }
                }
                output.push_str("\n");
            }

            output.push_str("### Relevant Context\n\n");
            for item in &exception.relevant_context {
                output.push_str(&format!("- **{}:** {}\n", item.description, item.value));
            }
        }

        // Add execution path summary
        output.push_str("\n### Execution Path\n\n");
        let path_summary = self.summarize_execution_path(&trace.events);
        output.push_str(&path_summary);

        output
    }

    fn summarize_execution_path(&self, events: &[TraceEvent]) -> String {
        let mut summary = String::new();

        // Extract function call sequence
        let calls: Vec<_> = events.iter()
            .filter_map(|e| match e {
                TraceEvent::FunctionEntry { function_name, .. } => Some(function_name.clone()),
                _ => None,
            })
            .collect();

        summary.push_str(&format!("**Call sequence:** {}\n", calls.join(" → ")));

        // Extract branches taken
        let branches: Vec<_> = events.iter()
            .filter_map(|e| match e {
                TraceEvent::BranchTaken { condition, evaluated_to, .. } => {
                    Some(format!("{} = {}", condition, evaluated_to))
                }
                _ => None,
            })
            .collect();

        if !branches.is_empty() {
            summary.push_str(&format!("**Branches:** {}\n", branches.join(", ")));
        }

        summary
    }
}
````

---

#### State 4.2: SemanticVerification

**Purpose:** Verify code actually fulfills user intent

```rust
// src-tauri/src/testing/intelligence/semantic_verification.rs

#[derive(Debug, Clone)]
pub struct SemanticVerifier {
    original_intent: String,
    expectations: Vec<BehavioralExpectation>,
    generated_code: String,
    test_results: Vec<TestResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticVerificationResult {
    pub overall_fulfillment: f32,
    pub expectation_results: Vec<ExpectationFulfillment>,
    pub unmet_expectations: Vec<UnmetExpectation>,
    pub unexpected_behaviors: Vec<UnexpectedBehavior>,
    pub confidence: VerificationConfidence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectationFulfillment {
    pub expectation_id: String,
    pub description: String,
    pub status: FulfillmentStatus,
    pub evidence: Vec<Evidence>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FulfillmentStatus {
    Met,
    PartiallyMet { details: String },
    Unmet { reason: String },
    Untested { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub evidence_type: EvidenceType,
    pub description: String,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceType {
    TestPassed,
    TestFailed,
    CodeAnalysis,
    BehaviorObserved,
    PropertyHolds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnmetExpectation {
    pub expectation: BehavioralExpectation,
    pub reason: String,
    pub suggested_fix: Option<String>,
    pub severity: Severity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnexpectedBehavior {
    pub description: String,
    pub observed: String,
    pub potential_issue: String,
    pub severity: Severity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Critical,
    Major,
    Minor,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfidence {
    pub level: ConfidenceLevel,
    pub factors: Vec<ConfidenceFactor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    High,       // >90% expectations verified with strong evidence
    Medium,     // 70-90% expectations verified
    Low,        // <70% expectations verified
    Unknown,    // Insufficient data
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceFactor {
    pub factor: String,
    pub impact: f32,
    pub details: String,
}

impl SemanticVerifier {
    pub async fn verify(&self, llm: &LLMOrchestrator) -> Result<SemanticVerificationResult> {
        let mut result = SemanticVerificationResult::default();

        // Step 1: Verify each expectation
        for expectation in &self.expectations {
            let fulfillment = self.verify_expectation(expectation, llm).await?;

            if matches!(fulfillment.status, FulfillmentStatus::Unmet { .. }) {
                result.unmet_expectations.push(UnmetExpectation {
                    expectation: expectation.clone(),
                    reason: match &fulfillment.status {
                        FulfillmentStatus::Unmet { reason } => reason.clone(),
                        _ => "Unknown".to_string(),
                    },
                    suggested_fix: self.suggest_fix_for_unmet(expectation)?,
                    severity: self.assess_severity(expectation)?,
                });
            }

            result.expectation_results.push(fulfillment);
        }

        // Step 2: Look for unexpected behaviors
        result.unexpected_behaviors = self.find_unexpected_behaviors(llm).await?;

        // Step 3: Calculate overall fulfillment
        let met_count = result.expectation_results.iter()
            .filter(|e| matches!(e.status, FulfillmentStatus::Met))
            .count();
        result.overall_fulfillment = met_count as f32 / self.expectations.len() as f32;

        // Step 4: Assess confidence
        result.confidence = self.assess_confidence(&result)?;

        Ok(result)
    }

    async fn verify_expectation(
        &self,
        expectation: &BehavioralExpectation,
        llm: &LLMOrchestrator,
    ) -> Result<ExpectationFulfillment> {
        let mut evidence = Vec::new();

        // Check if any test directly verifies this expectation
        let related_tests: Vec<_> = self.test_results.iter()
            .filter(|t| t.oracle_id == expectation.id)
            .collect();

        if !related_tests.is_empty() {
            let all_passed = related_tests.iter().all(|t| t.passed);
            evidence.push(Evidence {
                evidence_type: if all_passed { EvidenceType::TestPassed } else { EvidenceType::TestFailed },
                description: format!("{}/{} related tests passed",
                    related_tests.iter().filter(|t| t.passed).count(),
                    related_tests.len()),
                source: "Test execution".to_string(),
            });

            if all_passed {
                return Ok(ExpectationFulfillment {
                    expectation_id: expectation.id.clone(),
                    description: expectation.description.clone(),
                    status: FulfillmentStatus::Met,
                    evidence,
                    confidence: 0.9,
                });
            }
        }

        // Analyze code to check if expectation is implemented
        let code_evidence = self.analyze_code_for_expectation(expectation)?;
        evidence.extend(code_evidence);

        // Use LLM to assess fulfillment if unclear
        if evidence.is_empty() || evidence.iter().any(|e| e.evidence_type == EvidenceType::CodeAnalysis) {
            let llm_assessment = self.llm_assess_fulfillment(expectation, llm).await?;
            evidence.push(llm_assessment);
        }

        // Determine final status
        let status = self.determine_fulfillment_status(&evidence)?;
        let confidence = self.calculate_evidence_confidence(&evidence);

        Ok(ExpectationFulfillment {
            expectation_id: expectation.id.clone(),
            description: expectation.description.clone(),
            status,
            evidence,
            confidence,
        })
    }

    fn analyze_code_for_expectation(
        &self,
        expectation: &BehavioralExpectation,
    ) -> Result<Vec<Evidence>> {
        let mut evidence = Vec::new();

        // Check for precondition handling
        for precondition in &expectation.preconditions {
            if let Some(check_code) = &precondition.check_code {
                let pattern = self.extract_pattern_from_check(check_code)?;
                if self.generated_code.contains(&pattern) {
                    evidence.push(Evidence {
                        evidence_type: EvidenceType::CodeAnalysis,
                        description: format!("Precondition '{}' appears to be checked",
                            precondition.natural_language),
                        source: "Static analysis".to_string(),
                    });
                }
            }
        }

        // Check for postcondition implementation
        for postcondition in &expectation.postconditions {
            // Look for code that could produce the expected postcondition
            // This is a heuristic analysis
        }

        Ok(evidence)
    }

    async fn find_unexpected_behaviors(
        &self,
        llm: &LLMOrchestrator,
    ) -> Result<Vec<UnexpectedBehavior>> {
        let mut unexpected = Vec::new();

        // Analyze test results for unexpected patterns
        for test_result in &self.test_results {
            if test_result.passed {
                // Even passing tests might reveal unexpected behaviors
                if let Some(output) = &test_result.output {
                    let concerns = self.analyze_output_for_concerns(output)?;
                    unexpected.extend(concerns);
                }
            } else {
                // Failing test might indicate unexpected behavior
                if let Some(error) = &test_result.error {
                    // Check if this is an expected error or unexpected
                    let is_expected = self.expectations.iter()
                        .any(|e| matches!(e.expectation_type,
                            ExpectationType::ErrorHandling { .. }));

                    if !is_expected {
                        unexpected.push(UnexpectedBehavior {
                            description: "Unexpected error occurred".to_string(),
                            observed: error.clone(),
                            potential_issue: "Code may not handle this error case".to_string(),
                            severity: Severity::Major,
                        });
                    }
                }
            }
        }

        // Use LLM to identify potential unexpected behaviors
        let llm_findings = self.llm_find_unexpected(llm).await?;
        unexpected.extend(llm_findings);

        Ok(unexpected)
    }

    fn assess_confidence(&self, result: &SemanticVerificationResult) -> Result<VerificationConfidence> {
        let mut factors = Vec::new();
        let mut total_impact = 0.0;

        // Factor 1: Percentage of expectations tested
        let tested_ratio = result.expectation_results.iter()
            .filter(|e| !matches!(e.status, FulfillmentStatus::Untested { .. }))
            .count() as f32 / result.expectation_results.len() as f32;
        factors.push(ConfidenceFactor {
            factor: "Test coverage of expectations".to_string(),
            impact: tested_ratio * 0.4,
            details: format!("{}% of expectations have test evidence", tested_ratio * 100.0),
        });
        total_impact += tested_ratio * 0.4;

        // Factor 2: Strong evidence ratio
        let strong_evidence = result.expectation_results.iter()
            .filter(|e| e.confidence > 0.8)
            .count() as f32 / result.expectation_results.len() as f32;
        factors.push(ConfidenceFactor {
            factor: "Evidence strength".to_string(),
            impact: strong_evidence * 0.3,
            details: format!("{}% of verifications have strong evidence", strong_evidence * 100.0),
        });
        total_impact += strong_evidence * 0.3;

        // Factor 3: No unexpected behaviors
        let no_unexpected = result.unexpected_behaviors.is_empty();
        factors.push(ConfidenceFactor {
            factor: "No unexpected behaviors".to_string(),
            impact: if no_unexpected { 0.3 } else { 0.0 },
            details: if no_unexpected {
                "No unexpected behaviors detected".to_string()
            } else {
                format!("{} unexpected behaviors found", result.unexpected_behaviors.len())
            },
        });
        if no_unexpected { total_impact += 0.3; }

        let level = if total_impact > 0.9 {
            ConfidenceLevel::High
        } else if total_impact > 0.7 {
            ConfidenceLevel::Medium
        } else if total_impact > 0.5 {
            ConfidenceLevel::Low
        } else {
            ConfidenceLevel::Unknown
        };

        Ok(VerificationConfidence { level, factors })
    }
}
```

---

## Integration with Existing State Machines

### Enhanced CodeGen State Machine

Add new state after `SecurityScanning`:

```rust
// Addition to existing CodeGen state machine

pub enum CodeGenState {
    // ... existing states ...

    SecurityScanning,

    // NEW: Testability validation before completion
    TestabilityValidation,  // NEW STATE

    FixingIssues,
    Complete,
    Failed,
}

// New state implementation
impl TestabilityValidation {
    pub async fn execute(&mut self, context: &mut Context) -> Result<StateTransition> {
        let analyzer = TestabilityAnalyzer::new(&context.generated_code);
        let result = analyzer.analyze().await?;

        // Check testability score
        if result.testability_score < 0.7 {
            // Code is hard to test - regenerate with testability hints
            context.add_regeneration_hint(format!(
                "Improve testability: {}. Suggestions: {}",
                result.issues.join(", "),
                result.suggestions.join(", ")
            ));

            return Ok(StateTransition::To(CodeGenState::CodeGeneration));
        }

        // Code is testable - proceed to testing
        Ok(StateTransition::To(CodeGenState::Complete))
    }
}

#[derive(Debug, Clone)]
pub struct TestabilityAnalyzer {
    code: String,
    ast: AST,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestabilityResult {
    pub testability_score: f32,
    pub issues: Vec<String>,
    pub suggestions: Vec<String>,
    pub metrics: TestabilityMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestabilityMetrics {
    pub coupling_score: f32,          // Low is better
    pub cohesion_score: f32,          // High is better
    pub dependency_injection: bool,   // Does it use DI?
    pub pure_function_ratio: f32,     // % of pure functions
    pub external_dependency_count: usize,
    pub mockable_dependencies: usize,
}

impl TestabilityAnalyzer {
    pub async fn analyze(&self) -> Result<TestabilityResult> {
        let mut issues = Vec::new();
        let mut suggestions = Vec::new();

        // Check 1: Global state usage
        let global_usages = self.find_global_state_usage()?;
        if !global_usages.is_empty() {
            issues.push(format!("Uses global state: {}", global_usages.join(", ")));
            suggestions.push("Inject state as parameters or use dependency injection".to_string());
        }

        // Check 2: Hard-coded dependencies
        let hard_coded = self.find_hard_coded_dependencies()?;
        if !hard_coded.is_empty() {
            issues.push(format!("Hard-coded dependencies: {}", hard_coded.join(", ")));
            suggestions.push("Accept dependencies as constructor/function parameters".to_string());
        }

        // Check 3: Side effects in functions
        let side_effects = self.find_side_effects()?;
        if !side_effects.is_empty() {
            issues.push(format!("Functions with side effects: {}", side_effects.join(", ")));
            suggestions.push("Separate pure logic from side effects".to_string());
        }

        // Check 4: Tight coupling
        let coupling = self.calculate_coupling()?;
        if coupling > 0.7 {
            issues.push("High coupling between components".to_string());
            suggestions.push("Use interfaces/protocols to reduce coupling".to_string());
        }

        // Calculate overall score
        let metrics = TestabilityMetrics {
            coupling_score: coupling,
            cohesion_score: self.calculate_cohesion()?,
            dependency_injection: self.uses_dependency_injection()?,
            pure_function_ratio: self.calculate_pure_function_ratio()?,
            external_dependency_count: hard_coded.len(),
            mockable_dependencies: self.count_mockable_dependencies()?,
        };

        let score = self.calculate_testability_score(&metrics);

        Ok(TestabilityResult {
            testability_score: score,
            issues,
            suggestions,
            metrics,
        })
    }
}
```

### Enhanced Testing State Machine

Replace simple TestGeneration with Intelligence pipeline:

```rust
// Enhanced Testing State Machine

pub enum TestingState {
    // Phase 1: Oracle Synthesis (NEW)
    IntentExtraction,
    ExpectationValidation,
    OracleGeneration,

    // Phase 2: Test Data Generation (NEW)
    InputDomainAnalysis,
    TestDataSynthesis,
    TestDataValidation,

    // Phase 3: Test Generation (Enhanced)
    TestGeneration,         // Now uses oracles + test data

    // Phase 4: Quality Assurance (NEW)
    MutationTesting,
    AssertionAnalysis,
    FlakeDetection,

    // Phase 5: Execution (Enhanced)
    EnvironmentSetup,
    UnitTesting,
    IntegrationTesting,
    ExecutionTracing,       // NEW
    SemanticVerification,   // NEW

    // Phase 6: Results
    CoverageAnalysis,
    FixingIssues,
    Complete,
    Failed,
}

impl TestingStateMachine {
    pub async fn run(&mut self) -> Result<TestingResult> {
        loop {
            match self.current_state {
                // Oracle Synthesis Phase
                TestingState::IntentExtraction => {
                    let extractor = IntentExtraction::new(
                        &self.context.user_intent,
                        &self.context.generated_code,
                        &self.context.gnn_context,
                    );
                    self.context.expectations = extractor.execute(&self.llm).await?;
                    self.transition_to(TestingState::ExpectationValidation);
                }

                TestingState::ExpectationValidation => {
                    let validator = ExpectationValidator::new(
                        &self.context.expectations,
                        &self.context.code_ast,
                        &self.context.gnn,
                    );
                    let result = validator.validate().await?;

                    if !result.is_valid {
                        // Refine expectations and retry
                        self.context.expectations = result.refined_expectations;
                    }

                    self.transition_to(TestingState::OracleGeneration);
                }

                TestingState::OracleGeneration => {
                    let generator = OracleGenerator::new(
                        &self.context.expectations,
                        self.context.language,
                        self.context.test_framework,
                    );
                    self.context.oracles = generator.generate(&self.llm).await?;
                    self.transition_to(TestingState::InputDomainAnalysis);
                }

                // Test Data Phase
                TestingState::InputDomainAnalysis => {
                    let analyzer = InputDomainAnalyzer::new(
                        &self.context.function_signature,
                        &self.context.type_info,
                        &self.context.expectations,
                    );
                    self.context.input_domain = analyzer.analyze().await?;
                    self.transition_to(TestingState::TestDataSynthesis);
                }

                TestingState::TestDataSynthesis => {
                    let synthesizer = TestDataSynthesizer::new(
                        &self.context.input_domain,
                        &self.context.oracles,
                        vec![
                            DataGenerationStrategy::BoundaryValues,
                            DataGenerationStrategy::EquivalencePartitioning,
                            DataGenerationStrategy::PropertyBased { num_samples: 50 },
                            DataGenerationStrategy::HistoricalFailures,
                        ],
                    );
                    self.context.test_data = synthesizer.synthesize().await?;
                    self.transition_to(TestingState::TestDataValidation);
                }

                TestingState::TestDataValidation => {
                    let validator = TestDataValidator::new(
                        &self.context.test_data.test_cases,
                        &self.context.input_domain,
                        &self.context.code_ast,
                    );
                    let result = validator.validate().await?;

                    // Use only valid, non-redundant test cases
                    self.context.test_data.test_cases = result.valid_cases;
                    self.transition_to(TestingState::TestGeneration);
                }

                // Test Generation (Enhanced)
                TestingState::TestGeneration => {
                    let generator = EnhancedTestGenerator::new(
                        &self.context.oracles,
                        &self.context.test_data,
                        self.context.language,
                        self.context.test_framework,
                        QualityRequirements {
                            min_assertion_count: 2,
                            require_error_message: true,
                            require_setup_teardown: true,
                            require_timeout: true,
                            timeout_seconds: 30,
                            require_isolation: true,
                        },
                    );
                    self.context.tests = generator.generate().await?;
                    self.transition_to(TestingState::MutationTesting);
                }

                // Quality Assurance Phase
                TestingState::MutationTesting => {
                    let tester = MutationTester::new(
                        &self.context.generated_code,
                        &self.context.tests,
                        self.context.language,
                        vec![
                            MutationOperator::ArithmeticReplacement,
                            MutationOperator::RelationalReplacement,
                            MutationOperator::OffByOne,
                            MutationOperator::ReturnValueMutation,
                        ],
                    );
                    let result = tester.run().await?;

                    // If mutation score is too low, tests are weak
                    if result.mutation_score < 0.7 {
                        // Add suggested tests to kill surviving mutants
                        for survivor in &result.surviving_mutants {
                            if let Some(suggested) = &survivor.suggested_test {
                                // Parse and add suggested test
                                self.context.tests.push(/* parse suggested test */);
                            }
                        }
                    }

                    self.context.mutation_result = Some(result);
                    self.transition_to(TestingState::AssertionAnalysis);
                }

                TestingState::AssertionAnalysis => {
                    let analyzer = AssertionAnalyzer::new(
                        &self.context.tests,
                        &self.context.code_ast,
                    );
                    let result = analyzer.analyze().await?;

                    // Strengthen weak assertions
                    for weak in &result.weak_assertions {
                        // Update assertion in test
                        self.update_assertion(&weak.test_id, &weak.strengthened_version)?;
                    }

                    // Add missing assertions
                    for missing in &result.missing_assertions {
                        if missing.importance == AssertionImportance::Critical {
                            self.add_assertion(&missing.test_id, &missing.suggested_assertion)?;
                        }
                    }

                    self.transition_to(TestingState::FlakeDetection);
                }

                TestingState::FlakeDetection => {
                    let detector = FlakeDetector::new(
                        &self.context.tests,
                        3,    // repeat count
                        0.2,  // variance threshold (20%)
                    );
                    let result = detector.detect().await?;

                    // Quarantine flaky tests
                    for quarantined in &result.quarantined {
                        self.context.quarantined_tests.push(quarantined.test.test_id.clone());
                    }

                    // Apply suggested fixes
                    for flaky in &result.flaky_tests {
                        for fix in &flaky.suggested_fixes {
                            if fix.confidence > 0.8 {
                                // Apply fix automatically
                                self.apply_flake_fix(&flaky.test_id, fix)?;
                            }
                        }
                    }

                    self.transition_to(TestingState::EnvironmentSetup);
                }

                // Execution Phase
                TestingState::EnvironmentSetup => {
                    // ... existing implementation ...
                    self.transition_to(TestingState::UnitTesting);
                }

                TestingState::UnitTesting => {
                    // Run tests (excluding quarantined)
                    let tests_to_run: Vec<_> = self.context.tests.iter()
                        .filter(|t| !self.context.quarantined_tests.contains(&t.id))
                        .collect();

                    self.context.test_results = self.run_tests(&tests_to_run).await?;

                    // Check for failures
                    let failures: Vec<_> = self.context.test_results.iter()
                        .filter(|r| !r.passed)
                        .collect();

                    if !failures.is_empty() {
                        self.transition_to(TestingState::ExecutionTracing);
                    } else {
                        self.transition_to(TestingState::IntegrationTesting);
                    }
                }

                TestingState::ExecutionTracing => {
                    // Trace failed tests for detailed debugging info
                    for failure in self.context.test_results.iter().filter(|r| !r.passed) {
                        let tracer = ExecutionTracer::new(
                            &self.context.generated_code,
                            self.context.language,
                            TraceDepth::Standard,
                        );

                        let test = self.context.tests.iter()
                            .find(|t| t.id == failure.test_id)
                            .unwrap();

                        let trace = tracer.trace_execution(test).await?;

                        // Format trace for LLM
                        let trace_context = tracer.format_for_llm(&trace);
                        self.context.add_fix_context(trace_context);
                    }

                    self.transition_to(TestingState::FixingIssues);
                }

                TestingState::IntegrationTesting => {
                    // ... existing implementation ...
                    self.transition_to(TestingState::SemanticVerification);
                }

                TestingState::SemanticVerification => {
                    let verifier = SemanticVerifier::new(
                        &self.context.user_intent,
                        &self.context.expectations,
                        &self.context.generated_code,
                        &self.context.test_results,
                    );
                    let result = verifier.verify(&self.llm).await?;

                    // Check if intent is fulfilled
                    if result.overall_fulfillment < 0.8 {
                        // Intent not met - need to fix
                        for unmet in &result.unmet_expectations {
                            if unmet.severity == Severity::Critical {
                                self.context.add_fix_requirement(format!(
                                    "CRITICAL: {} - {}",
                                    unmet.expectation.description,
                                    unmet.reason
                                ));
                            }
                        }
                        self.transition_to(TestingState::FixingIssues);
                    } else {
                        self.transition_to(TestingState::CoverageAnalysis);
                    }
                }

                TestingState::CoverageAnalysis => {
                    // ... existing implementation ...
                    self.transition_to(TestingState::Complete);
                }

                TestingState::FixingIssues => {
                    // Enhanced with execution trace context
                    // ... implementation ...
                }

                TestingState::Complete => {
                    return Ok(self.build_result());
                }

                TestingState::Failed => {
                    return Err(anyhow!("Testing failed after max retries"));
                }
            }
        }
    }
}
```

---

## Performance Targets

| State                 | Target Time | Notes                 |
| --------------------- | ----------- | --------------------- |
| IntentExtraction      | <2s         | Single LLM call       |
| ExpectationValidation | <500ms      | Static analysis + LLM |
| OracleGeneration      | <1s         | Code generation       |
| InputDomainAnalysis   | <200ms      | AST analysis          |
| TestDataSynthesis     | <1s         | Depends on strategy   |
| TestDataValidation    | <300ms      | Constraint checking   |
| TestGeneration        | <2s         | Per test file         |
| MutationTesting       | <30s        | 100 mutants max       |
| AssertionAnalysis     | <500ms      | Static analysis       |
| FlakeDetection        | <60s        | 3x test repeats       |
| ExecutionTracing      | <5s         | Per failed test       |
| SemanticVerification  | <2s         | LLM assessment        |

**Total Pipeline:** 2-5 minutes for typical project

---

## Success Criteria

### Test Oracle Quality

- [ ]
- [ ]
- [ ] Zero false positives (test passes when code is wrong)

### Test Data Quality

- [ ]
- [ ]
- [ ] Zero invalid test inputs generated

### Test Quality Assurance

- [ ]
- [ ] <5% flaky test rate
- [ ] Average assertion strength >0.7

### Execution Intelligence

- [ ] 100% of failures include execution trace
- [ ]

### Semantic Verification

- [ ]
- [ ] Zero critical unmet expectations in released code

---

## Files to Create

```
src-tauri/src/testing/intelligence/
├── mod.rs
├── intent_extraction.rs
├── expectation_validation.rs
├── oracle_generation.rs
├── input_domain.rs
├── test_data_synthesis.rs
├── test_data_validation.rs
├── test_generation.rs
├── mutation_testing.rs
├── assertion_analysis.rs
├── flake_detection.rs
├── execution_tracing.rs
├── semantic_verification.rs
└── testability_analysis.rs
```

---

## Implementation Priority

**Week 1-2:** Oracle synthesis (IntentExtraction, ExpectationValidation, OracleGeneration)
**Week 3-4:** Test data generation (InputDomainAnalysis, TestDataSynthesis, TestDataValidation)
**Week 5-6:** Quality assurance (MutationTesting, AssertionAnalysis, FlakeDetection)
**Week 7-8:** Execution intelligence (ExecutionTracing, SemanticVerification)
**Week 9-10:** Integration and testing

---

_End of Test Intelligence Layer Specification_
