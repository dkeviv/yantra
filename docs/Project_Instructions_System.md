# Project Instructions System - Design Document

**Date:** November 24, 2025  
**Status:** 🔴 Proposed (Not Yet Implemented)  
**Priority:** HIGH - Core Differentiator vs VS Code

## Executive Summary

Yantra should revolutionize how AI follows project-level instructions by treating them as **active, verified, context-aware rules** rather than passive markdown files. This leverages our GNN to ensure instructions are:
1. **Always applied** (injected into relevant contexts)
2. **Automatically verified** (validated after code generation)
3. **Continuously learned** (strengthened based on violations)
4. **Context-specific** (different rules for different code areas)

## The Gap We're Filling

### VS Code Copilot Approach (Current Industry Standard)
```
.github/copilot-instructions.md → Hope AI reads it → No verification
```

**Problems:**
- ❌ No guarantee AI reads the entire file
- ❌ Instructions get lost when context window fills
- ❌ No way to verify compliance
- ❌ Same instructions for all tasks (no context awareness)
- ❌ Manual maintenance required
- ❌ Can't learn from violations

### Yantra Approach (Revolutionary)
```
Project Rules in GNN → Context-aware injection → Validation → Learning Loop
```

**Advantages:**
- ✅ GNN ensures relevant rules ALWAYS included
- ✅ Automatic validation after generation
- ✅ Different rules for different contexts
- ✅ Self-updating based on patterns
- ✅ Violation tracking and prevention
- ✅ Measurable compliance metrics

## Architecture

### 1. Instruction Types in GNN

```rust
// src/gnn/instructions.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InstructionType {
    /// Code style rule (e.g., "Use descriptive names")
    CodeStyle {
        scope: InstructionScope,
        rule: String,
        severity: Severity,
        examples: Vec<String>,
    },
    
    /// Testing requirement (e.g., "100% coverage")
    TestRequirement {
        scope: InstructionScope,
        requirement: String,
        verification: VerificationMethod,
    },
    
    /// Security rule (e.g., "No plaintext passwords")
    SecurityRule {
        scope: InstructionScope,
        rule: String,
        detection_pattern: Option<Regex>,
        auto_fix: Option<String>,
    },
    
    /// Performance target (e.g., "< 100ms response")
    PerformanceTarget {
        scope: InstructionScope,
        metric: String,
        threshold: f64,
        measurement: MeasurementMethod,
    },
    
    /// Architecture pattern (e.g., "Use dependency injection")
    ArchitecturePattern {
        scope: InstructionScope,
        pattern: String,
        rationale: String,
        anti_patterns: Vec<String>,
    },
    
    /// Documentation requirement (e.g., "All public APIs need docstrings")
    DocumentationRule {
        scope: InstructionScope,
        rule: String,
        format: DocumentationFormat,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InstructionScope {
    /// Applies to entire project
    Global,
    /// Applies to specific directory
    Directory(String),
    /// Applies to specific file pattern
    FilePattern(String),
    /// Applies to specific module
    Module(String),
    /// Applies to specific code type (e.g., all API endpoints)
    CodeType(NodeType),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Error,      // MUST be followed (code generation fails if violated)
    Warning,    // SHOULD be followed (warning shown but code accepted)
    Suggestion, // Nice to have (informational only)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectInstruction {
    pub id: String,
    pub instruction_type: InstructionType,
    pub priority: u8,              // 1-10, for context injection ordering
    pub auto_inject: bool,          // Automatically inject into prompts?
    pub auto_verify: bool,          // Automatically verify after generation?
    pub violation_count: usize,     // Track how often violated
    pub compliance_count: usize,    // Track how often followed
    pub last_violated: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
```

### 2. GNN Integration

Add instructions as nodes and edges:

```rust
// Update GNN node types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    // Existing...
    Function,
    Class,
    Variable,
    Import,
    Module,
    
    // New instruction types
    Instruction(InstructionType),
}

// New edge types for instructions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    // Existing...
    Calls,
    Uses,
    Imports,
    Inherits,
    Defines,
    
    // New instruction edges
    AppliesTo,      // Instruction → Code node
    ViolatedBy,     // Instruction → Code node (for tracking)
    RelatedTo,      // Instruction → Instruction (for grouping)
    RequiredBy,     // Code node → Instruction (explicit requirement)
}
```

### 3. Context-Aware Injection

When assembling context for code generation:

```rust
// src/llm/context.rs (enhanced)

pub fn assemble_context_with_instructions(
    engine: &GNNEngine,
    target_node: Option<&str>,
    file_path: Option<&str>,
    max_tokens: usize,
) -> Result<ContextWithInstructions, String> {
    // 1. Assemble code context (existing logic)
    let code_context = assemble_hierarchical_context(
        engine,
        target_node,
        file_path,
        max_tokens * 0.7, // Reserve 30% for instructions
    )?;
    
    // 2. Find applicable instructions
    let instructions = find_applicable_instructions(
        engine,
        target_node,
        file_path,
    )?;
    
    // 3. Prioritize and fit within token budget
    let instruction_context = format_instructions_for_prompt(
        instructions,
        max_tokens * 0.3, // Use 30% of tokens for instructions
    )?;
    
    Ok(ContextWithInstructions {
        code_context,
        instruction_context,
        total_tokens: code_context.total_tokens + instruction_context.len(),
    })
}

fn find_applicable_instructions(
    engine: &GNNEngine,
    target_node: Option<&str>,
    file_path: Option<&str>,
) -> Result<Vec<ProjectInstruction>, String> {
    let mut instructions = Vec::new();
    
    // 1. Get global instructions (always applicable)
    instructions.extend(engine.get_instructions_by_scope(InstructionScope::Global)?);
    
    // 2. Get directory-specific instructions
    if let Some(path) = file_path {
        let dir = Path::new(path).parent().unwrap();
        instructions.extend(
            engine.get_instructions_by_scope(InstructionScope::Directory(dir.to_string()))?
        );
    }
    
    // 3. Get node-type-specific instructions
    if let Some(node_id) = target_node {
        if let Some(node) = engine.find_node(node_id, file_path) {
            instructions.extend(
                engine.get_instructions_by_scope(InstructionScope::CodeType(node.node_type))?
            );
        }
    }
    
    // 4. Sort by priority and violation history
    instructions.sort_by(|a, b| {
        // Higher priority first
        let priority_cmp = b.priority.cmp(&a.priority);
        if priority_cmp != Ordering::Equal {
            return priority_cmp;
        }
        
        // More frequently violated instructions get higher priority
        let violation_ratio_a = a.violation_count as f64 / (a.compliance_count + 1) as f64;
        let violation_ratio_b = b.violation_count as f64 / (b.compliance_count + 1) as f64;
        violation_ratio_b.partial_cmp(&violation_ratio_a).unwrap_or(Ordering::Equal)
    });
    
    Ok(instructions)
}
```

### 4. Validation Layer

After code generation, automatically validate:

```rust
// src/llm/validator.rs (NEW FILE)

pub struct InstructionValidator {
    engine: GNNEngine,
    llm_config: LLMConfig,
}

impl InstructionValidator {
    pub async fn validate_generated_code(
        &self,
        generated_code: &str,
        file_path: &str,
        instructions: &[ProjectInstruction],
    ) -> Result<ValidationReport, String> {
        let mut report = ValidationReport::new();
        
        for instruction in instructions {
            if !instruction.auto_verify {
                continue;
            }
            
            let result = match &instruction.instruction_type {
                InstructionType::CodeStyle { rule, .. } => {
                    self.validate_code_style(generated_code, rule).await?
                }
                InstructionType::SecurityRule { rule, detection_pattern, .. } => {
                    self.validate_security_rule(generated_code, rule, detection_pattern)?
                }
                InstructionType::TestRequirement { requirement, verification, .. } => {
                    self.validate_test_requirement(generated_code, requirement, verification).await?
                }
                InstructionType::DocumentationRule { rule, format, .. } => {
                    self.validate_documentation(generated_code, rule, format)?
                }
                _ => ValidationResult::Skipped,
            };
            
            report.add_result(instruction.id.clone(), result);
        }
        
        Ok(report)
    }
    
    async fn validate_code_style(
        &self,
        code: &str,
        rule: &str,
    ) -> Result<ValidationResult, String> {
        // Use LLM to validate complex style rules
        let prompt = format!(
            "Check if the following code follows this rule: {}\n\nCode:\n{}\n\n\
             Respond with JSON: {{\"compliant\": true/false, \"reason\": \"explanation\"}}",
            rule, code
        );
        
        // Call LLM for validation
        let response = self.call_llm_for_validation(&prompt).await?;
        
        // Parse response
        let validation: StyleValidation = serde_json::from_str(&response)?;
        
        if validation.compliant {
            Ok(ValidationResult::Passed)
        } else {
            Ok(ValidationResult::Failed {
                reason: validation.reason,
                suggestion: None,
            })
        }
    }
    
    fn validate_security_rule(
        &self,
        code: &str,
        rule: &str,
        pattern: &Option<Regex>,
    ) -> Result<ValidationResult, String> {
        // Use regex for pattern-based rules
        if let Some(regex) = pattern {
            if regex.is_match(code) {
                return Ok(ValidationResult::Failed {
                    reason: format!("Code violates security rule: {}", rule),
                    suggestion: Some("Review security guidelines".to_string()),
                });
            }
        }
        
        Ok(ValidationResult::Passed)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub passed: Vec<String>,
    pub failed: Vec<ValidationFailure>,
    pub warnings: Vec<ValidationWarning>,
    pub overall_compliant: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationFailure {
    pub instruction_id: String,
    pub instruction_text: String,
    pub reason: String,
    pub suggestion: Option<String>,
    pub can_auto_fix: bool,
}
```

### 5. Learning Loop

Track violations and strengthen instructions:

```rust
// src/llm/learning.rs (NEW FILE)

pub struct InstructionLearner {
    engine: GNNEngine,
}

impl InstructionLearner {
    /// Update instruction weights based on validation results
    pub fn learn_from_validation(
        &mut self,
        instruction_id: &str,
        validation_result: &ValidationResult,
    ) -> Result<(), String> {
        let mut instruction = self.engine.get_instruction(instruction_id)?;
        
        match validation_result {
            ValidationResult::Passed => {
                instruction.compliance_count += 1;
                // Decrease priority slightly (rule is being followed)
                instruction.priority = instruction.priority.saturating_sub(1);
            }
            ValidationResult::Failed { .. } => {
                instruction.violation_count += 1;
                instruction.last_violated = Some(Utc::now());
                // Increase priority (rule needs emphasis)
                instruction.priority = (instruction.priority + 2).min(10);
            }
            _ => {}
        }
        
        self.engine.update_instruction(instruction)?;
        Ok(())
    }
    
    /// Suggest new instructions based on code patterns
    pub async fn suggest_new_instructions(
        &self,
        llm_config: &LLMConfig,
    ) -> Result<Vec<ProjectInstruction>, String> {
        // Analyze codebase for patterns
        let patterns = self.analyze_codebase_patterns()?;
        
        // Use LLM to suggest instructions
        let prompt = format!(
            "Based on these code patterns, suggest project instructions:\n{}\n\n\
             Respond with JSON array of: {{\"type\": \"...\", \"rule\": \"...\", \"rationale\": \"...\"}}",
            serde_json::to_string_pretty(&patterns)?
        );
        
        // Call LLM
        let response = call_llm_for_suggestions(llm_config, &prompt).await?;
        
        // Parse and create instructions
        let suggestions: Vec<InstructionSuggestion> = serde_json::from_str(&response)?;
        
        Ok(suggestions.into_iter().map(|s| s.into()).collect())
    }
}
```

### 6. User Interface

Add a new panel for managing instructions:

```typescript
// src-ui/components/InstructionsPanel.tsx

export function InstructionsPanel() {
  const [instructions, setInstructions] = createSignal<ProjectInstruction[]>([]);
  const [showAddDialog, setShowAddDialog] = createSignal(false);
  
  // Load instructions
  onMount(async () => {
    const loaded = await invoke<ProjectInstruction[]>('get_project_instructions');
    setInstructions(loaded);
  });
  
  return (
    <div class="instructions-panel">
      <div class="header">
        <h2>Project Instructions</h2>
        <button onClick={() => setShowAddDialog(true)}>
          Add Instruction
        </button>
      </div>
      
      <div class="instructions-list">
        <For each={instructions()}>
          {(instruction) => (
            <InstructionCard
              instruction={instruction}
              onEdit={(updated) => updateInstruction(updated)}
              onDelete={() => deleteInstruction(instruction.id)}
            />
          )}
        </For>
      </div>
      
      {/* Compliance dashboard */}
      <ComplianceMetrics instructions={instructions()} />
      
      {/* Suggested instructions from learning */}
      <SuggestedInstructions />
    </div>
  );
}

function ComplianceMetrics(props: { instructions: ProjectInstruction[] }) {
  const overall = () => {
    const total = props.instructions.reduce(
      (acc, i) => acc + i.compliance_count + i.violation_count,
      0
    );
    const compliant = props.instructions.reduce(
      (acc, i) => acc + i.compliance_count,
      0
    );
    return total > 0 ? (compliant / total * 100).toFixed(1) : 100;
  };
  
  return (
    <div class="compliance-metrics">
      <h3>Compliance Rate: {overall()}%</h3>
      <div class="metrics-grid">
        <For each={props.instructions}>
          {(instruction) => {
            const rate = instruction.compliance_count / 
              (instruction.compliance_count + instruction.violation_count) * 100;
            return (
              <div class="metric-card">
                <span class="rule-name">{instruction.instruction_type}</span>
                <span class="compliance-rate" 
                      style={{ color: rate > 90 ? 'green' : rate > 70 ? 'orange' : 'red' }}>
                  {rate.toFixed(0)}%
                </span>
                <span class="violation-count">
                  {instruction.violation_count} violations
                </span>
              </div>
            );
          }}
        </For>
      </div>
    </div>
  );
}
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 9)
- [ ] Create `src/gnn/instructions.rs` with types
- [ ] Extend GNN to store instruction nodes
- [ ] Add instruction CRUD operations
- [ ] Basic UI for viewing/adding instructions
- [ ] Storage in SQLite (new `instructions` table)

### Phase 2: Context Integration (Week 10)
- [ ] Modify `assemble_context()` to include instructions
- [ ] Implement `find_applicable_instructions()`
- [ ] Token budget management for instructions
- [ ] Format instructions for LLM prompts
- [ ] Test with sample instructions

### Phase 3: Validation (Week 11)
- [ ] Create `src/llm/validator.rs`
- [ ] Implement regex-based validation
- [ ] Implement LLM-based validation
- [ ] Validation report generation
- [ ] UI for viewing validation results

### Phase 4: Learning Loop (Week 12)
- [ ] Track compliance/violation metrics
- [ ] Auto-adjust instruction priorities
- [ ] Suggest new instructions from patterns
- [ ] Compliance dashboard UI
- [ ] Export instructions for sharing

## Example Workflow

### Setup
```typescript
// User adds instruction via UI
await invoke('add_project_instruction', {
  type: 'SecurityRule',
  scope: 'Global',
  rule: 'Never store passwords in plain text. Always use bcrypt or argon2.',
  severity: 'Error',
  detectionPattern: '(password|passwd)\\s*=\\s*["\']', // Regex for detection
  autoInject: true,
  autoVerify: true,
});
```

### Code Generation
```typescript
// User requests: "Add user registration endpoint"
const response = await invoke('generate_code', {
  intent: "Add user registration endpoint with email and password",
  filePath: "src/api/auth.py"
});

// Behind the scenes:
// 1. GNN finds applicable instructions (SecurityRule for auth)
// 2. Context includes: existing auth code + security instructions
// 3. LLM generates code with security awareness
// 4. Validator checks for plaintext passwords
// 5. If found, generation fails with clear error
// 6. User sees: "Code violates security rule: [instruction text]"
```

### Learning
```typescript
// After 100 code generations:
// - SecurityRule followed 95 times, violated 5 times
// - Priority auto-increased from 5 to 7 (needs more emphasis)
// - UI shows 95% compliance rate
// - Violations highlighted in compliance dashboard
```

## Benefits Over VS Code Approach

| Aspect | VS Code (.github/copilot-instructions.md) | Yantra (GNN-Based Instructions) |
|--------|-------------------------------------------|--------------------------------|
| **Enforcement** | Hope AI reads it | Guaranteed injection via GNN |
| **Verification** | None | Automated validation |
| **Context Awareness** | One-size-fits-all | Different rules per context |
| **Learning** | Static | Auto-adjusts based on violations |
| **Visibility** | Hidden in file | Compliance dashboard |
| **Token Efficiency** | Wastes tokens on irrelevant rules | Only relevant rules injected |
| **Measurable** | No metrics | Compliance rate tracked |
| **Evolution** | Manual updates only | Suggests new rules automatically |

## Metrics for Success

- **Compliance Rate**: >90% of generated code follows instructions
- **False Positives**: <5% of validations incorrectly flag violations
- **Context Overhead**: <30% of tokens used for instructions
- **Time to Validate**: <500ms per generation
- **User Adoption**: >80% of users add custom instructions

## Migration Path

For existing `.github/copilot-instructions.md` files:

```typescript
// Auto-import from markdown
await invoke('import_instructions_from_markdown', {
  markdownPath: '.github/copilot-instructions.md'
});

// Behind the scenes:
// 1. Parse markdown file
// 2. Extract rules using LLM
// 3. Classify into instruction types
// 4. Create GNN nodes
// 5. Show user for review/edit
```

## Conclusion

This approach makes Yantra **10x better** than VS Code because:

1. **GNN = Enforcement** - Instructions are structural, not hopeful
2. **Validation = Trust** - Generated code is verified, not assumed correct
3. **Learning = Evolution** - System improves automatically
4. **Context-Aware = Efficiency** - Right rules at right time
5. **Measurable = Accountability** - Compliance is tracked and visible

This is a **fundamental differentiator** that no other tool has. VS Code's markdown file is a band-aid; Yantra's GNN-based system is revolutionary.
