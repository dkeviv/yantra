## **Yantra Architecture v4 (Final)**

### **Role Matrix**

| #   | Role           | Model             | Cost (In/Out $/M) | When                |
| --- | -------------- | ----------------- | ----------------- | ------------------- |
| 1   | **Architect**  | Claude Sonnet 4.5 | $3 / $15          | Start of task       |
| 2   | **Executor**   | Qwen3-Coder-480B  | $0.30 / $1.20     | Implementation      |
| 3   | **Consultant** | GPT-5 Codex       | $1.25 / $10       | First failure       |
| 4   | **Swarm**      | Claude Sonnet 4.5 | $3 / $15          | Persistent failures |
| 5   | **Reviewer**   | Claude Sonnet 4.5 | $3 / $15          | After tests pass    |

---

### **Workflow**

```
┌──────────────────────────────────────────────────────────────────┐
│  1. ARCHITECT (Claude)                                           │
│     • Create spec                                                │
│     • Define interfaces                                          │
│     • Enumerate edge cases                                       │
│     • Write test criteria                                        │
└───────────────────────────┬──────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. EXECUTOR (Qwen3)                                             │
│     • Implement from spec                                        │
└───────────────────────────┬──────────────────────────────────────┘
                            ▼
                       ┌─────────┐
                       │  TEST   │
                       └────┬────┘
                      Pass  │  Fail
                       │    │
          ┌────────────┘    └────────────┐
          ▼                              ▼
┌──────────────────┐          ┌──────────────────────────────────┐
│  5. REVIEWER     │          │  3. CONSULTANT (GPT-5 Codex)     │
│     (Claude)     │          │     • Diagnose error             │
│                  │          │     • Suggest quick fix          │
└────────┬─────────┘          └───────────────┬──────────────────┘
         │                                    ▼
         │                    ┌──────────────────────────────────┐
         │                    │  2. EXECUTOR (Qwen3)             │
         │                    │     • Implement fix              │
         │                    └───────────────┬──────────────────┘
         │                                    ▼
         │                               ┌─────────┐
         │                               │  TEST   │
         │                               └────┬────┘
         │                              Pass  │  Fail
         │                               │    │
         │                  ┌────────────┘    └────────────┐
         │                  ▼                              ▼
         │       ┌──────────────────┐      ┌─────────────────────────┐
         │       │  5. REVIEWER     │      │  4. SWARM LOOP          │
         │       │     (Claude)     │      │                         │
         │       └────────┬─────────┘      │  GPT-5: Propose fix     │
         │                │                │         ▼               │
         │                │                │  Claude: Review/refine  │
         │                │                │         ▼               │
         │                │                │  Qwen3: Implement       │
         │                │                │         ▼               │
         │                │                │       TEST              │
         │                │                │      ↙    ↘             │
         │                │                │   Pass    Fail ─────┐   │
         │                │                │    │                │   │
         │                │                └────┼────────────────┘   │
         │                │                     │                    │
         │                │    ┌────────────────┘                    │
         │                │    ▼                                     │
         │                │  ┌──────────────────┐                    │
         │                └──│  5. REVIEWER     │                    │
         │                   │     (Claude)     │                    │
         │                   └────────┬─────────┘                    │
         │                            │                              │
         ▼                            ▼                              │
┌──────────────────────────────────────────────────────────────────┐│
│  REVIEWER OUTPUT                                                 ││
│  • Code review comments                                          ││
│  • Refactoring suggestions                                       ││
│  • Documentation (API docs, README, docstrings)                  ││
│  • Architecture notes                                            ││
└──────────────────────────────────────────────────────────────────┘│
                            ▼                                       │
                        ✅ DONE
```

---

### **Role Responsibilities**

#### **1. Architect (Claude)**

```
Input:  Task description
Output: Implementation spec

Generates:
├── Function signatures
├── Input/output examples
├── Edge cases
├── Error handling requirements
├── Test criteria
└── Acceptance conditions
```

#### **2. Executor (Qwen3)**

```
Input:  Spec (+ optional fix hints)
Output: Working code

Responsibilities:
├── Implement spec exactly
├── Follow coding standards
├── Handle enumerated edge cases
└── Make tests pass
```

#### **3. Consultant (GPT-5 Codex)**

```
Input:  Spec + failed code + error
Output: Diagnosis + fix suggestion

Responsibilities:
├── Quick diagnosis
├── Practical fix (not rewrite)
├── Confidence score
└── Alternative approaches (if low confidence)
```

#### **4. Swarm (Claude + GPT-5)**

```
Input:  Full history of attempts
Output: Consensus fix

Flow:
├── GPT-5 proposes fix
├── Claude reviews/refines
├── Claude has final authority
└── Loop until resolved
```

#### **5. Reviewer (Claude)**

```
Input:  Passing code
Output: Review + documentation

Generates:
├── Code review (quality, security, performance)
├── Refactoring suggestions (optional improvements)
├── API documentation
├── README section
├── Inline docstrings/comments
└── Usage examples
```

---

### **Cost Breakdown**

| Phase                   | Model          | Est. Tokens     | Est. Cost |
| ----------------------- | -------------- | --------------- | --------- |
| Architect               | Claude         | 5K in, 3K out   | ~$0.06    |
| Execute                 | Qwen3          | 10K in, 8K out  | ~$0.01    |
| Consult (if needed)     | GPT-5          | 8K in, 2K out   | ~$0.03    |
| Retry (if needed)       | Qwen3          | 12K in, 8K out  | ~$0.01    |
| Swarm round (if needed) | Claude + GPT-5 | 15K in, 5K out  | ~$0.12    |
| Review + Document       | Claude         | 15K in, 10K out | ~$0.20    |

---

### **Expected Cost per Task**

| Scenario                 | Path                         | Cost   |
| ------------------------ | ---------------------------- | ------ |
| **Happy path**(70%)      | Architect → Execute → Review | ~$0.27 |
| **Consult needed**(20%)  | + Consult → Retry            | ~$0.31 |
| **Swarm 1-2 rounds**(8%) | + Swarm                      | ~$0.55 |
| **Swarm 3+ rounds**(2%)  | Extended swarm               | ~$0.90 |

**Weighted average: ~$0.32/task**

---

### **vs Alternatives**

| Approach      | Cost/Task  | Quality                      |
| ------------- | ---------- | ---------------------------- |
| Claude-only   | ~$1.80     | High                         |
| GPT-5-only    | ~$1.20     | High                         |
| **Yantra v4** | **~$0.32** | High (Claude specs + review) |
| **Savings**   | **~82%**   | —                            |

---

### **Message Protocols**

#### **1. Architect → Executor**

```json
{
  "task_id": "uuid",
  "spec": {
    "description": "Create JWT auth middleware",
    "functions": [
      {
        "name": "validateToken",
        "signature": "(token: string) => User | null",
        "inputs": ["valid JWT", "expired JWT", "malformed"],
        "outputs": ["User object", "null", "null"]
      }
    ],
    "edge_cases": ["empty token", "missing claims"],
    "test_commands": ["npm test auth.test.js"],
    "acceptance": ["All tests pass", "No security warnings"]
  }
}
```

#### **2. Executor → Test Result**

```json
{
  "task_id": "uuid",
  "code": "// implementation...",
  "test_result": {
    "passed": false,
    "error": "JsonWebTokenError: invalid signature",
    "stack": "..."
  }
}
```

#### **3. Consultant Request**

```json
{
  "task_id": "uuid",
  "spec": {...},
  "code": "// failed implementation",
  "error": "JsonWebTokenError: invalid signature",
  "attempts": 1
}
```

#### **4. Consultant Response**

```json
{
  "diagnosis": "Secret key mismatch",
  "fix": "Use process.env.JWT_SECRET consistently",
  "confidence": 0.9,
  "code_hint": "const secret = process.env.JWT_SECRET"
}
```

#### **5. Swarm Exchange**

```json
{
  "round": 1,
  "gpt5_proposal": {
    "diagnosis": "Async timing issue",
    "fix": "Await the verify call"
  },
  "claude_review": {
    "verdict": "partial_agree",
    "refinement": "Also wrap in try-catch for error handling",
    "final_fix": "await verify() wrapped in try-catch"
  }
}
```

#### **6. Review Output**

```json
{
  "task_id": "uuid",
  "review": {
    "quality_score": 8.5,
    "security_issues": [],
    "performance_notes": ["Consider caching decoded tokens"],
    "refactor_suggestions": ["Extract secret to config module"]
  },
  "documentation": {
    "api_doc": "## validateToken\n\nValidates a JWT...",
    "readme_section": "### Authentication\n\nThis module provides...",
    "docstrings": "/** Validates JWT and returns User or null */",
    "usage_example": "const user = await validateToken(req.headers.auth)"
  }
}
```

---

### **Implementation**

```python
MODELS = {
    "architect": "anthropic/claude-sonnet-4.5",
    "executor": "qwen/qwen3-coder-480b-a35b",
    "consultant": "openai/gpt-5-codex",
    "swarm_reviewer": "anthropic/claude-sonnet-4.5",
    "reviewer": "anthropic/claude-sonnet-4.5"
}

async def execute_task(task_description):
    # 1. ARCHITECT
    spec = await call_model("architect", f"Create spec for: {task_description}")

    # 2. EXECUTOR
    code = await call_model("executor", f"Implement: {spec}")
    result = await run_tests(code, spec.test_commands)

    if result.passed:
        # 5. REVIEWER
        review = await call_model("reviewer", f"Review and document: {code}")
        return {"code": code, "review": review}

    # 3. CONSULTANT
    advice = await call_model("consultant", {
        "spec": spec, "code": code, "error": result.error
    })

    # 2. EXECUTOR (retry)
    code = await call_model("executor", f"Implement with fix: {spec}\n{advice}")
    result = await run_tests(code)

    if result.passed:
        # 5. REVIEWER
        review = await call_model("reviewer", f"Review and document: {code}")
        return {"code": code, "review": review}

    # 4. SWARM LOOP
    history = [{"code": code, "error": result.error, "advice": advice}]

    for round in range(MAX_SWARM_ROUNDS):
        # GPT-5 proposes
        proposal = await call_model("consultant", {"history": history})

        # Claude reviews (final say)
        consensus = await call_model("swarm_reviewer", {
            "history": history, "proposal": proposal
        })

        # Qwen implements
        code = await call_model("executor", f"Implement: {spec}\n{consensus}")
        result = await run_tests(code)

        if result.passed:
            # 5. REVIEWER
            review = await call_model("reviewer", f"Review and document: {code}")
            return {"code": code, "review": review}

        history.append({"code": code, "error": result.error, "consensus": consensus})

    return flag_for_human(task_description, history)
```

---

### **Monthly Projection**

| Volume     | Claude-Only | Yantra v4 | Savings      |
| ---------- | ----------- | --------- | ------------ | --- |
| 1K tasks   | $1,800      | $320      | **$1,480**   |     |
| 10K tasks  | $18,000     | $3,200    | **$14,800**  |     |
| 100K tasks | $180,000    | $32,000   | **$148,000** |     |

---

Want me to create the full implementation spec document with all prompts?
