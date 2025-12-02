## Security Layer: LLM + GNN Risk Assessment

**Core Idea**

Before executing any command or generated code, run it through a multi-stage risk assessment pipeline that combines semantic understanding (LLM) with structural impact analysis (GNN). Abort execution if risk score exceeds threshold. The security boundary becomes intelligent analysis rather than physical isolation.

---

### Why This Could Work Better Than Sandboxing

**Sandboxing Limitations You're Identifying**

Sandboxing treats symptoms, not causes. It asks "how do we limit damage from bad code?" rather than "how do we prevent bad code from running?" It also introduces operational complexity: container runtimes, platform-specific configurations, cold start latency, and debugging difficulty when things fail inside the sandbox. For a desktop app targeting developers on macOS/Windows/Linux, requiring Docker or Firecracker is a significant adoption barrier.

**Proactive vs Reactive Security**

The LLM/GNN approach is proactive - it understands intent and predicts impact before execution. Sandboxing is reactive - it contains damage after malicious code runs. Proactive is philosophically aligned with Yantra's entire value proposition: understand code deeply, prevent problems before they occur.

---

### Multi-Stage Risk Assessment Pipeline

**Stage 1: Static Pattern Detection (Fast, Deterministic)**

Before involving LLM, run fast deterministic checks that catch obvious risks in under 5ms. Flag known dangerous patterns: `eval()`, `exec()`, `__import__()`, `subprocess.call(shell=True)`, `os.system()`, file operations outside workspace (`/etc`, `/usr`, `~/.ssh`), network operations to non-allowlisted hosts, and environment variable access for sensitive keys (`AWS_SECRET`, `API_KEY`). This stage has zero false negatives for known attack patterns and filters out 90% of malicious code before expensive LLM analysis.

**Stage 2: GNN Impact Analysis (Fast, Structural)**

Query the GNN to understand what the code  *touches* . For generated code: what files will be modified, what functions will be called, what external systems will be contacted. For shell commands: parse the command to identify target files and operations. Compute an impact score based on: number of files affected, whether critical files are touched (auth, database, config), whether the change is reversible, and blast radius if something goes wrong. A command that modifies 50 files across 10 modules has higher inherent risk than one that modifies a single test file.

**Stage 3: LLM Semantic Analysis (Slower, Intelligent)**

For code that passes Stages 1-2 but has medium-risk signals, invoke LLM analysis. The prompt structure asks the LLM to analyze code for security risks and respond with a structured assessment covering intent classification (legitimate development task vs suspicious), hidden behaviors (does the code do anything beyond its stated purpose), data exfiltration risks (does it send data to external systems), and persistence mechanisms (does it modify system state in ways that survive the session). The key insight is that LLMs are remarkably good at understanding code intent. A human-written malicious script often has telltale semantic patterns that differ from legitimate automation code.

**Stage 4: Human Approval Gate (For High-Risk Only)**

If the combined risk score exceeds threshold but the code might be legitimate, pause and ask the user. Present the specific risks identified and let them approve or reject. This should trigger rarely (less than 1% of executions) - if it triggers frequently, the earlier stages need tuning.

---

### Risk Scoring Model

**Composite Risk Score Calculation**

Each stage contributes to a weighted risk score from 0.0 (safe) to 1.0 (dangerous).

The static pattern score is binary for known-bad patterns (1.0 if detected, 0.0 otherwise) but contributes 0.3 weight since patterns can have legitimate uses.

The GNN impact score scales from 0.0 to 1.0 based on blast radius. Single file in tests directory scores near 0.0. Multiple files across core modules scores 0.5 or higher. Critical files (auth, config, migrations) add 0.2 to the score. This contributes 0.3 weight.

The LLM semantic score is the model's confidence that the code has malicious or unintended behaviors, contributing 0.4 weight since it catches sophisticated attacks that evade pattern matching.

**Thresholds and Actions**

Score below 0.3: Auto-execute with logging. Score 0.3 to 0.6: Execute with enhanced monitoring and user notification. Score 0.6 to 0.8: Require explicit user confirmation before execution. Score above 0.8: Block execution entirely with explanation.

---

### Addressing the Obvious Counterargument

**"Can't an attacker just craft code that fools the LLM?"**

Yes, adversarial attacks on LLM-based security are possible. But consider the threat model: who is attacking? For Yantra, the primary risks are not sophisticated adversaries crafting adversarial prompts. The risks are: LLM hallucinating dangerous code unintentionally, user accidentally requesting something destructive, and prompt injection through malicious input data.

For these threat categories, LLM analysis is highly effective because the dangerous code isn't intentionally obfuscated. The LLM generated it or the user wrote it without adversarial intent.

For sophisticated adversaries, no security model is perfect. But Yantra isn't a high-value target like a bank or government system. Defense in depth with LLM/GNN analysis plus user confirmation for high-risk actions is appropriate for the threat model.

---

### Practical Implementation

**Performance Budget**

Stage 1 (static patterns): under 5ms. Stage 2 (GNN impact): under 50ms. Stage 3 (LLM analysis): under 2 seconds, but only for 10-20% of executions. Stage 4 (human approval): only for under 1% of executions.

Average overhead for low-risk commands: approximately 55ms. This is acceptable for command execution workflows.

**Caching and Learning**

Cache risk assessments by code hash. If identical code was assessed as safe previously, skip re-analysis. Track false positives (user approved something flagged as risky) and false negatives (something assessed as safe caused problems). Use this feedback to tune thresholds and improve the static pattern database.

**Workspace Boundary Enforcement**

Regardless of risk score, hard-enforce that file operations cannot escape the workspace directory. This is a simple path canonicalization check that doesn't require sandboxing. Any path that resolves outside the workspace after following symlinks is rejected unconditionally.

---

### What This Approach Cannot Catch

**Honest Limitations**

Resource exhaustion (infinite loops, memory bombs) cannot be detected statically with certainty - this is the halting problem. Mitigation: implement timeouts and memory limits at the process level (not full sandboxing, just basic OS-level resource limits).

Time-of-check to time-of-use attacks where code behaves differently during analysis than execution are theoretically possible. Mitigation: the code being analyzed is the exact code being executed, not a representation of it.

Zero-day attack patterns not in the static database and too subtle for LLM detection could slip through. Mitigation: the human approval gate catches high-risk executions, and the feedback loop improves detection over time.

---

### Comparison: LLM/GNN Security vs Sandboxing

**LLM/GNN Security Advantages**

Cross-platform with no Docker or VM dependencies. Proactive prevention rather than reactive containment. Understands intent and context rather than just restricting capabilities. Integrates naturally with Yantra's existing GNN infrastructure. Lower latency for low-risk operations. Provides explanations for why something was blocked.

**Sandboxing Advantages**

Provides defense against unknown attack vectors. Guaranteed containment regardless of analysis accuracy. Industry-standard approach with well-understood properties.

**Hybrid Recommendation**

Use LLM/GNN security as the primary layer for all executions. Offer optional sandboxing for enterprise deployments or users who want defense-in-depth. This gives you the clean cross-platform developer experience by default while allowing security-conscious users to enable additional isolation.


---



**Layer 1: Intent Confirmation**

Before any code generation, summarize what Yantra understands the user wants. User confirms or clarifies. This catches misunderstandings before code is even generated.

**Layer 2: Impact Preview**

After code generation, before execution, show exactly what will happen. Files to be modified (with diffs). Commands to be executed. External APIs to be called. User approves the specific changes, not just the intent.

**Layer 3: Workspace Boundaries**

Hard-enforce that file operations stay within the project directory. This is a simple path check, not sandboxing. Prevents accidental damage to system files regardless of what the LLM generates.

**Layer 4: Checkpoint and Restore**

Before any modification, create a Git stash or equivalent checkpoint. If anything goes wrong, one-click restore to pre-change state. This makes mistakes recoverable rather than trying to prevent all mistakes.

**Layer 5: Retry Limits with Human Checkpoints**

Autonomous retry is limited to 3 attempts. After any failure, show the user what happened and what was tried. Require approval to continue after failed retries.
