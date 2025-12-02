## Lint-First Architecture

**Problem:** LLMs generate code → lint → fix → repeat = waste

**Solution:** Lint as a gate, not a step

---

## Three Layers

### 1. Pre-Generation (Context Injection)

```
Claude receives:
├── .eslintrc / ruff.toml / etc.
├── Project's existing patterns (via Codex)
├── "Follow these rules strictly: [extracted rules]"
└── Example snippets from codebase
```

### 2. Streaming Validation (Tree-sitter)

```
Claude outputs token-by-token
         ↓
Tree-sitter parses incrementally
         ↓
Syntax error detected mid-stream?
         ↓
Interrupt + "Fix this: unexpected token at..."
```

### 3. Pre-Delivery Gate

```
Code complete
    ↓
Run linter (ESLint/Ruff/etc.)
    ↓
Errors? ──Yes──→ Auto-fix (deterministic)
    ↓                    ↓
   No              Still errors?
    ↓                    ↓
 Deliver         Re-prompt Claude with errors
                 (max 1 retry, then flag)
```

---

## Implementation

| Stage            | Tool                      | Cost   |
| ---------------- | ------------------------- | ------ |
| Parse config     | Tree-sitter + file read   | $0     |
| Stream validate  | Tree-sitter incremental   | $0     |
| Auto-fix         | ESLint --fix / Ruff --fix | $0     |
| Re-prompt (rare) | Claude                    | $0.018 |

---

## The Key Insight

Don't ask Claude to "write lint-free code."

**Instead:**

1. Tell Claude the rules upfront
2. Catch errors as they stream
3. Auto-fix what's deterministic
4. Only re-prompt for logic issues

**Result:** User never sees lint errors. Ever.

This becomes a Yantra differentiator—"code that works on first paste."
