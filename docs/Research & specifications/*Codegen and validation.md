## Tiered Validation Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                   VALIDATION TIERS                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TIER 0: Instant (Every write)                    ~5-10ms       │
│  ├── Tree-sitter syntax check                                    │
│  └── Basic lint (syntax rules only)                              │
│                                                                  │
│  TIER 1: Fast (Every file)                        ~100-500ms    │
│  ├── LSP diagnostics (type errors)                               │
│  └── Full lint check                                             │
│                                                                  │
│  TIER 2: Batched (End of task/subtask)            ~1-5s         │
│  ├── Build/compile check                                         │
│  └── Affected tests only                                         │
│                                                                  │
│  TIER 3: Full (Before commit/deploy)              ~30s-5min     │
│  ├── Full test suite                                             │
│  ├── Security scan                                               │
│  └── E2E tests (if configured)                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## When to Run What

| Trigger              | Tier    | Why                                   |
| -------------------- | ------- | ------------------------------------- |
| **Every file write** | 0       | Catch obvious errors immediately      |
| **File complete**    | 1       | Catch type errors before moving on    |
| **Subtask complete** | 2       | Verify batch of changes work together |
| **Task complete**    | 2-3     | Confidence before showing user        |
| **Before commit**    | 3       | Full safety net                       |
| **Before deploy**    | 3 + E2E | Production safety                     |

---

## Smart Test Selection (Using Your GNN)

Don't run all tests — run the RIGHT tests:

```typescript
interface TestStrategy {
  // Use dependency graph to determine risk
  async selectTests(changedFiles: string[]): Promise<TestSelection> {

    // 1. Get impact from dependency graph
    const impact = await depGraph.getImpact(changedFiles);

    // 2. Categorize risk
    const risk = assessRisk(impact);

    // 3. Select tests based on risk
    switch (risk.level) {
      case 'low':
        // Only direct unit tests for changed files
        return { tests: impact.directTests, estimated: '2-5s' };

      case 'medium':
        // Direct + first-level dependent tests
        return { tests: [...impact.directTests, ...impact.dependentTests], estimated: '5-15s' };

      case 'high':
        // Affected module tests
        return { tests: impact.moduleTests, estimated: '15-60s' };

      case 'critical':
        // Full test suite
        return { tests: 'all', estimated: '1-5min' };
    }
  }
}

function assessRisk(impact: ImpactAnalysis): RiskAssessment {
  // High risk indicators
  const isHighRisk =
    impact.affectedFiles.length > 20 ||
    impact.touchesCore ||           // Core modules
    impact.touchesAuth ||           // Security-related
    impact.touchesDatabase ||       // Data layer
    impact.changesPublicAPI ||      // Breaking change potential
    impact.crossModuleDependents > 5;

  // Low risk indicators
  const isLowRisk =
    impact.affectedFiles.length <= 3 &&
    impact.isLeafNode &&            // No dependents
    impact.hasDirectTests &&        // Tests exist
    !impact.touchesCriticalPath;

  if (isHighRisk) return { level: 'high' };
  if (isLowRisk) return { level: 'low' };
  return { level: 'medium' };
}
```

---

## Deferred/Background Testing

Run expensive tests without blocking:

```typescript
class DeferredTestRunner {
  private queue: TestJob[] = [];
  private running: boolean = false;

  // Queue tests to run in background
  queueTests(files: string[], priority: 'low' | 'normal' | 'high') {
    this.queue.push({ files, priority, queuedAt: Date.now() });
    this.queue.sort((a, b) => priorityScore(b) - priorityScore(a));

    if (!this.running) {
      this.runBackground();
    }
  }

  // Run tests in background, notify on failure
  private async runBackground() {
    this.running = true;

    while (this.queue.length > 0) {
      const job = this.queue.shift()!;

      const result = await runTests(job.files);

      if (!result.success) {
        // Notify agent/user of failure
        this.emit('testFailure', {
          files: job.files,
          failures: result.failures,
          suggestion: 'Tests failed for recent changes. Review needed.',
        });
      }
    }

    this.running = false;
  }
}

// Usage
const testRunner = new DeferredTestRunner();

// Agent writes files with instant validation only
await writeFileWithTier0(path, content);

// Queue tests in background
testRunner.queueTests([path], 'normal');

// Agent continues working...
// If tests fail, agent gets notified and can fix
```

---

## Validation Modes

Let user/agent choose:

```typescript
type ValidationMode = 'fast' | 'balanced' | 'strict';

const validationModes: Record<ValidationMode, ValidationConfig> = {
  fast: {
    onWrite: ['syntax'], // Tier 0 only
    onFileComplete: ['syntax'], // Skip types
    onTaskComplete: ['syntax', 'types'], // Basic check
    onCommit: ['syntax', 'types', 'lint'], // No tests
    description: 'Speed over safety. For exploration/prototyping.',
  },

  balanced: {
    onWrite: ['syntax'], // Tier 0
    onFileComplete: ['syntax', 'types'], // Tier 1
    onTaskComplete: ['syntax', 'types', 'lint', 'affected-tests'], // Tier 2
    onCommit: ['syntax', 'types', 'lint', 'affected-tests'],
    description: 'Default. Good balance of speed and safety.',
  },

  strict: {
    onWrite: ['syntax', 'types'], // Tier 0+1
    onFileComplete: ['syntax', 'types', 'lint'],
    onTaskComplete: ['syntax', 'types', 'lint', 'affected-tests'],
    onCommit: ['syntax', 'types', 'lint', 'all-tests', 'security'], // Tier 3
    description: 'Maximum safety. For production code.',
  },
};
```

---

## Practical Flow

```
User: "Add user authentication"

Agent Plan:
├── Step 1: Create User model
├── Step 2: Create Auth service
├── Step 3: Create Auth routes
├── Step 4: Create Auth middleware
├── Step 5: Create tests
└── Step 6: Update app.ts

Execution:
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Create User model                                    │
│   └── Write user.ts                                          │
│       ├── ✓ Tier 0: Syntax check (3ms)                       │
│       └── ✓ Tier 1: Type check (150ms)                       │
│   (No tests yet for this file)                               │
├─────────────────────────────────────────────────────────────┤
│ Step 2: Create Auth service                                  │
│   └── Write auth.service.ts                                  │
│       ├── ✓ Tier 0: Syntax (3ms)                             │
│       └── ✓ Tier 1: Types (200ms)                            │
│   (Background: queue affected tests)                         │
├─────────────────────────────────────────────────────────────┤
│ Step 3-4: Routes + Middleware                                │
│   └── Similar...                                             │
├─────────────────────────────────────────────────────────────┤
│ Step 5: Create tests                                         │
│   └── Write auth.test.ts                                     │
│       └── ✓ Tier 0+1                                         │
├─────────────────────────────────────────────────────────────┤
│ Step 6: Update app.ts                                        │
│   └── Edit app.ts                                            │
│       └── ✓ Tier 0+1                                         │
├─────────────────────────────────────────────────────────────┤
│ TASK COMPLETE → Run Tier 2                                   │
│   ├── ✓ Build check (1.2s)                                   │
│   ├── ✓ Affected tests (4.3s)                                │
│   │     └── auth.test.ts: 5 tests passed                     │
│   └── Ready to commit                                        │
└─────────────────────────────────────────────────────────────┘

Total validation time: ~6s (not 30s+ for full suite)
```

---

## Summary

| Strategy                   | When                    | Time Cost             |
| -------------------------- | ----------------------- | --------------------- |
| **Tier 0**(Syntax)         | Every write             | ~5ms                  |
| **Tier 1**(Types)          | File complete           | ~100-500ms            |
| **Tier 2**(Affected tests) | Task complete           | ~2-15s                |
| **Tier 3**(Full tests)     | Commit/deploy only      | ~30s-5min             |
| **Background tests**       | Async, non-blocking     | 0ms (deferred)        |
| **Smart selection**        | Use GNN impact analysis | Reduces test set 80%+ |
| **Mode selection**         | User preference         | Flexibility           |

---

## Key Insight

Your **GNN dependency graph** is the secret weapon here:

```
Without GNN: "File changed → run all tests"       → 5 minutes
With GNN:    "File changed → 3 files affected → 2 tests" → 5 seconds
```

This is how you deliver "never breaks" without sacrificing speed.

---

Want me to add this tiered validation strategy to the spec?
