### Rollback Strategy for Autonomous Deployments

**Core Problem Restated**

The spec mentions "auto-rollback on failure" as a bullet point but provides no mechanism. Autonomous deployment without robust rollback is dangerous: a bad deployment at 3 AM with no human oversight can cause extended outages, data corruption, or customer-facing errors. The complexity of rollback varies dramatically by what was deployed (stateless service vs. database migration vs. infrastructure change).

**Deployment as Immutable Artifact Generation**

Reframe Yantra's role: Yantra should not execute deployments directly. Instead, Yantra generates immutable deployment artifacts (Docker images, Kubernetes manifests, Terraform plans) and delegates execution to proven deployment infrastructure (ArgoCD, Flux, Terraform Cloud, GitHub Actions). This separation provides: audit trail (artifact is versioned in Git), reproducibility (same artifact deploys identically), and integration with existing enterprise deployment pipelines.

**Deployment Manifest Requirements**

When Yantra generates deployment configuration, enforce patterns that enable safe rollback:

**Blue-Green/Canary Structure:** Never generate in-place updates. All service deployments should provision new instances alongside existing ones, shift traffic gradually (canary) or atomically (blue-green), and preserve the previous version for instant rollback. For Kubernetes, this means generating Deployments with proper rolling update strategy or using Argo Rollouts.

**Health Check Contracts:** Every generated deployment must include health check endpoints and startup/readiness/liveness probes. Yantra should generate these checks based on the code it created - if it added a `/payments` endpoint, it should add a corresponding health check that verifies payment service connectivity.

**Rollback Trigger Definitions:** Generate SLO-based rollback triggers as code. For example: "if error rate exceeds 1% for 2 minutes, rollback." These triggers integrate with monitoring systems (Prometheus alerting rules, Datadog monitors) rather than requiring Yantra to stay running and poll.

**Database Migration Safety**

Database changes are the hardest rollback problem. Yantra should enforce expand-contract pattern for all generated migrations:

Phase 1 (Expand): Add new columns/tables without removing old ones. Application code handles both schemas. Phase 2 (Migrate): Backfill data from old to new schema. Phase 3 (Contract): Remove old columns/tables only after successful deployment is confirmed.

Yantra should validate that migrations are backward-compatible by analyzing the generated SQL: no `DROP COLUMN` in the same release as the column is added, no `NOT NULL` constraints without default values, no renaming (use add-new, copy-data, drop-old pattern).

**Autonomous Deployment Loop Design**

When autonomous deployment is implemented (Phase 3+), use this loop:

Step 1 - Generate and validate artifact locally (Yantra's responsibility)
Step 2 - Push artifact to registry, trigger deployment pipeline (Yantra initiates, pipeline executes)
Step 3 - Pipeline deploys to canary (10% traffic)
Step 4 - Monitoring system observes SLOs for configurable window (5-15 minutes)
Step 5 - If SLOs met: pipeline promotes to full traffic. If SLOs breached: pipeline rolls back automatically.
Step 6 - Yantra receives webhook notification of outcome, updates known issues database if rollback occurred

Note that Yantra is not in the critical path of Steps 4-5. The deployment pipeline and monitoring system handle rollback autonomously. This means a Yantra crash mid-deployment doesn't leave the system in an inconsistent state.

**Approval Gates for High-Risk Changes**

Not all changes should deploy autonomously. Yantra should classify change risk and require human approval for high-risk deployments:

Low risk (auto-deploy): CSS changes, copy updates, new endpoints that don't touch existing code
Medium risk (auto-deploy to staging, human approval for prod): Business logic changes, new dependencies
High risk (human approval required): Authentication/authorization changes, database migrations, infrastructure changes, changes to payment/billing code

The risk classifier uses signals from the GNN (how many dependents does this change affect?), file path patterns (anything in `/auth/` or `/billing/` is high-risk), and explicit annotations in code.
