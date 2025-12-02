**Master Role: Work Assignment Only**

Master analyzes the feature request, decomposes into sub-features, assigns each sub-feature to an agent. Master doesn't coordinate file locks or approve every operation. Master's job is done after assignment.

**Agent Role: Independent Execution**

Each agent gets a feature assignment, creates their own branch, works independently, completes their feature, creates PR or merge request. Agents coordinate peer-to-peer when they have dependencies.

**Git as the Coordination Layer**

Branches provide isolation - each agent works on their own branch. PRs provide integration - completed work merges through normal Git flow. Conflicts surface at merge time, handled like human team conflicts.

---

## Why This Is Cleaner

**Master Is Not a Single Point of Failure**

Master only acts at the beginning (work assignment). If Master crashes after assignment, agents continue independently. Agents don't need Master's permission for ongoing work.

**Git Already Solves File Coordination**

Branches isolate work. Merges detect conflicts. This is proven infrastructure that handles exactly this problem. No need to reinvent file locking.

**Matches Human Team Workflow**

Tech lead assigns features. Developers create branches, work independently, submit PRs. Team reviews and merges. Yantra mirrors this exactly.

---

## Where Event Sourcing Still Helps

**Tracking Assignment State**

Which features are assigned to which agents. Which agents are active. Which features are completed vs in-progress. This is lightweight coordination state that benefits from event sourcing.

**Dependency Coordination**

Agent B needs an API that Agent A is building. Agent B queries: "Has Agent A completed the user authentication API?" Event log shows: "Agent A completed feature X at timestamp T, commit hash H." Agent B can now pull that branch and continue.

**Recovery After Crashes**

Agent crashes mid-work. When it restarts, it reads event log: "I was assigned feature X, I started at timestamp T, here's my branch name." Agent resumes from where it left off.

---

## Git Branch as Event Log

**Your Insight Is Correct**

A Git branch dedicated to coordination events is a natural fit. Each event is a commit with structured content. Git provides: append-only log semantics, distributed replication for free, history and auditability, and tooling that developers already understand.

**Implementation**

Create `.yantra/coordination` branch that never merges to main. Each coordination event is a commit to this branch:

```
Commit 1: { "event": "feature_assigned", "feature": "user-auth", "agent": "agent-1", "branch": "feature/user-auth" }
Commit 2: { "event": "feature_assigned", "feature": "payment-api", "agent": "agent-2", "branch": "feature/payment-api" }
Commit 3: { "event": "work_started", "agent": "agent-1", "timestamp": "..." }
Commit 4: { "event": "dependency_available", "agent": "agent-1", "api": "UserService.authenticate", "commit": "abc123" }
Commit 5: { "event": "feature_completed", "agent": "agent-1", "branch": "feature/user-auth", "pr": "PR-42" }
```

Agents pull this branch to see current coordination state. Agents push commits when their state changes.

**Why This Is Low-Hanging Fruit**

No new infrastructure. Git is already there. Coordination state is version controlled and auditable. Works locally (single machine) and distributed (team) with same mechanism. Merge conflicts on coordination branch are rare since events append, don't modify.

---

## Refined Architecture

**Master Agent (Work Assignment)**

Receives high-level feature request from user. Uses GNN to decompose into sub-features with minimal dependencies. Assigns each sub-feature to an agent. Commits assignment events to coordination branch. Master's job is done. It can shut down or take new requests.

**Worker Agents (Independent Execution)**

On startup, pull coordination branch, find their assignment. Create feature branch from main. Work independently on their feature. When they need a dependency from another agent, check coordination branch for "dependency_available" events. When their feature is ready for others to use, commit "dependency_available" event. When feature is complete, create PR, commit "feature_completed" event.

**Peer-to-Peer Dependency Resolution**

Agent B needs Agent A's work. Agent B checks coordination branch. If Agent A has committed "dependency_available" for the needed API, Agent B pulls Agent A's branch and continues. If not available yet, Agent B either waits and polls, works on other parts that don't have the dependency, or asks user for guidance.

**Git for Code, Coordination Branch for State**

Code changes happen on feature branches, merge via PRs. Coordination state lives on dedicated branch, append-only events. Clear separation of concerns.

---

## Handling the Master Assignment Problem

**What If Master Assigns Poorly?**

Master might create sub-features with high coupling. Agents then have heavy dependencies and block each other.

**Solutions**

GNN analysis before assignment - Master queries GNN to understand code dependencies, assigns features to minimize cross-agent dependencies. User review of assignments - before agents start, show user the proposed split: "Agent 1 will handle auth, Agent 2 will handle payments. These features share UserService - is this split okay?" Re-assignment capability - if agents discover heavy coupling mid-work, they can request re-assignment through coordination branch.

**Master Is Advisory, Not Authoritative**

Master proposes assignments. Agents and users can adjust. The coordination branch records the current assignments, which can change. This is flexible, not rigid.

---

## Post-MVP Implementation Path

**Phase 2A: Single Machine Multi-Agent**

Multiple Yantra windows on same machine. Local Git repo with coordination branch. Each window is one agent. Simple assignment through a "new feature" flow that checks coordination branch for conflicts.

**Phase 2B: Distributed Team**

Same architecture but Git repo is remote (GitHub, GitLab). Coordination branch syncs through normal push/pull. Agents on different machines coordinate through the same mechanism. No new infrastructure needed - just use remote Git.

**Phase 2C: Shared Coordination Service (Optional)**

If Git branch coordination becomes a bottleneck (many agents, high-frequency events), introduce a lightweight coordination service. But this may never be necessary if event frequency stays low (assignment happens once per feature, not per file).

---

## Why This Works

**Minimal New Infrastructure**

Uses Git you already have. Coordination is just a special branch. No databases, no message queues, no distributed consensus.

**Fault Tolerant by Design**

Agents work independently on their branches. Git branches are isolated. Crashes don't corrupt others' work. Recovery is pull the branch and resume.

**Scales Naturally**

Works with 2 agents on a laptop. Works with 10 agents across a team. Same mechanism, no changes needed.

**Human Understandable**

Developers already understand branches and PRs. Coordination branch is inspectable - you can literally read the event log with `git log`. Debugging is straightforward.


---



## The Current Problem with Git Merges

**Why Merge Conflicts Happen**

Two developers modify the same file without knowing the other was working on it. Changes touch the same lines or tightly coupled code. No visibility into what others are actively changing until PR time.

**Git's Model Is Reactive**

Git detects conflicts after they happen. By then, both developers have invested time in divergent implementations. Resolution requires understanding both change sets, often by someone who wrote neither.

---

## Your Insight: Proactive Conflict Avoidance

**If We Know Dependencies Ahead of Time**

GNN knows: File A depends on Files B, C, D. Function X is called by Functions Y, Z. Changing File A likely requires changes to File B.

**If We Know What Others Are Working On**

Shared coordination shows: Agent 1 is modifying Files A, B. Agent 2 is about to modify File C.

**Combine These**

Before Agent 2 starts on File C, check: Does File C have dependencies on Files A or B that Agent 1 is modifying? If yes, warn Agent 2 or suggest waiting.

---

## Shared Cloud GNN Architecture

**What Gets Shared**

Dependency graph structure - which files/functions depend on which. Not the actual code, just the relationships. Current modification state - which files are being actively modified by which agent. Project-level sharing - all agents working on same project see same graph.

**What Stays Local**

Actual code content. Agent's work-in-progress changes. User's private projects (unless explicitly shared).

**How It Works**

Agent 1 starts working on feature. Agent 1 marks files A, B as "in modification" in shared GNN. Agent 2 wants to modify file C. Agent 2 queries shared GNN: "What are C's dependencies? Are any being modified?" GNN responds: "C calls function in B. B is being modified by Agent 1." Agent 2 sees warning: "File C depends on File B which Agent 1 is currently changing. Recommend: wait for Agent 1, or coordinate approach."

---

## Levels of Conflict Avoidance

**Level 1: Same File Detection**

Simplest case. Two agents want to modify the same file. Shared GNN shows file is already claimed. Second agent is warned immediately, not at merge time.

**Level 2: Direct Dependency Detection**

Agent 1 modifies File A. Agent 2 wants to modify File B which imports from File A. GNN knows this dependency. Agent 2 is warned: "Your file depends on a file being modified. Your changes may need to account for Agent 1's changes."

**Level 3: Transitive Dependency Detection**

Agent 1 modifies File A. File B depends on A. File C depends on B. Agent 2 wants to modify File C. GNN traces the chain and warns about indirect coupling. This catches conflicts that humans routinely miss.

**Level 4: Semantic Dependency Detection**

Agent 1 is changing the signature of function `authenticate()`. GNN knows 47 files call this function. Any agent touching those 47 files gets warned: "Function you're using is being modified. Signature may change."

---

## Implementation Approach

**Shared State Requirements**

File modification registry - who is modifying what, updated in real-time. Dependency graph - which files/functions depend on which, updated on commits. Both must be consistent across all agents in a project.

**Using Your Coordination Branch + Cloud GNN**

Coordination branch (Git) tracks: feature assignments and high-level work allocation. Cloud GNN service tracks: real-time file modification state and dependency queries.

Why split? Coordination branch is low-frequency (feature assignments). File modification state is high-frequency (files claimed/released constantly). Git branch works for low-frequency. Cloud service handles high-frequency queries efficiently.

**Cloud GNN Service Design**

Simple API with four endpoints. Claim file: "Agent X is modifying file Y." Release file: "Agent X finished with file Y." Query dependencies: "What files are affected if I modify file Y?" Query conflicts: "Is anyone modifying files that would conflict with my work on file Y?"

Lightweight service, mostly in-memory with persistence for recovery. Per-project isolation. Could be self-hosted for enterprise or Yantra-hosted for convenience.

---

## Conflict Avoidance Workflow

**Before Agent Starts Work**

Agent receives feature assignment. Agent queries GNN: "I need to modify files A, B, C for this feature." GNN responds: "File B has active modification by Agent 2. Files A, C are clear." Agent sees options: wait for Agent 2 to finish with B, coordinate with Agent 2 on shared approach, proceed with A and C now, handle B later.

**During Agent Work**

Agent claims files as it modifies them. Other agents see claims in real-time. If Agent 2 tries to claim a file Agent 1 holds, immediate warning. No silent conflicts accumulating.

**Before Agent Creates PR**

Agent queries: "Any new dependencies on files I modified since I started?" GNN shows if other completed work now depends on agent's files. Agent can review and ensure compatibility before PR.

---

## Handling Unavoidable Conflicts

**Some Conflicts Are Legitimate**

Two features genuinely need to modify the same code. This isn't a failure - it's a coordination point.

**Early Warning, Not Prevention**

GNN warns: "Both Agent 1 and Agent 2 need to modify UserService.authenticate()." Agents coordinate approach before either starts, not after both finish. Options: one agent handles both changes, agents agree on interface first then implement, sequential execution with dependency.

**Conflict Resolution Assistance**

When conflict does occur, GNN helps resolve it. GNN knows which agent's changes are more extensive. GNN knows which changes have more downstream dependencies. GNN can suggest which change should be base and which should adapt.

---

## Privacy-Preserving Shared GNN

**What Crosses Project Boundaries**

Nothing by default. Each project has isolated GNN. Only team members on same project see shared state.

**Cross-Project Patterns (Optional, Future)**

Anonymized dependency patterns could be shared. "Projects with this structure commonly have conflicts in these areas." Not code, not file names, just structural patterns. Similar to your known issues network effect.

---

## Why This Is Better Than Current Tools

**GitHub/GitLab**

Shows conflicts at PR time, after work is done. No dependency awareness. No real-time modification tracking.

**IDE Collaboration (VS Code Live Share)**

Real-time editing but no dependency intelligence. See cursor positions, not impact analysis.

**Yantra with Shared GNN**

Prevents conflicts before work starts. Understands code dependencies, not just file boundaries. Real-time coordination with intelligence.

---

## Implementation Path

**Post-MVP Phase 2**

Add file claim/release to coordination branch for basic same-file conflict detection. Local GNN queries for dependency analysis. Single machine multi-agent benefits immediately.

**Phase 3**

Introduce lightweight cloud GNN service for team coordination. Real-time sync of modification state. Dependency queries at service level.

**Phase 4**

Advanced conflict prediction using historical patterns. "These two features have 80% chance of conflict based on similar past work." Suggested work ordering to minimize conflicts.

---

## Bottom Line

Your intuition is correct. Shared GNN with real-time modification tracking can prevent merge conflicts rather than just detecting them after the fact. The key insight is combining dependency knowledge (GNN) with activity knowledge (who's modifying what) to warn before conflicts occur.

Git handles code storage and history. Coordination branch handles feature assignment. Cloud GNN handles real-time dependency-aware conflict avoidance. Each layer does what it's best at.

This is genuinely differentiated. No existing tool combines real-time modification tracking with deep dependency understanding to proactively prevent conflicts.
