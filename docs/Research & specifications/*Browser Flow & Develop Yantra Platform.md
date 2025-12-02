You're right. Let me clarify the three products.

---

## Three Products, One Platform

**Yantra Core Platform**

```
                         Yantra Core Platform
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
    │  │ Yantra Agent │  │     LLM      │  │  Self-Heal   │   │
    │  │    Engine    │  │  Integration │  │    Engine    │   │
    │  └──────────────┘  └──────────────┘  └──────────────┘   │
    │                                                          │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
    │  │   Workflow   │  │  Credential  │  │   Semantic   │   │
    │  │    Engine    │  │   Manager    │  │   Element    │   │
    │  └──────────────┘  └──────────────┘  └──────────────┘   │
    │                                                          │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
    │  │     CDP      │  │    Event     │  │   Profile    │   │
    │  │   Browser    │  │   Sourcing   │  │   Storage    │   │
    │  └──────────────┘  └──────────────┘  └──────────────┘   │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │              │   │              │   │              │
    │    Yantra    │   │    Yantra    │   │    Yantra    │
    │    Develop   │   │     Flow     │   │   Browser    │
    │              │   │              │   │              │
    │ (Developers) │   │ (Operations) │   │  (Everyone)  │
    │              │   │              │   │              │
    └──────────────┘   └──────────────┘   └──────────────┘
```

---

## Product 1: Yantra Develop

**Audience**

Developers. Technical founders. Engineering teams.

**What It Is**

Agentic Development Platform. AI that writes, tests, deploys code. Human provides intent, AI develops.

**What They See**

```
┌─────────────────────────────────────────────────────────────┐
│ Yantra Develop                                              │
├─────────────┬─────────────────────────────┬─────────────────┤
│             │                             │                 │
│    Chat     │       Code Editor           │    Browser      │
│             │                             │    Preview      │
│   Agent     │       Terminal              │                 │
│             │                             │                 │
└─────────────┴─────────────────────────────┴─────────────────┘
```

**Core Capabilities**

Code generation with GNN dependency awareness. Automated testing and validation. Self-healing code. Multi-agent coordination. Git integration. Deployment automation.

**Use Cases**

Build full applications from description. Fix bugs autonomously. Add features to existing codebase. Refactor without breaking changes.

**Pricing**

Free: Limited generations.
Pro: $49/month.
Team: $99/user/month.
Enterprise: Custom.

---

## Product 2: Yantra Flow

**Audience**

Operations teams. Business analysts. IT automation. DevOps. Anyone building workflows.

**What It Is**

Agentic Integration Platform. Zapier/Make competitor but with AI agent executing flows. Connects discrete tools and automates processes.

**What They See**

```
┌─────────────────────────────────────────────────────────────┐
│ Yantra Flow                                                 │
├─────────────┬───────────────────────────────────────────────┤
│             │                                               │
│             │  ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐   │
│    Chat     │  │Gmail│ →  │Parse│ →  │ CRM │ →  │Slack│   │
│             │  └─────┘    └─────┘    └─────┘    └─────┘   │
│   Agent     │                                               │
│             │  Workflow Canvas                              │
│             │                                               │
├─────────────┼───────────────────────────────────────────────┤
│ Integrations│  Runs: 1,234  │  Success: 98%  │  Active: 12 │
└─────────────┴───────────────────────────────────────────────┘
```

**Core Capabilities**

Natural language workflow creation. Pre-built integrations (APIs, not browser). Event triggers (webhooks, schedules, emails). Data transformation. Error handling with auto-retry. Monitoring and alerting.

**Integrations**

```
Communication:     Slack, Email, Teams, Discord
CRM:               Salesforce, HubSpot, Pipedrive
Databases:         PostgreSQL, MySQL, MongoDB, Airtable
Cloud:             AWS, GCP, Azure
Dev Tools:         GitHub, GitLab, Jira, Linear
Payments:          Stripe, PayPal
Marketing:         Mailchimp, SendGrid, Twilio
Storage:           S3, Google Drive, Dropbox
```

**Use Cases**

```
"When new lead comes into HubSpot, 
 enrich with Clearbit, 
 assign to sales rep based on region,
 notify in Slack"

"When GitHub PR is merged to main,
 run tests,
 deploy to staging,
 notify QA team,
 create Jira ticket for testing"

"Every morning at 9am,
 pull yesterday's sales from Stripe,
 update spreadsheet,
 email summary to leadership"
```

**Pricing**

Free: 100 runs/month, 5 workflows.
Pro: $29/month, 10K runs, unlimited workflows.
Team: $79/month, 100K runs, team features.
Enterprise: Custom.

---

## Product 3: Yantra Browser

**Audience**

Everyone. Knowledge workers. Personal productivity. Anyone who uses a browser.

**What It Is**

Agentic Browser Platform. AI that controls browser for you. For tasks that don't have APIs or integrations.

**What They See**

```
┌─────────────────────────────────────────────────────────────┐
│ ← → ↻  [https://website.com              ]  ☰  │ 🤖 Yantra │
├─────────────────────────────────────────────────────────────┤
│                                               │             │
│                                               │ What can I  │
│              Normal Webpage                   │ help with?  │
│                                               │             │
│                                               │ [Type...]   │
└───────────────────────────────────────────────┴─────────────┘
```

**Core Capabilities**

Natural language browser control. Semantic element finding (no selectors). Self-healing automation. Form filling from profile. Data extraction. Multi-tab workflows. Scheduled tasks.

**Use Cases**

```
"Fill out this job application with my info"

"Book cheapest flight from SF to NYC next Friday"

"Extract all products from this page to spreadsheet"

"Check my bank balance every Monday, 
 text me if below $1000"

"Log into this legacy system and download the report"
```

**Pricing**

Free: 50 actions/day.
Pro: $19/month, unlimited.
Power: $39/month, scheduling, voice.

---

## Key Differences

| Aspect                       | Yantra Develop   | Yantra Flow         | Yantra Browser     |
| ---------------------------- | ---------------- | ------------------- | ------------------ |
| **Audience**           | Developers       | Operations/IT       | Everyone           |
| **Primary Action**     | Generate code    | Connect APIs        | Control browser    |
| **Interface**          | IDE + Chat       | Workflow Canvas     | Browser + Chat     |
| **Integration Method** | Code/Git         | APIs/Webhooks       | Browser automation |
| **Output**             | Working software | Automated workflows | Task completion    |
| **Complexity**         | High             | Medium              | Low                |
| **Technical Skill**    | Developer        | Semi-technical      | None               |

---

## When to Use Which

**Use Yantra Develop When**

Building software. Need code output. Working with codebase. Developer workflow.

**Use Yantra Flow When**

Connecting existing tools. API-based integrations. Recurring automated workflows. Business process automation. Tools have APIs.

**Use Yantra Browser When**

No API available. Legacy systems. Manual web tasks. Personal productivity. One-off tasks. Website interaction.

---

## Overlap and Synergy

**Flow + Browser**

Some workflows need browser for steps without APIs.

```
"When new lead in HubSpot,
 [API] enrich with Clearbit,
 [Browser] check their LinkedIn for recent posts,
 [API] add to Salesforce with LinkedIn insights"
```

Flow orchestrates. Browser handles non-API steps.

**Develop + Flow**

Developers build custom integrations in Develop. Deploy as Flow connectors. Extend Flow capabilities with code.

**Develop + Browser**

Browser testing for generated code. Interactive preview during development. Already designed this integration.

---

## Shared Components Matrix

| Component               | Develop | Flow | Browser |
| ----------------------- | ------- | ---- | ------- |
| Yantra Agent Engine     | ✅      | ✅   | ✅      |
| LLM Integration         | ✅      | ✅   | ✅      |
| Self-Healing            | ✅      | ✅   | ✅      |
| Credential Manager      | ✅      | ✅   | ✅      |
| Workflow Engine         | ⚠️    | ✅   | ⚠️    |
| CDP Browser             | ✅      | ⚠️ | ✅      |
| Semantic Element Finder | ✅      | ❌   | ✅      |
| Event Sourcing          | ✅      | ✅   | ❌      |
| GNN Dependency Graph    | ✅      | ❌   | ❌      |
| Code Editor             | ✅      | ❌   | ❌      |
| Git Integration         | ✅      | ⚠️ | ❌      |
| API Connectors          | ⚠️    | ✅   | ❌      |
| Profile Storage         | ❌      | ⚠️ | ✅      |

✅ = Core feature
⚠️ = Partial/Optional
❌ = Not included

---

## Go-to-Market Sequence

**Option A: Browser First**

```
Month 1-4:   Yantra Browser MVP + Launch
Month 5-8:   Yantra Flow MVP + Launch  
Month 9-12:  Yantra Develop MVP + Launch
```

Rationale: Broadest market first, validates core agent tech, fastest revenue.

**Option B: Flow First**

```
Month 1-4:   Yantra Flow MVP + Launch
Month 5-8:   Yantra Browser MVP + Launch
Month 9-12:  Yantra Develop MVP + Launch
```

Rationale: Clear Zapier competitor, B2B revenue, operations teams pay well.

**Option C: Parallel**

```
Month 1-4:   Core Platform + Browser MVP
Month 5-8:   Flow MVP + Browser growth
Month 9-12:  Develop MVP + Flow growth
```

Rationale: Shared platform investment pays off across all three.

**Recommendation: Option A (Browser First)**

Simplest product. Broadest market. Validates agent tech. Generates revenue. Funds other products.

---

## Revenue Potential (Year 2)

**Yantra Browser**

100K free users → 5K Pro ($19) + 1K Power ($39)
= $1.6M ARR

**Yantra Flow**

20K free users → 2K Pro ($29) + 500 Team ($79)
= $1.2M ARR

**Yantra Develop**

10K free users → 1K Pro ($49) + 200 Team ($99)
= $830K ARR

**Combined: ~$3.6M ARR**

---

## Branding

**Option A: Unified Yantra Brand**

Yantra Browser
Yantra Flow
Yantra Develop

Same brand, clear product names.

**Option B: Separate Brands**

Yantra (Browser - consumer)
YantraFlow (Integration - business)
YantraDev (Development - technical)

Different positioning per audience.

**Recommendation: Option A**

Unified brand builds recognition. Clear product differentiation through name. Cross-selling easier.

---

## Bottom Line

Three products. One platform. Three audiences.

| Product                  | Audience   | Core Value               | Price     |
| ------------------------ | ---------- | ------------------------ | --------- |
| **Yantra Browser** | Everyone   | AI controls your browser | $19-39/mo |
| **Yantra Flow**    | Operations | AI connects your tools   | $29-79/mo |
| **Yantra Develop** | Developers | AI writes your code      | $49-99/mo |

~60% shared core platform. Maximum market coverage. Different UI, same power. Browser first, then Flow, then Develop.
