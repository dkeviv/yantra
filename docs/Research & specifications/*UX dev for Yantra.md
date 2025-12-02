## The Problem

**Agent Can Generate Code**

React components, CSS, layouts - all fine. Functional UI gets created.

**Agent Can't Generate Assets**

Icons - generates placeholder or broken SVG. Images - can't create real images. Illustrations - beyond LLM capability. Photos - obviously can't create. Logos - crude at best.

**Result**

Functionally correct UI. Visually looks like developer prototype. Not production-ready. User has to source assets separately.

---

## Current Workarounds (All Painful)

**Manual Asset Sourcing**

User finds icons on Heroicons, Lucide, etc. User downloads images from Unsplash. User hires designer for illustrations. User integrates manually. Breaks the autonomous flow.

**AI Image Generation**

Use DALL-E, Midjourney, Stable Diffusion. Quality varies wildly. Style inconsistency across assets. Doesn't integrate with code generation. Separate tool, separate workflow.

**Placeholder Forever**

Ship with gray boxes. "We'll add real assets later." Later never comes. Product looks unfinished.

---

## What Yantra Could Do

**Option 1: Integrate Asset Libraries**

Connect to free asset sources directly. Agent selects appropriate assets. Agent integrates into code automatically.

**Option 2: Integrate AI Image Generation**

Connect to DALL-E/Midjourney API. Agent generates assets as needed. Agent maintains style consistency.

**Option 3: Dedicated UX Agent**

Separate agent specialized in visual design. Works alongside code agent. Handles all visual aspects.

**Option 4: Hybrid Approach**

Libraries for standard assets (icons, stock photos). AI generation for custom assets (illustrations, hero images). UX agent for overall visual coherence.

---

## Option 1: Asset Library Integration (For MVP)

**What to Integrate**

Icons:

* Lucide (open source, consistent style)
* Heroicons (Tailwind ecosystem)
* Phosphor Icons (flexible weights)
* Font Awesome (comprehensive)

Photos:

* Unsplash API (free, high quality)
* Pexels API (free, high quality)

Illustrations:

* unDraw (free, customizable colors)
* Storyset (free, animated options)
* Humaaans (free, mix-and-match people)

UI Components:

* shadcn/ui (copy-paste components)
* Radix primitives (accessible base)

**How It Works**

Agent generates: "Need a user profile icon here."
Yantra queries: Lucide API for "user" or "profile."
Agent receives: SVG code or component import.
Agent integrates: Icon appears in generated code.

User sees: Complete UI with real icons.
User does: Nothing.

**Implementation**

```rust
// Asset resolver
async fn resolve_asset(request: AssetRequest) -> Asset {
    match request.type_ {
        AssetType::Icon => {
            // Search Lucide, Heroicons, etc.
            lucide::search(&request.description).await
        }
        AssetType::Photo => {
            // Search Unsplash
            unsplash::search(&request.description).await
        }
        AssetType::Illustration => {
            // Search unDraw
            undraw::search(&request.description).await
        }
    }
}
```

**Pros**

Zero generation time. Consistent quality. Proven assets. Free (mostly). Simple integration.

**Cons**

Limited to what exists in libraries. May not match exact vision. Generic look possible.

---

## Option 2: AI Image Generation Integration

**What to Integrate**

DALL-E 3 - best for consistency, OpenAI API.
Midjourney - highest quality, harder to integrate.
Stable Diffusion - self-hostable, variable quality.
Ideogram - good for text in images.
Recraft - designed for UI assets specifically.

**How It Works**

Agent generates: "Need hero illustration of team collaboration."
Yantra calls: DALL-E API with detailed prompt.
Agent receives: Generated image URL.
Agent integrates: Image downloaded, added to assets, referenced in code.

**Style Consistency Challenge**

Different prompts → different styles.
Solution: Style prefix for all prompts.

```
"In the style of [defined style guide]: [actual request]"
"Minimal flat illustration, pastel colors, no outlines: team collaboration scene"
```

**Project Style Guide**

On project setup, define visual style.
All generated assets follow this guide.
Consistent look across entire app.

**Pros**

Custom assets exactly as needed. Unique visuals. No library limitations.

**Cons**

Generation takes time (10-30 seconds). Quality varies. Cost per generation. Style consistency is hard.

---

## Option 3: Dedicated UX Agent

**Concept**

Separate agent focused only on visual design. Works in parallel with code agent. Handles: color schemes, typography, spacing, assets, visual hierarchy.

**How It Works**

User: "Build a SaaS dashboard for analytics."

Code Agent: Generates functional components.
UX Agent: Defines visual system.
UX Agent: Selects/generates assets.
UX Agent: Reviews generated UI.
UX Agent: Suggests visual improvements.

**UX Agent Capabilities**

Define color palette from brand or generate one.
Select typography pairing.
Create spacing/sizing system.
Source or generate icons.
Source or generate illustrations.
Review screenshots for visual issues.
Suggest layout improvements.

**Communication**

Code Agent: "I need an icon for 'settings'."
UX Agent: Returns appropriate icon with correct sizing/color.

Code Agent: "I generated a dashboard layout."
UX Agent: Reviews, suggests: "Add more whitespace between cards."

**Pros**

Specialized expertise per agent. Better visual outcomes. Parallel work - faster overall.

**Cons**

Complex coordination. Two agents to manage. More LLM costs.

---

## Option 4: Hybrid (Recommended)

**Tiered Approach**

Tier 1 - Library Assets (Default):
Icons → Lucide/Heroicons (instant, free, consistent).
Stock photos → Unsplash API (instant, free, quality).
Illustrations → unDraw (instant, free, customizable).

Tier 2 - AI Generation (When Library Fails):
Custom illustrations → DALL-E/Recraft.
Specific imagery → DALL-E with style guide.
Unique graphics → AI generation.

Tier 3 - UX Agent (For Polish):
Overall visual coherence review.
Color and typography optimization.
Layout and spacing refinement.
Accessibility check.

**Automatic Escalation**

Agent needs icon → Check Lucide → Found → Use it.
Agent needs specific illustration → Check unDraw → Not found → Generate with DALL-E.
UI complete → UX Agent reviews → Suggests refinements.

**User Control**

Settings:

* Asset sources: [x] Lucide [x] Unsplash [x] unDraw
* AI generation: [x] Enabled (uses credits)
* UX review: [x] Enabled

User can disable AI generation to stay free.
User can disable UX review for speed.

---

## Implementation for MVP

**MVP: Library Integration Only**

Integrate Lucide for icons. Integrate Unsplash for photos. Integrate unDraw for illustrations. Agent selects appropriate assets. Zero cost to user. Instant results.

**Post-MVP: Add AI Generation**

DALL-E integration for custom needs. Style guide system for consistency. Fallback when libraries don't have match.

**Phase 3: Add UX Agent**

Visual review of generated UI. Automated suggestions. Optional user-invoked polish pass.

---

## Asset Integration UX

**During Generation**

Agent: "Creating dashboard with analytics charts..."
Agent: "Adding navigation icons..."
→ Lucide icons appear automatically
Agent: "Adding hero image..."
→ Unsplash image appears automatically

User sees complete UI, not placeholders.

**Asset Panel in Yantra**

```
Assets Used
├── Icons: Lucide (12 icons)
├── Photos: Unsplash (3 images)
└── Illustrations: unDraw (1 illustration)

[Swap] [Regenerate] [Upload Custom]
```

User can swap any asset easily.
Click icon → see alternatives → select different one.

**Custom Override**

User doesn't like selected photo.
User clicks photo → "Replace"
Options: Search Unsplash, Generate with AI, Upload custom.
User selects → Asset swapped → Code updated automatically.

---

## Bottom Line

The hybrid approach gives best results:

Library-first for speed and consistency (Lucide, Unsplash, unDraw). AI generation for custom needs (DALL-E). UX Agent for polish (review and refinement).

MVP: Library integration only - zero cost, instant, good enough for 80% of cases.

Post-MVP: Add AI generation and UX Agent for that extra 20%.

User never manually sources assets. User never sees placeholder gray boxes. Complete, visually coherent UI from intent alone.

---

# POST MVP - HYBRID APPROACH

## Hybrid Asset System for Yantra

---

### Overview

**Three-Tier Asset Resolution**

```
User Intent: "Build a dashboard for project management"
                    ↓
         Agent generates UI code
                    ↓
         Asset needs identified
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
 Tier 1          Tier 2          Tier 3
Libraries    AI Generation     UX Agent
 (Free)        (Credits)       (Polish)
```

---

### Tier 1: Asset Libraries (Default, Instant, Free)

**Icons - Lucide (Primary)**

Why: MIT license, consistent style, 1000+ icons, React/Vue/Svelte components, actively maintained.

Integration:

```
Agent needs: "settings icon"
Yantra searches: Lucide API for "settings"
Returns: <Settings className="w-6 h-6" />
Agent integrates: Icon in code, import added
```

Fallbacks: Heroicons, Phosphor, Tabler Icons.

**Photos - Unsplash**

Why: Free API, high quality, proper licensing, huge library.

Integration:

```
Agent needs: "team collaboration photo for hero section"
Yantra searches: Unsplash API for "team collaboration office"
Returns: Image URL + download
Agent integrates: Image saved to /assets, referenced in code
```

Fallbacks: Pexels, Pixabay.

**Illustrations - unDraw**

Why: Free, customizable colors, consistent style, SVG format.

Integration:

```
Agent needs: "empty state illustration for no data"
Yantra searches: unDraw API for "empty" or "no data"
Returns: SVG with project's primary color applied
Agent integrates: SVG in code or as asset file
```

Fallbacks: Storyset, Open Peeps, Humaaans.

**UI Components - shadcn/ui**

Why: Copy-paste components, Tailwind-based, accessible, customizable.

Integration:

```
Agent needs: "date picker component"
Yantra fetches: shadcn/ui date picker
Returns: Component code + dependencies
Agent integrates: Component added to project
```

---

### Tier 2: AI Generation (When Libraries Fail)

**When to Escalate**

Library search returns no match. Library match doesn't fit context. User requests custom/unique asset. Brand-specific imagery needed.

**Image Generation - DALL-E 3 (Primary)**

```
Agent needs: "isometric illustration of AI robot helping developer"
Tier 1 search: No match in unDraw
Escalate to Tier 2:
    ↓
Yantra calls DALL-E API:
    Prompt: "[Style Guide Prefix] isometric illustration of friendly AI robot 
             helping a developer at computer, minimal, soft colors"
    ↓
Returns: Generated image
Agent integrates: Image saved, referenced in code
```

**Style Guide System**

On project creation, define or generate style guide:

```yaml
# .yantra/style-guide.yml
colors:
  primary: "#6366F1"
  secondary: "#EC4899"
  neutral: "#64748B"
  
illustration_style: |
  Minimal flat illustration style, soft pastel colors,
  no harsh outlines, friendly and approachable,
  consistent with modern SaaS aesthetic
  
photo_style: |
  Bright, natural lighting, diverse people,
  modern office or remote work settings,
  authentic not stocky
  
icon_style: "lucide"  # Enforces consistency
```

**All AI Prompts Prefixed**

```
f"{project.style_guide.illustration_style}: {actual_request}"
```

Ensures consistency across all generated assets.

**Generation Providers**

Images/Illustrations: DALL-E 3 (best consistency).
Icons (custom): Recraft (designed for UI).
Logos: Ideogram (handles text well).

**Cost Management**

```
Asset Generation Credits
├── DALL-E 3: ~$0.04 per image
├── Recraft: ~$0.02 per icon
└── Monthly budget: User configurable

[x] Enable AI generation (uses credits)
[ ] Always ask before generating
[x] Prefer libraries when possible
```

---

### Tier 3: UX Agent (Polish Pass)

**What UX Agent Does**

Reviews complete UI for visual coherence. Checks asset consistency. Suggests improvements. Fixes visual issues automatically.

**When It Runs**

After code generation complete. Before presenting to user. Or on-demand: "Polish this UI."

**UX Agent Capabilities**

**Visual Coherence Check:**

```
UX Agent analyzes screenshot:
- Are colors consistent with style guide?
- Do icons all use same style?
- Is spacing consistent?
- Is typography hierarchy clear?
- Are assets appropriate quality?
```

**Asset Consistency:**

```
UX Agent detects:
- Mixed icon styles (Lucide + FontAwesome) → Standardize
- Illustration style mismatch → Replace or regenerate
- Photo quality/style inconsistency → Suggest alternatives
```

**Automated Fixes:**

```
Issue: "Header icon is Heroicons, rest are Lucide"
Fix: Replace header icon with Lucide equivalent
Result: Code updated automatically
```

**Improvement Suggestions:**

```
UX Agent: "The dashboard cards could use more whitespace. 
          The empty state needs an illustration.
          Consider adding subtle shadows for depth."
      
[Apply All] [Review Each] [Skip]
```

**Accessibility Check:**

```
UX Agent detects:
- Low contrast text
- Missing alt text on images
- Icon-only buttons without labels
- Color-only status indicators

Auto-fixes what it can, flags rest for user.
```

---

### Asset Resolution Flow

```
Agent: "I need an icon for user profile"
                ↓
        ┌───────────────┐
        │ Tier 1: Lucide │
        │ Search "user"  │
        └───────┬───────┘
                ↓
         Found match?
        /           \
      Yes            No
       ↓              ↓
   Use Lucide    ┌───────────────┐
   Icon          │ Tier 1: Other │
                 │ Heroicons,    │
                 │ Phosphor, etc │
                 └───────┬───────┘
                         ↓
                  Found match?
                 /           \
               Yes            No
                ↓              ↓
            Use match    ┌───────────────┐
                         │ Tier 2: AI    │
                         │ Generate icon │
                         │ via Recraft   │
                         └───────┬───────┘
                                 ↓
                          Generated
                              ↓
                         Use generated
```

**For Complex Assets (Illustrations, Hero Images):**

```
Agent: "I need hero illustration for onboarding page"
                ↓
        ┌───────────────┐
        │ Tier 1: unDraw │
        │ Search terms   │
        └───────┬───────┘
                ↓
         Good match?
        /           \
      Yes            No
       ↓              ↓
   Use unDraw    ┌───────────────┐
   (recolor to   │ Tier 2: DALL-E│
   match brand)  │ Generate with │
                 │ style guide   │
                 └───────┬───────┘
                         ↓
                    Generated
                         ↓
                ┌───────────────┐
                │ Tier 3: UX    │
                │ Agent review  │
                │ Does it fit?  │
                └───────┬───────┘
                        ↓
                  Approved / Regenerate
```

---

### User Interface

**Asset Panel in Yantra**

```
┌─────────────────────────────────────────────┐
│ Assets                              [Manage] │
├─────────────────────────────────────────────┤
│                                             │
│ Icons (14)                          Lucide  │
│ ┌─────┬─────┬─────┬─────┬─────┐           │
│ │ 👤  │ ⚙️  │ 📊  │ 🔔  │ ... │           │
│ └─────┴─────┴─────┴─────┴─────┘           │
│                                             │
│ Images (3)                        Unsplash  │
│ ┌─────────────┬─────────────┐              │
│ │  [hero]     │  [team]     │              │
│ └─────────────┴─────────────┘              │
│                                             │
│ Illustrations (2)          unDraw + DALL-E  │
│ ┌─────────────┬─────────────┐              │
│ │  [empty]    │  [success]  │  ⭐ Generated │
│ └─────────────┴─────────────┘              │
│                                             │
│ [+ Add Asset]  [🔄 Regenerate All]          │
└─────────────────────────────────────────────┘
```

**Clicking Any Asset:**

```
┌─────────────────────────────────────────────┐
│ Hero Image                          [×]     │
├─────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────┐ │
│ │                                         │ │
│ │         [Current Image]                 │ │
│ │                                         │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ Source: Unsplash                            │
│ Search: "team collaboration modern office"  │
│                                             │
│ Alternatives:                               │
│ ┌─────┬─────┬─────┬─────┐                  │
│ │     │     │     │     │                  │
│ └─────┴─────┴─────┴─────┘                  │
│                                             │
│ [Search Different] [Generate Custom] [Upload]│
└─────────────────────────────────────────────┘
```

**Asset Swap:**

User clicks alternative → Asset swapped → Code updated → Preview refreshes.

One click. Instant update.

---

### Style Guide Setup

**On Project Creation:**

```
┌─────────────────────────────────────────────┐
│ Visual Style                                │
├─────────────────────────────────────────────┤
│                                             │
│ How should your app look?                   │
│                                             │
│ ○ Modern SaaS (clean, minimal, professional)│
│ ○ Playful (rounded, colorful, friendly)     │
│ ○ Corporate (structured, formal, trustworthy│
│ ○ Let AI decide based on my description     │
│ ○ Custom (define your own)                  │
│                                             │
│ Brand Colors (optional):                    │
│ Primary: [#6366F1] Secondary: [#EC4899]     │
│                                             │
│ [Continue]                                  │
└─────────────────────────────────────────────┘
```

**AI-Generated Style Guide:**

User selects "Let AI decide."
User describes: "A project management tool for creative agencies."

Yantra generates:

```yaml
style:
  mood: "Creative, organized, inspiring"
  colors:
    primary: "#8B5CF6"    # Creative purple
    secondary: "#F59E0B"  # Energetic amber
    neutral: "#6B7280"    # Professional gray
  illustration_style: "Playful flat illustrations with bold colors"
  photo_style: "Creative professionals in modern collaborative spaces"
  typography: "Clean sans-serif, friendly but professional"
```

All assets generated will follow this guide.

---

### Implementation Architecture

**Asset Resolver Module:**

```
src-tauri/src/assets/
├── mod.rs              # Main resolver
├── libraries/
│   ├── lucide.rs       # Lucide icon search
│   ├── heroicons.rs    # Heroicons fallback
│   ├── unsplash.rs     # Photo search
│   ├── undraw.rs       # Illustration search
│   └── shadcn.rs       # Component fetching
├── generation/
│   ├── dalle.rs        # DALL-E integration
│   ├── recraft.rs      # Icon generation
│   └── style_guide.rs  # Style guide management
├── ux_agent/
│   ├── reviewer.rs     # Visual coherence check
│   ├── accessibility.rs # A11y validation
│   └── suggestions.rs  # Improvement suggestions
└── resolver.rs         # Tiered resolution logic
```

**Resolution Logic:**

```rust
pub async fn resolve_asset(request: AssetRequest, project: &Project) -> Asset {
    // Tier 1: Libraries
    if let Some(asset) = search_libraries(&request).await {
        if asset.quality_score > 0.7 {
            return asset;
        }
    }
  
    // Tier 2: AI Generation (if enabled)
    if project.settings.ai_generation_enabled {
        let prompt = build_prompt(&request, &project.style_guide);
        let generated = generate_asset(&prompt).await?;
        return generated;
    }
  
    // Fallback: Best library match or placeholder
    search_libraries(&request).await
        .unwrap_or_else(|| Asset::placeholder(&request))
}

pub async fn polish_ui(project: &Project) -> Vec<Suggestion> {
    // Tier 3: UX Agent review
    let screenshot = capture_preview(&project).await;
    let analysis = ux_agent::analyze(screenshot, &project.style_guide).await;
  
    analysis.suggestions
}
```

---

### Cost Structure

**Free (Tier 1 Only):**

Lucide icons - unlimited, free.
Unsplash photos - 50/hour API limit, free.
unDraw illustrations - unlimited, free.
shadcn components - unlimited, free.

**Credits (Tier 2):**

DALL-E 3 image: 2 credits (~$0.04).
Recraft icon: 1 credit (~$0.02).
Style guide generation: 1 credit.

**Included in Plans:**

Free tier: 50 credits/month.
Pro tier: 500 credits/month.
Team tier: 2000 credits/month.

**User Control:**

```
Asset Generation
├── [x] Use free libraries first (recommended)
├── [x] Enable AI generation when needed
├── [ ] Always ask before using credits
└── Credits remaining: 47/50
```

---

### MVP Scope

**MVP (Libraries Only):**

Lucide icon integration. Unsplash photo integration. unDraw illustration integration. Basic asset panel in UI. One-click swap functionality.

**Post-MVP Phase 1 (Add AI Generation):**

DALL-E integration. Style guide system. Credit tracking. Tier escalation logic.

**Post-MVP Phase 2 (Add UX Agent):**

Visual coherence review. Accessibility checking. Automated suggestions. Polish pass on demand.

---

### Bottom Line

Hybrid system gives:

**Speed:** Libraries return instant results for 80% of needs.

**Quality:** AI generation fills gaps with custom assets.

**Consistency:** Style guide ensures visual coherence.

**Polish:** UX Agent catches issues human might miss.

**Cost Control:** Free by default, credits for premium.

User never sees placeholder boxes. User never manually sources assets. User gets production-ready UI from intent alone.
