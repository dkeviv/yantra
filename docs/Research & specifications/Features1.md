\*\*

1. Project Initialization: (MVP)

- When app opens should prompt user to show options to new project or open an existing project or clone project from git
- Should Generate architecture BEFORE any code
- Mandatory user review and approval

2. Architecture view and management (MVP)

- Agent to create architecture view from the documentation files , chat or coding files
- Agent should be able to extract Architecture from documentation maintained in the external tools like Notion, Confluence (post MVP), LInear
- Import existing files
- No editing by user directly for MVP or post MVP needed. Changes made through Agent using chat by the user
- When multiple users are working on a project - all should be able to see the same architecture.
- When a new feature/capability is added by user, agent should confirm and update the architecture automatically on confirmation.
- Agent should confirm with user on the final Architecture before moving to the Plan phase
- Agent should be able to flag scaling bottlenecks and single point of failure and identify security vulnerabilities
- Personas to use: Architects, Developer
- Agent should generate ADR appropriately
- Refer to section 1.1 and 1.5

3. Feature view

- Agent to extract Features from documentation files, chat or coding files and create the features in the Feature view
- Agent should be able to extract Features from documentation maintained in the external tools like Notion, Confluence (post MVP), LInear
- Should track completion of the feature accurately
- When multiple users are working on same project, all users should be able to see the same features view
- Personas to use: Product manager

4. Decisions view - Agent to document any decisions for the project based on the chat. Will serve as approval audit view

- When multiple users are working on same project, all users should be able to see the same Decisions view
- For MVP, this will serve as the Approver audit view

5. Dependency view

- Agent to create and maintain different dependencies
  - File to File (code file, documentation files and testing files)
  - Methods/Classes
  - File to package/tools
  - Package to package/tools
  - File to API
  - Maintain at version level

- Agent should do techstack dependency assessment after architecture completion using LLM as the starting point
- Dependency validation and adjustment : Agent should do dry runs in containers based on techstack from llm to determine any conflicts before making updates to the code
- Agent should be able to use websearch to resolve any dependency issues
- Should check for known vulnerabilities for the packages/tools used to determine the right versions.
- Agent should maintain file registry used for dependency view. The file registry should have all the files in the code base and the document files and testing files
- Refer to section 1.2 Techstack validation and 5.2 Two Critical System

6. Changes view - Agent to document the changes in changes view

- When multiple users are working on same project, all users should be able to see the same Changes view

7. Plan view

- Agent to create project Plan with tasks
- This is a persistent project level plan that Agent should methodically finish and track. Agent can add any sub tasks to track the work
- When multiple users are working on same project, all users should be able to see the same Changes view
- Agent should confirm with the user on the milestones and prioritization

8. Code generation:

- Agent should review the coding file entirely to make sure there are no duplications that it might create
- Agent should look for check dependencies and file registry for existing field before creating new files to avoid duplicative work
- Agent to create perfect code without any syntax issues following the tiered validation strategy.

## Tiered Validation Strategy

│ TIER 0: Instant (Every write) ~5-10ms

│ ├── Tree-sitter syntax check

│ └── Basic lint (syntax rules only)

│ TIER 1: Fast (Every file) ~100-500ms

│ ├── LSP diagnostics (type errors)

│ └── Full lint check

│ TIER 2: Batched (End of task/subtask) ~1-5s

│ ├── Build/compile check /Integration tests

│ └── Affected tests only

│ TIER 3: Full (Before commit/deploy) ~30s-5min

│ ├── Full test suite

│ ├── Security scan

│ └── E2E tests (if configured)

## When to Run What

| Trigger | Tier | Why |

| -------------------------- | ------- | ------------------------------------- |

| **Every file write** | 0 | Catch obvious errors immediately |

| **File complete** | 1 | Catch type errors before moving on |

| **Subtask complete** | 2 | Verify batch of changes work together |

| **Task complete** | 2-3 | Confidence before showing user |

| **Before commit** | 3 | Full safety net |

| **Before deploy** | 3 + E2E | Production safety |

9. Version control: Agent to automatically do git management regularly after each task is completed or change is made with descriptive commit messages
10. Agent should focus on shipping features with horizontal slicing strategy after the core architecture work is completed with higher impact features
11. Security Framework: Agent should follow the security framework in Section 5.8 before generating code, while generating code and after.

MVP:

- SAST analysis
- CVE checks on packages (integration with CVE db)
- Find exposed credentials
- Encrypted credentials storage

12. Context Management : Agent should always have the unlimited context of the project not limited by LLM context windows through

- Token aware context management
- Hierachical assembly
- Intelligent compression
- Chunking
- Dependency graph (GNN)
- RAG (not for MVP)

13. Refactoring and hardening: In clean mode (settings configuration), agent should

- regularly check for refactoring opportunities and fix them and also do regular hardening during code gen time.
- Should check for auto rollback if dependency fix leads to cascading issues

14. As a user, I need ability to search

- Features
- Decisions
- Changes
- Methods
- Classses
- Files

Dependency graph will have indexing

15. As a user, I need the ability to search dependencies using files, classes, packages and tools to easily navigate the dependency view
16. Agent/User should be able to find the relevant files with similarly search using the dependency graph with indexing and semantic embeddings

Need **Hybrid Search Capability (Structural + Semantic):**

**1. Structural Dependencies (Exact):**

```rust

// Track precise code relationships

✅ Imports: "UserService imports AuthContext"

✅ Function Calls: "create_user() calls validate_email()"

✅ Inheritance: "UserProfile extends BaseProfile"

✅ Data Flow: "function returns X, passed to Y"

```

**2. Semantic Similarity (Fuzzy):**

```rust

// Each code node can have optional embedding (384-dim vector)

pub struct CodeNode {

    id, name, type, location,        // Structural metadata

    semantic_embedding: Option<Vec<f32>>,  // 384-dim from all-MiniLM-L6-v2

    code_snippet: Option<String>,    // For generating embeddings

    docstring: Option<String>,       // Function/class documentation

}


17.  Yantra Codex (Pair programming: As a Founder, I want to make Yantra faster in code generation, accurate and reduce the cost of code [generation.To](http://generation.to) achieve that Yantra Codex a ML based GNN will be implemented that predict coding/logic patterns and will generate code with Tree sitter for syntax. It will be used in pair programming with Yantra Codex as the junior developer while LLM as the senior developer with learning loop to improve Yantra Codex in time reducing the reliance on LLM.  See Section 5.2 an 5.3

* Yantra Cloud Codex: Learn from crowd. Common coding patterns. Will be implemented with free version later. For paid version it will opt in
* Yantra learns from LLM corrections
  * Updates model weights periodically
  * Tracks confidence score improvements
  *

18. LLM orchestration: As a user, I want the agent to use top LLM models as well as opensource models as per my need to optimize my costs to provide the best agentic experience.

* User should be able to choose the LLM model they want to use
* LLM providers - Claude, Open AI, Gemini, Meta
* LLM/INference API Providers - Openrouter, Claude, Open AI, Gemini, Meta, Groq, Together


19. LLM consulting mode: As a user, I want the ability to resolve coding issues quicker leveraging the agent to have consulting mode between LLMs. After 2 failed turn, the agent should use a consultant LLM to consult. The primary LLM provides all the context and consult with the secondary aka consultant LLM to refine. The consulting will happen until the problem is resolved.

* User should be able to configure the primary and consultant LLM
* IF consultant not selected, in guided mode, should prompt the user for consulting . If user rejects, then it should keep trying
* Use the **Primary LLM itself** to generate the optimal consultation prompt based on context. This ensures the consultation request is well-framed and provides maximum value.
* UI should show the consulting process transparently

**Consultation Flow:**

Attempt 1: Primary LLM generates code

    ↓

    Tests fail

    ↓

Attempt 2: Primary retries with error context

    ↓

    Tests fail again (TRIGGER THRESHOLD)

    ↓

Consultation: Consultant LLM provides insight

    ↓

Attempt 3: Primary regenerates with consultant's insight

    ↓

    ✅ Success or try again


20. Interaction Modes: As a user, I want the option to select between guided and auto [mode.In](http://mode.in) guided moe, Agent should guide me and take actions only after my consent. In auto mode, I consent for Agent to take actions without my consent though want the Agent to still get approval for any architectural change, any change to the original features, at each phase of PDL. Should follow the approval gates.


21.  Smart terminal use: As a user, I want the agent to use terminal intelligently

1. Agent should have full terminal access to execute shell commands and should be able to create multiple terminals
1. Agent MUST check if a terminal has a process running in foreground before using it.
1. Agent MUST check existing open terminals before creating a new one.
1. Agent should intelligently assess if a process will be longer and run it in background with polling to monitor regularly


22. Known Issues and Fixes:Agent should track learnings from  prior issues with fixes and should always have that context so that it doesn’t repeat the same mistake again.

* Should have local (MVP) and cloud to learn from every failure across all users
* Should preserve privacy. Learn from failure patterns.


23. State Machines:Agentic capabilities should be implemented with proper separation of concerns with 4 state machines.

* Code generation
* Testing
* Deployment
* Maintanence


24. As a user want preview in full browser as code is generated. Browser validation should happen in three state machines - code gen , testing and maintanence



25. Browser Integration: Agent should be able to do interaction testing to validate the implementation in browser with browser integration through CDP. Need to have zero touch flow - User touches nothing. User configures nothing. Browser integration just works Refer to .*Browser Integration.md

**First Launch**


Yantra starts

    ↓

Check for Chrome/Chromium/Edge

    ↓

Found → Store path, done

    ↓

Not found → Show "Downloading browser engine..."

    ↓

Download minimal Chromium (~100MB) to app data

    ↓

Done, never ask again



**Every Subsequent Launch**


Yantra starts

    ↓

Browser path already known

    ↓

Ready instantly



**During Development**


Agent generates frontend code

    ↓

Yantra starts dev server

    ↓

Yantra launches Chrome via CDP (hidden from user)

    ↓

Preview appears in Yantra panel

    ↓

Errors flow to agent automatically


Broswer automation features:

* Interactive element selection (P3 Post-MVP)
* Web-socket - bidirectional browser - Yantra communication
* Map browser elements to source code (React DevTools style).
* Right-click menu in browser preview with Replace/Edit/Remove/Duplicate.(P3 Post-MVP)
* Before/After split view, Undo/Redo stack, change history.(P3 Post-MVP)
* Asset Picker Integration (P3 Post-MVP)


### Error Handling & Edge Cases


**Chrome Not Found:**


- Show user-friendly message: "Downloading browser engine..."

- Download Chromium automatically (~100MB, 30-60 seconds)

- Cache for future use (~/.yantra/browser/chromium)

- Fallback: Ask user to install Chrome manually (rare)


**Dev Server Fails to Start:**


- Check for port conflicts (try next port: 3001, 3002...)

- Check for missing dependencies (run `npm install`)

- Show clear error message with fix suggestions

- Allow manual port specification


**CDP Connection Fails:**


- Retry with exponential backoff (3 attempts)

- Show user-friendly error: "Browser preview unavailable, code validation continues"

- Degrade gracefully: Skip browser validation, rely on unit tests


**Browser Crashes:**


- Detect process exit

- Auto-restart browser

- Restore previous state (URL, tab)

- Log crash for debugging

---


### Security Considerations


**Local-Only Communication:**


- WebSocket server binds to 127.0.0.1 (localhost only)

- No external access

- Random port selection (no fixed port conflicts)


**Chrome Sandbox:**


- Chrome runs in sandboxed mode (default)

- No filesystem access beyond project folder

- No network access to Yantra's internal APIs


**User Privacy:**


- No telemetry sent to Yantra servers

- All browser data stays local

- Anonymous crash reports only (opt-in)



26. Package Management: Agent should be able to do full package management using agentic tools


27. Build and Compilation: Agent should be able to build and compile using agentic tools


28. Testing: Agent should be able to do auto testing with

* Unit tests, integration tests and E2E tests using agentic tools.
* Do automatic debugging.
* Do automatic coverage analysis and aim for >90% coverage unless instructed by user
* Should do mock UI testing with browser automation
* Should do parallel test execution
* Check for race conditions when tests fail

* Should run long tests in the background with polling so that the agent is not blocked.


29. Agent should be able to set env with terminal
29. Agent should always create venv for the workspace and always makes sure venv is activate before running any terminal commands.
29.  Agent should do intelligent API monitoring and contract validation


32. Agent should have access to database tools


33.  Agent should be able to do impact assessment based on the dependencies.

* What does X df epend on
* What depends on X
* Should be able to assess chain of dependencies
*  Should be able to generate full project graph
* Detect circular dependencies
* Assess external dependencies (API)
* Identify architectural layers

Agent should be able to provide all of it through the chat panel


34. Data Analysis and visualization: Agent should be able to do data analysis and show visualizations in the chat panel  using agentic tools


35. Command Classification & Execution: Agent should Automatically detect command duration and choose optimal execution pattern


36. Complex Reasoning and decision making:Agent should have should be able to makes complex analysis, reasoning and make decision to drive autonomous development


37. Resources management:Agent should be able to access system resources information like cpu, memory, disk usage and able to do adaptive resource management


38. File system operations: Agent should be able to use all file system operations tools and should be able to read docx, markdown and pdf


39.  Team of agents should be able to work on same project. (Post MVP) See *Team of Agents & merge conflicts


40.  Cascading failure protection: Agent should implement the below to avoid cascading failures leading to corruption of working code.

1. Checkpoint System (Critical - MVP Foundation): Create a checkpoint before **any** modification that can be restored with one click.Below checkpoints should be created

├─ Session Checkpoint (every chat session)

├─ Feature Checkpoint (before each feature implementation)

├─ File Checkpoint (before modifying each file)

└─ Test Checkpoint (before running tests)
Agent should revert to checkpoint based on testing - if errors worsen after 2 failures before trying again without losing context on what was tried to use in consulting mode.

2. Impact Assessment (GNN-Based):  Assess impact using GNN dependencies before making changes. Assess if refactoring is needed and get user’s consent before making broader refactoring change

3. Automated Testing After Changes : Run automated tests after each change to detect impact immediately.

4. After the 1st failure Agent should do Known issues and fixes db search for a fix

5. Agent should initiate LLM Consulting mode after 2 failed tries
6. Agent should do web search (MVP) and RAG (post MVP) to augment LLM consulting


41. Transparency Requirements (Must-Have):

Every long-running command MUST show:

1. **Start:** "Detected long-running command (npm build), executing in background"

2. **During:** Poll every 10s, show progress: "[12s] Still building... (47/150 files)"

3. **Available:** Remind user: "💬 I'm still available! Ask me anything."

4. **Complete:** Report results: "✅ Build completed in 23s!"


42. Should be able to support multiple languages


43. Provide intelligent code autocompletion in the Monaco editor to enhance developer productivity with LLM

* - Multi-line completions
*  Context-aware suggestions
* Function implementation suggestions
*  Docstring generation
* Show function signatures and parameter hints when typing function calls.
* Should fallback to static completion if LLM fails


44. In the chat panel, any file mentioned should be shown distinctly in UIand on click should open in editor


45. Multi user collaboration features

* Shared architecture
* Shared feature
* Shared changes
* Shared plan
* Shared dependency
* Shared usage to avoid conflicts


46. In order to scale, need 4 tiered storage architecture. Refer to 5.10
46. RAG/Vector db for code patterns (Post MVP) will not be used for indexing or semantics


48. Should be able to automatically deploy to Railway for MVP. Other platforms for post MVP


49. Should automatically rollback on deployment faiure


50. As a developer, I want automated bug fixes when errors are detected
    1. Detects runtime errors from logs
    1. Queries Known Issues DB for fixes
    1. Generates patch code
    1. Tests patch automatically
    1. Deploys fix if tests pass


**
```
