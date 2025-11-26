# Yantra - GitHub Copilot Instructions

## Project Overview

Yantra is an AI-first development platform that generates production-quality code with a revolutionary guarantee: code that never breaks. The platform makes AI the primary developer, with humans providing intent and oversight.

**Core Technology Stack:**
- **Desktop Framework:** Tauri 1.5+ (Rust backend + web frontend)
- **Frontend:** SolidJS 1.8+, Monaco Editor 0.44+, TailwindCSS 3.3+
- **Backend:** Rust with Tokio 1.35+, SQLite 3.44+, petgraph 0.6+ for GNN
- **LLM Integration:** Multi-LLM orchestration (Claude Sonnet 4 + GPT-4 Turbo)
- **Testing:** pytest 7.4+ (Python), Jest (JavaScript)
- **Security:** Semgrep with OWASP rules
- **Browser:** Chrome DevTools Protocol (CDP) via chromiumoxide

## Critical Documentation Requirements

### Mandatory Files to Maintain (Update Immediately After Implementation)

1. **Project_Plan.md** - Track all tasks and their status
2. **Specifications.md** - Detailed requirements for features to be implemented (design specs, UX flows, technical details)
3. **Features.md** - Document all implemented features from user perspective with use cases
4. **UX.md** - Explain user flows (admin and end-user) from user perspective
5. **Technical_Guide.md** - Detailed technical information for developers:
   - How each component was implemented and why
   - Algorithm/methodology details (no code/pseudo-code)
   - References to code files and scripts
   - Workflows and use cases
6. **File_Registry.md** - For all valid files:
   - Purpose of each file
   - What's implemented in it
   - Dependencies and relationships with other files
   - Mark deprecated files with strikethrough
7. **Decision_Log.md** - Capture all design and architecture changes
8. **Session_Handoff.md** - Maintain in `.github` folder for session continuity:
   - Capture full context for session transitions
   - Enable AI assistant to continue seamlessly
   - Essential for context window management
9. **Known_Issues.md** - Track all bugs, issues, and their fixes
10. **Unit_Test_Results.md** - Track unit test results with details and fixes
11. **Integration_Test_Results.md** - Track integration test results with details and fixes
12. **Regression_Test_Results.md** - Track regression test results with details and fixes
13. **Admin_Guide.md** - Comprehensive guide for system administrators:
    - System requirements and installation procedures
    - Configuration management and environment setup
    - Monitoring, logging, and health checks
    - Backup/recovery procedures
    - Regular maintenance tasks and schedules
    - Troubleshooting common issues
    - Performance optimization strategies
    - Security best practices
    - Scaling considerations
    - Upgrade and disaster recovery procedures

## Code Quality Standards

### Rust Code Requirements
- Use Clippy pedantic mode
- Maintain 80%+ test coverage
- NO panics in production code
- Follow Rust best practices and idioms
- Use proper error handling with Result types
- Document public APIs with doc comments

### Frontend Code Requirements
- ESLint strict mode enabled
- Prettier formatting enforced
- TypeScript strict mode
- Component-based architecture with SolidJS
- Proper state management using stores

### Generated Python Code Requirements
- Follow PEP 8 style guide
- Include type hints
- Add comprehensive docstrings
- Implement proper error handling
- Generate unit tests automatically

### Git Workflow
- **Branches:** `main` (production), `develop` (integration), `feature/*` (features)
- **Commits:** Use Conventional Commits format
- **PRs:** Required reviews, CI must pass

## File Management

### Before Creating Any File
1. Check File_Registry.md first to see if file exists
2. Verify purpose and avoid duplicates
3. Update registry after creating new files

### When Updating Files
1. Add/update comments at the top explaining file purpose
2. Review and update comments after completing changes
3. Ensure comments accurately reflect current state
4. Update File_Registry.md with changes

## Testing Requirements (100% Compliance Mandatory)

### Automated Testing Coverage
1. **Unit Tests:** 90%+ code coverage required
2. **Integration Tests:** End-to-end flows must be tested
3. **Performance Tests:** Benchmark critical operations (GNN, LLM calls)
4. **Mock UI Tests:** Test UI components in isolation

### Testing Philosophy
- **100% of tests MUST pass** - No exceptions
- DO NOT change test conditions to make tests pass
- FIX the underlying issues instead
- DO NOT skip tests or defer failures
- DO NOT ask user to accept failing tests

### Performance Targets
- GNN graph build: <5s for 10k LOC (MVP), <30s for 100k LOC (scale)
- GNN incremental update: <50ms per file change
- Dependency lookup: <10ms
- Context assembly: <100ms
- Test execution: <30s for typical project
- Security scan: <10s
- Total cycle (intent → commit): <2 minutes

## Implementation Strategy

### Horizontal Slices Over Vertical Slices
- **Focus:** Ship features faster, not layers
- Implement complete user-facing features incrementally
- Each slice should deliver working functionality
- Prioritize end-to-end value delivery

### Development Phases

#### Phase 1 (Months 1-2): MVP - Code That Never Breaks
**Objectives:**
- Python codebase support
- GNN for code dependencies
- Multi-LLM orchestration
- Automated testing
- Security scanning
- Browser integration
- Git integration via MCP

**Success Metrics:**
- 95%+ generated code passes tests without human intervention
- Zero breaking changes to existing code
- <3% critical security vulnerabilities (auto-fixed)
- Developer NPS >40

#### Phase 2 (Months 3-4): Workflow Foundation
- Workflow execution runtime
- Cron scheduler and webhook triggers
- External API integration (Slack, SendGrid, Stripe)
- Multi-step workflows (3-5 steps)

#### Phase 3 (Months 5-8): Enterprise Automation
- Cross-system dependency tracking
- Browser automation (Playwright)
- Self-healing workflows
- Multi-language support (Python + JavaScript)

#### Phase 4 (Months 9-12): Platform Maturity
- Performance optimization (99.9% uptime)
- Advanced refactoring
- Plugin ecosystem and marketplace
- Enterprise deployment options

## Architecture Principles

### Core Components

1. **User Interface (AI-First)**
   - Chat/Task Interface (primary - 60% screen)
   - Code Viewer (secondary - 25% screen)
   - Browser Preview (live - 15% screen)

2. **Orchestration Layer**
   - Multi-LLM Manager (Claude primary, GPT-4 secondary)
   - Routing and failover logic
   - Cost optimization through smart routing

3. **Intelligence Layer**
   - Graph Neural Network (GNN) for dependencies
   - Vector Database (RAG) for templates and patterns
   - External API tracking
   - Data flow analysis

4. **Validation Layer**
   - Testing engine (pytest/jest)
   - Security scanner (Semgrep)
   - Browser integration (CDP)
   - Dependency validator (GNN)

5. **Integration Layer**
   - Git (MCP Protocol)
   - File system operations
   - External APIs

### GNN Implementation
- Use petgraph for graph operations
- tree-sitter for parsing (Python, JS, etc.)
- SQLite for persistence
- Incremental updates for performance
- Track: functions, classes, imports, calls, data flow

### LLM Integration
- Primary: Claude Sonnet 4
- Secondary: GPT-4 Turbo (validation/fallback)
- Implement rate limiting
- Retry logic with exponential backoff
- Circuit breaker pattern
- Cache responses where possible

## Security & Privacy

### Data Handling
- User code never leaves machine unless explicitly sent to LLM APIs
- Encrypt LLM calls in transit (HTTPS)
- No code storage on Yantra servers
- Anonymous crash reports (opt-in)
- Usage analytics only, no PII (opt-in)

### Security Scanning
- Run Semgrep with OWASP rules
- Check dependencies (Safety for Python, npm audit)
- Scan for secrets using TruffleHog patterns
- Auto-fix critical vulnerabilities when possible

## Error Handling

### Common Coding Mistakes to Avoid

**Document recurring mistakes here as they're discovered:**

1. **GNN Updates:** Always use incremental updates, never rebuild entire graph
2. **LLM Timeouts:** Implement proper timeout handling with fallback to secondary LLM
3. **Browser Automation:** Wait for elements to be ready before interaction
4. **Git Operations:** Always check for conflicts before committing
5. **File Operations:** Use absolute paths, handle permission errors gracefully
6. **Async Operations:** Proper error propagation in Tokio tasks
7. **State Management:** Immutable updates in SolidJS stores

### Error Recovery
- Provide clear error messages to users
- Log errors with full context for debugging
- Implement automatic retry for transient failures
- Fallback to manual intervention when auto-fix fails

## Code Generation Guidelines

### When Generating Code
1. Always check GNN for existing dependencies
2. Generate comprehensive unit tests
3. Include integration tests for external interactions
4. Add security scanning to pipeline
5. Validate in browser if UI-related
6. Commit with descriptive message

### Code Quality Checklist
- [ ] Follows language-specific style guide
- [ ] Includes type hints/annotations
- [ ] Has comprehensive docstrings/comments
- [ ] Implements proper error handling
- [ ] Includes unit tests (90%+ coverage)
- [ ] Passes security scan
- [ ] No breaking changes to existing code
- [ ] Validated in browser (if applicable)

## User Experience Guidelines

### UI/UX Principles
- AI-first interface: Chat is primary interaction
- Code viewer for transparency and learning
- Live preview for immediate feedback
- Clear loading states and progress indicators
- Helpful error messages with suggested fixes
- Keyboard shortcuts for power users

### Response Times
- UI interactions: <100ms
- Code generation: <3s (LLM dependent)
- Full validation cycle: <2 minutes
- Provide progress updates for longer operations

## Development Workflow

### Starting New Feature
1. Create feature branch: `feature/your-feature`
2. Update Project_Plan.md with tasks
3. Implement in horizontal slices
4. Write tests first (TDD encouraged)
5. Run full test suite
6. Update all documentation files
7. Commit with conventional commit message
8. Create PR with detailed description

### Before Committing
1. Run linters: `cargo clippy` and `npm run lint`
2. Run tests: `cargo test` and `npm test`
3. Check coverage: `cargo tarpaulin`
4. Update File_Registry.md
5. Update relevant documentation
6. Review generated code for quality

### Code Review Checklist
- [ ] All tests pass (100%)
- [ ] Documentation updated
- [ ] File_Registry.md updated
- [ ] No security vulnerabilities
- [ ] Performance targets met
- [ ] Code follows style guidelines
- [ ] Breaking changes documented
- [ ] Session_Handoff.md updated if needed

## Continuous Improvement

### Learning from Issues
- Document all recurring issues in this file
- Update Decision_Log.md for architectural changes
- Refine prompts based on LLM output quality
- Optimize performance based on benchmarks
- Gather user feedback and iterate

### Metrics to Track
- Code generation success rate (target: 95%+)
- Test pass rate (target: 100%)
- Security scan results (target: <3% critical)
- Performance benchmarks
- User satisfaction (NPS target: >40)
- Response times
- Error rates

## Communication Guidelines

### With Users
- Be transparent about capabilities and limitations
- Provide clear progress updates
- Explain decisions and tradeoffs
- Ask for clarification when intent is unclear
- Offer multiple solutions when appropriate

### In Documentation
- Use clear, concise language
- Include examples and use cases
- Keep technical accuracy
- Update immediately after changes
- Cross-reference related documents

## Project Structure Reference

```
yantra/
├── src/                    # Rust backend
│   ├── main.rs            # Tauri entry point
│   ├── gnn/               # Graph Neural Network
│   ├── llm/               # LLM orchestration
│   ├── testing/           # Test generation & execution
│   ├── security/          # Security scanning
│   └── git/               # Git integration
├── src-ui/                # Frontend (SolidJS)
│   ├── components/        # UI components
│   ├── stores/            # State management
│   └── App.tsx            # Main app
├── .github/               # GitHub configs and documentation
│   ├── prompts/           # AI prompt templates
│   └── Session_Handoff.md # Session continuity
├── Project_Plan.md        # Task tracking
├── Features.md            # Feature documentation
├── UX.md                  # User flows
├── Technical_Guide.md     # Developer guide
├── File_Registry.md       # File inventory
├── Decision_Log.md        # Design decisions
├── Known_Issues.md        # Bug tracking
├── Unit_Test_Results.md   # Unit test tracking
├── Integration_Test_Results.md  # Integration test tracking
└── Regression_Test_Results.md   # Regression test tracking
```

## Success Criteria

### MVP (Month 2)
- 20 beta users successfully generating code
- >90% of generated code passes tests
- NPS >40

### Month 6
- 10,000 active users
- >95% code success rate
- 50%+ user retention

### Month 12
- 50,000 active users
- Workflow automation live
- 80%+ retention

## Final Reminders

1. **Documentation First:** Update all required docs immediately after implementation
2. **Test Everything:** 100% pass rate is mandatory, no exceptions
3. **Ship Features:** Focus on horizontal slices, deliver complete functionality
4. **Never Break Code:** Use GNN validation, comprehensive testing
5. **Performance Matters:** Meet all performance targets
6. **Security Always:** Scan and fix vulnerabilities automatically
7. **User Experience:** AI-first, transparent, fast, helpful
8. **Context Preservation:** Maintain Session_Handoff.md for continuity

---

*This document should be updated as the project evolves and new patterns/issues are discovered.*
