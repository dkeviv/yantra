# Yantra - User Experience Guide

**Version:** MVP 1.0  
**Last Updated:** November 20, 2025  
**Audience:** End Users and Administrators

---

## Overview

This guide explains how to use Yantra from a user perspective, covering all workflows and user interactions.

---

## Getting Started

### Status: 🔴 Not Implemented

### Installation (Planned)

1. **Download Yantra**
   - Visit yantra.dev
   - Download for your platform (macOS, Windows, Linux)
   - Run the installer

2. **First Launch**
   - Open Yantra application
   - You'll see the 3-panel interface
   - Chat panel (left) - where you interact
   - Code viewer (center) - see generated code
   - Browser preview (right) - test UI code

3. **Load Your Project**
   - Click "Open Project" button
   - Select your Python project folder
   - Yantra analyzes your codebase (takes 5-30 seconds)
   - You're ready to start!

---

## Main User Interface

### 3-Panel Layout

```
┌─────────────────────────────────────────────────────────┐
│  YANTRA                                          [–][×] │
├─────────────┬──────────────────────┬───────────────────┤
│             │                      │                   │
│   CHAT      │   CODE VIEWER        │  BROWSER PREVIEW  │
│   (60%)     │   (25%)              │   (15%)          │
│             │                      │                   │
│  Tell me    │  def calculate_      │  [Live Preview]   │
│  what you   │    total(items):     │                   │
│  want to    │    """               │  [Web Page]       │
│  build...   │    Calculate         │                   │
│             │    total price       │  [Interactive]    │
│  [Input]    │    """               │                   │
│             │    return sum(...)   │                   │
│             │                      │                   │
└─────────────┴──────────────────────┴───────────────────┘
```

### Panel Descriptions

**Chat Panel (Left - 60%)**
- Your main interaction point
- Type what you want to build in natural language
- See Yantra's responses and progress
- Review test results and validations

**Code Viewer (Center - 25%)**
- See the code Yantra generates
- Syntax-highlighted Python code
- Line numbers and formatting
- View-only (Yantra manages the code)

**Browser Preview (Right - 15%)**
- Live preview of UI code
- Interactive - you can test functionality
- Auto-refreshes when code changes
- Shows console errors if any

---

## End User Workflows

### Workflow 1: Generate New Feature

**Scenario:** You want to add a new feature to your Python application

**Steps:**

1. **Start Conversation**
   ```
   You: "I need to add user authentication to my app"
   ```

2. **Yantra Asks Clarifying Questions**
   ```
   Yantra: "I can help with that! What authentication method would you prefer?
   - Email/password with JWT tokens
   - OAuth (Google, GitHub, etc.)
   - Both
   
   Also, should I include password reset functionality?"
   ```

3. **You Provide Details**
   ```
   You: "Email/password with JWT, and yes include password reset"
   ```

4. **Yantra Works**
   ```
   Yantra: "🔄 Analyzing your existing codebase..."
   Yantra: "✓ Found your user model at models/user.py"
   Yantra: "✓ Detected you're using Flask"
   Yantra: "🔄 Generating authentication code..."
   Yantra: "🔄 Generating unit tests..."
   Yantra: "🔄 Running tests... (15/15 passed) ✓"
   Yantra: "🔄 Scanning for security issues... (0 critical) ✓"
   Yantra: "✓ Code committed: feat: Add JWT authentication with password reset"
   ```

5. **Review in Code Viewer**
   - See the generated code
   - Review the implementation
   - Check the tests

6. **Done!**
   - Working authentication code is ready
   - All tests pass
   - No security vulnerabilities
   - Committed to Git

**Time:** 2-3 minutes (vs 2-3 hours manually)

---

### Workflow 2: Fix a Bug

**Scenario:** There's a bug in your existing code

**Steps:**

1. **Describe the Bug**
   ```
   You: "The date formatting is wrong in the monthly report. It shows 'MM/DD/YYYY' but we need 'DD/MM/YYYY'"
   ```

2. **Yantra Investigates**
   ```
   Yantra: "🔄 Searching for date formatting in report generation..."
   Yantra: "✓ Found the issue in reports/monthly.py line 47"
   Yantra: "The format string is using %m/%d/%Y instead of %d/%m/%Y"
   Yantra: "Shall I fix this?"
   ```

3. **You Confirm**
   ```
   You: "Yes, fix it"
   ```

4. **Yantra Fixes and Validates**
   ```
   Yantra: "🔄 Updating date format..."
   Yantra: "🔄 Running affected tests... (8/8 passed) ✓"
   Yantra: "🔄 Checking for other date format issues..."
   Yantra: "✓ Found and fixed 2 similar issues in other reports"
   Yantra: "✓ Code committed: fix: Correct date format to DD/MM/YYYY in all reports"
   ```

5. **Done!**
   - Bug fixed
   - Similar issues also fixed
   - Tests confirm it works
   - Changes committed

**Time:** 30 seconds (vs 30-60 minutes manually)

---

### Workflow 3: Add Tests to Existing Code

**Scenario:** You have code without tests

**Steps:**

1. **Request Tests**
   ```
   You: "Generate tests for the payment processing module"
   ```

2. **Yantra Analyzes**
   ```
   Yantra: "🔄 Analyzing payment processing module..."
   Yantra: "Found 5 functions without tests:
   - process_payment()
   - refund_payment()
   - validate_card()
   - calculate_fees()
   - send_receipt()"
   Yantra: "🔄 Generating comprehensive test suite..."
   ```

3. **Yantra Generates Tests**
   ```
   Yantra: "✓ Generated 23 unit tests covering:
   - Successful payment scenarios
   - Failed payment handling
   - Edge cases (zero amount, invalid card, etc.)
   - Refund scenarios
   - Fee calculations
   - Email sending
   
   Coverage: 94%"
   Yantra: "🔄 Running new tests... (23/23 passed) ✓"
   Yantra: "✓ Code committed: test: Add comprehensive tests for payment processing"
   ```

4. **Done!**
   - Complete test coverage
   - All tests pass
   - Committed to Git

**Time:** 1 minute (vs 2-3 hours manually)

---

### Workflow 4: Refactor Code Safely

**Scenario:** You want to refactor code without breaking anything

**Steps:**

1. **Request Refactoring**
   ```
   You: "Refactor the user service class to use dependency injection"
   ```

2. **Yantra Analyzes Impact**
   ```
   Yantra: "🔄 Analyzing dependencies of UserService..."
   Yantra: "Found 12 classes that use UserService:
   - AuthController (3 methods)
   - ProfileController (5 methods)
   - AdminController (2 methods)
   - ... and 9 more
   
   This refactoring will require updating all these classes.
   Shall I proceed?"
   ```

3. **You Confirm**
   ```
   You: "Yes, proceed"
   ```

4. **Yantra Refactors**
   ```
   Yantra: "🔄 Refactoring UserService with dependency injection..."
   Yantra: "🔄 Updating all 12 dependent classes..."
   Yantra: "🔄 Running full test suite... (147/147 passed) ✓"
   Yantra: "🔄 Checking for breaking changes... (0 found) ✓"
   Yantra: "✓ Code committed: refactor: Implement dependency injection in UserService"
   ```

5. **Done!**
   - Code refactored
   - All dependencies updated
   - Zero breaking changes
   - All tests pass

**Time:** 2-3 minutes (vs 4-6 hours manually)

---

### Workflow 5: Understand Existing Code

**Scenario:** You need to understand how something works

**Steps:**

1. **Ask Yantra**
   ```
   You: "How does the order fulfillment process work?"
   ```

2. **Yantra Explains**
   ```
   Yantra: "Here's the order fulfillment flow:

   1. Order placed → OrderController.create_order()
   2. Payment processed → PaymentService.process()
   3. Inventory checked → InventoryService.reserve_items()
   4. Shipment created → ShippingService.create_shipment()
   5. Email sent → NotificationService.send_order_confirmation()
   
   Would you like me to show you any specific part of this flow?"
   ```

3. **Dive Deeper**
   ```
   You: "Show me the payment processing part"
   ```

4. **Yantra Shows Code**
   ```
   Yantra: [Opens payment code in code viewer and highlights relevant sections]
   
   "The payment processing has these steps:
   1. Validates card details
   2. Calls Stripe API
   3. Handles success/failure
   4. Updates database
   5. Returns result
   
   It handles these error cases:
   - Invalid card
   - Insufficient funds
   - Network timeout
   - Stripe API errors"
   ```

5. **Done!**
   - Clear understanding of the code
   - Visual code walkthrough
   - No need to trace through files manually

**Time:** 1-2 minutes (vs 30-60 minutes manually)

---

## Administrator Workflows

### Workflow 1: Configure LLM API Keys

**Status:** 🔴 Not Implemented (Planned for Week 5-6)

**Steps:**

1. Open Settings (⌘, or Ctrl+,)
2. Navigate to "LLM Configuration"
3. Enter API keys:
   - Claude API key (primary)
   - OpenAI API key (secondary/fallback)
4. Test connection
5. Save

---

### Workflow 2: Configure Security Rules

**Status:** 🔴 Not Implemented (Planned for Week 7)

**Steps:**

1. Open Settings → Security
2. Configure Semgrep rules
3. Set vulnerability severity thresholds
4. Enable/disable auto-fix
5. Configure secret scanning patterns
6. Save

---

### Workflow 3: View Project Statistics

**Status:** 🔴 Not Implemented (Planned for Week 8)

**Steps:**

1. Click "Project Stats" button
2. View metrics:
   - Total lines of code
   - Test coverage
   - Security score
   - Code quality metrics
   - GNN graph statistics
3. Export report

---

## Keyboard Shortcuts

### Status: 🔴 Not Implemented (Planned for Week 8)

**Planned Shortcuts:**

| Action | macOS | Windows/Linux |
|--------|-------|---------------|
| New chat | ⌘N | Ctrl+N |
| Open project | ⌘O | Ctrl+O |
| Settings | ⌘, | Ctrl+, |
| Focus chat | ⌘1 | Ctrl+1 |
| Focus code | ⌘2 | Ctrl+2 |
| Focus preview | ⌘3 | Ctrl+3 |
| Run tests | ⌘T | Ctrl+T |
| Commit code | ⌘Enter | Ctrl+Enter |

---

## Tips & Best Practices

### For Best Results:

1. **Be Specific**
   - ✅ "Add JWT authentication with email/password and password reset"
   - ❌ "Add auth"

2. **Provide Context**
   - ✅ "Fix the date format bug in the monthly report to use DD/MM/YYYY"
   - ❌ "Fix bug"

3. **Review Generated Code**
   - Always check the code viewer
   - Understand what Yantra created
   - Learn from the generated code

4. **Trust the Process**
   - Yantra runs comprehensive tests
   - Security scans are automatic
   - Breaking changes are prevented
   - If all checks pass, the code is solid

5. **Iterate**
   - Start with a simple request
   - Refine based on Yantra's output
   - Build complex features incrementally

---

## Common Questions

### Q: Can I edit the generated code?
**A:** Yes, but Yantra manages the code. If you need changes, ask Yantra to modify it.

### Q: What if tests fail?
**A:** Yantra will automatically fix failing tests or report what it couldn't fix.

### Q: What if I don't like the generated code?
**A:** Ask Yantra to change it: "Use a different approach" or "Simplify this"

### Q: Can I undo changes?
**A:** Yes, all changes are committed to Git, so you can revert using Git commands.

### Q: How does Yantra know my codebase?
**A:** The GNN (Graph Neural Network) analyzes all your code and tracks dependencies.

### Q: Is my code sent to the cloud?
**A:** Only when calling LLM APIs (Claude/GPT-4). Your code stays on your machine otherwise.

---

## Troubleshooting

### Issue: Yantra is slow
**Solution:** Check your internet connection (LLM API calls require internet)

### Issue: Generated code doesn't work
**Solution:** This shouldn't happen (tests should catch it). Report as a bug.

### Issue: Can't load project
**Solution:** Ensure the folder contains Python files and is a valid project.

### Issue: API key errors
**Solution:** Check your LLM API keys in Settings → LLM Configuration

---

## What's Next?

Check the [Features.md](Features.md) document to see what features are coming next!

---

**Last Updated:** November 20, 2025  
**Next Update:** After Week 2 (UI Implementation Complete)
