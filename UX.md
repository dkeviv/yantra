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

### 3-Panel Layout (MVP v1.0)

```
┌──────────────────────────────────────────────────────────────────┐
│  YANTRA                                                   [–][×] │
├─────────────┬──────────────────────┬────────────────────────────┤
│             │                      │                            │
│   CHAT      │   CODE VIEWER        │  BROWSER PREVIEW           │
│   (60%)     │   (25%)              │   (15%)                    │
│             │                      │                            │
│  Tell me    │  def calculate_      │  [Live Preview]            │
│  what you   │    total(items):     │                            │
│  want to    │    """               │  [Web Page]                │
│  build...   │    Calculate         │                            │
│             │    total price       │  [Interactive]             │
│  [Input]    │    """               │                            │
│             │    return sum(...)   │                            │
│             │                      │                            │
└─────────────┴──────────────────────┴────────────────────────────┘
```

### 4-Panel Layout (🆕 Week 9-10: Full Automation)

```
┌──────────────────────────────────────────────────────────────────┐
│  YANTRA                                                   [–][×] │
├─────────────┬──────────────────────┬────────────────────────────┤
│             │                      │                            │
│   CHAT      │   CODE VIEWER        │  BROWSER PREVIEW           │
│   (50%)     │   (30%)              │   (20%)                    │
│             │                      │                            │
│  Tell me    │  def calculate_      │  [Live Preview]            │
│  what you   │    total(items):     │                            │
│  want to    │    """               │  [Web Page]                │
│  build...   │    Calculate         │                            │
│             │    total price       │  [Interactive]             │
│  [Input]    │    """               │                            │
│             │    return sum(...)   │                            │
│             │                      │                            │
├─────────────┴──────────────────────┴────────────────────────────┤
│                                                                  │
│  TERMINAL OUTPUT (Bottom Panel - 30% height)                    │
│  ──────────────────────────────────────────────────────────     │
│  $ pip install -r requirements.txt                              │
│  Collecting flask>=2.0.0                                         │
│    Using cached Flask-2.3.3-py3-none-any.whl (96 kB)           │
│  Installing collected packages: flask, pytest                    │
│  Successfully installed flask-2.3.3 pytest-7.4.0                │
│                                                                  │
│  $ python src/app.py                                            │
│  * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)     │
│  ✅ Execution successful                                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Panel Descriptions (MVP v1.0)

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

### Panel Descriptions (🆕 Full Automation - Week 9-10)

**Chat Panel (Left - 50%)**
- Your main interaction point
- Type what you want to build in natural language
- See Yantra's responses and progress
- Review test results, validations, and execution status

**Code Viewer (Center - 30%)**
- See the code Yantra generates
- Syntax-highlighted Python/JavaScript/Rust code
- Line numbers and formatting
- View-only (Yantra manages the code)
- Jump to errors from terminal output

**Browser Preview (Right - 20%)**
- Live preview of UI code
- Interactive - you can test functionality
- Auto-refreshes when code changes
- Shows console errors if any

**Terminal Output (Bottom - 30% height) 🆕**
- **Real-time streaming output** from all commands
- Shows:
  - Dependency installation (`pip install`, `npm install`)
  - Script execution output (`python app.py`, `node server.js`)
  - Test execution (`pytest`, `jest`)
  - Build processes (`docker build`, `cargo build`)
  - Deployment logs (`aws deploy`, `kubectl apply`)
- **Color-coded output:**
  - Stdout: white text
  - Stderr: yellow/red text
  - Success messages: green ✅
  - Errors: red ❌
- **Features:**
  - Auto-scroll (follow output)
  - Copy button for full output
  - Clear button
  - Search/filter output
  - Timestamp toggle
- **Interactive indicators:**
  - Spinner when command is running
  - Progress bars for long operations
  - Exit code display

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

### Workflow 6: 🆕 Full Autonomous Execution (Week 9-10)

**Scenario:** You want Yantra to not just generate code, but run it, test it, and deploy it automatically

**Steps:**

1. **Request Feature with Execution**
   ```
   You: "Create a REST API for user registration and deploy it to AWS"
   ```

2. **Yantra Generates Code**
   ```
   Yantra: "🔄 Analyzing your project structure..."
   Yantra: "✓ Detected Flask application"
   Yantra: "🔄 Generating REST API endpoints..."
   Yantra: "✓ Created routes/auth.py with registration endpoint"
   Yantra: "✓ Created tests/test_registration.py"
   ```

3. **Yantra Sets Up Environment**
   ```
   [Terminal Output Panel shows:]
   
   $ python -m venv .venv
   Created virtual environment at .venv
   
   $ source .venv/bin/activate
   Activated virtual environment
   
   $ pip install -r requirements.txt
   Collecting flask>=2.0.0
   Collecting pytest>=7.0.0
   Installing collected packages: flask, pytest, flask-jwt-extended
   Successfully installed flask-2.3.3 pytest-7.4.0 flask-jwt-extended-4.5.2
   ✅ Dependencies installed
   ```

4. **Yantra Runs the Code**
   ```
   Yantra: "🔄 Running the application..."
   
   [Terminal Output Panel shows:]
   
   $ python src/app.py
    * Serving Flask app 'app'
    * Debug mode: off
    * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
   
   GET /health - 200 OK
   ✅ Application started successfully
   ```

5. **Yantra Runs Tests**
   ```
   Yantra: "🔄 Running test suite..."
   
   [Terminal Output Panel shows:]
   
   $ pytest tests/ -v
   test_registration.py::test_valid_registration PASSED
   test_registration.py::test_duplicate_email PASSED
   test_registration.py::test_invalid_email PASSED
   test_registration.py::test_weak_password PASSED
   
   ============== 4 passed in 2.43s ==============
   ✅ All tests passed
   ```

6. **Yantra Builds Package**
   ```
   Yantra: "🔄 Building Docker image..."
   
   [Terminal Output Panel shows:]
   
   $ docker build -t user-registration-api:latest .
   [+] Building 45.2s (12/12) FINISHED
    => [1/6] FROM docker.io/library/python:3.11-slim
    => [2/6] WORKDIR /app
    => [3/6] COPY requirements.txt .
    => [4/6] RUN pip install -r requirements.txt
    => [5/6] COPY src/ ./src/
    => [6/6] EXPOSE 5000
    => exporting to image
   ✅ Docker image built successfully
   ```

7. **Yantra Deploys to AWS**
   ```
   Yantra: "🔄 Deploying to AWS ECS..."
   
   [Terminal Output Panel shows:]
   
   $ aws ecr get-login-password | docker login --username AWS --password-stdin
   Login Succeeded
   
   $ docker tag user-registration-api:latest 123456.dkr.ecr.us-east-1.amazonaws.com/user-api:latest
   $ docker push 123456.dkr.ecr.us-east-1.amazonaws.com/user-api:latest
   Pushed: latest
   
   $ aws ecs update-service --cluster prod --service user-api --force-new-deployment
   Service updated successfully
   
   $ aws ecs wait services-stable --cluster prod --services user-api
   ✅ Deployment complete
   
   🚀 API live at: https://api.yourdomain.com/auth/register
   ```

8. **Yantra Monitors**
   ```
   Yantra: "📊 Monitoring deployment..."
   Yantra: "✓ Health check passed (200 OK)"
   Yantra: "✓ No errors in CloudWatch logs"
   Yantra: "✓ Average response time: 45ms"
   Yantra: "✓ Code committed: feat: Add user registration API with AWS deployment"
   ```

9. **Done!**
   - Code generated
   - Environment configured
   - Dependencies installed
   - Application tested locally
   - Docker image built
   - Deployed to AWS
   - Health checks verified
   - Monitoring active
   - All committed to Git

**Time:** 5-10 minutes (vs 4-6 hours manually)

**What You See:**
- Chat panel: High-level progress updates
- Code viewer: Generated code
- Browser preview: API documentation (auto-generated)
- **Terminal output:** Real-time streaming of every command execution

---

### Workflow 7: 🆕 Debug Running Application (Week 9-10)

**Scenario:** The application is running but you see an error

**Steps:**

1. **Error Appears in Terminal**
   ```
   [Terminal Output Panel shows:]
   
   $ python src/app.py
   Traceback (most recent call last):
     File "src/app.py", line 12, in <module>
       from config import DATABASE_URL
   ImportError: cannot import name 'DATABASE_URL' from 'config'
   ❌ Execution failed (exit code: 1)
   ```

2. **Yantra Detects and Analyzes**
   ```
   Yantra: "❌ Runtime error detected"
   Yantra: "Error: ImportError - DATABASE_URL not found in config"
   Yantra: "🔄 Analyzing config.py..."
   Yantra: "Root cause: DATABASE_URL is not defined in config.py"
   ```

3. **Yantra Fixes Automatically**
   ```
   Yantra: "🔄 Adding DATABASE_URL to config.py..."
   Yantra: "✓ Added with default value for local development"
   Yantra: "🔄 Retrying execution..."
   
   [Terminal Output Panel shows:]
   
   $ python src/app.py
    * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
   ✅ Execution successful
   ```

4. **Done!**
   - Error automatically detected
   - Root cause identified
   - Fix applied
   - Re-executed successfully
   - No human intervention needed

**Time:** 30 seconds (fully automatic)

---

### Workflow 8: 🆕 Run Tests from UI (Week 9-10)

**Scenario:** You want to manually trigger tests

**Steps:**

1. **Click "Run Tests" Button** (or press ⌘T)

2. **Tests Execute**
   ```
   [Terminal Output Panel shows:]
   
   $ pytest tests/ -v --cov=src --cov-report=term-missing
   
   tests/test_auth.py::test_login PASSED                    [ 16%]
   tests/test_auth.py::test_logout PASSED                   [ 33%]
   tests/test_users.py::test_create_user PASSED             [ 50%]
   tests/test_users.py::test_get_user PASSED                [ 66%]
   tests/test_users.py::test_update_user PASSED             [ 83%]
   tests/test_users.py::test_delete_user PASSED             [100%]
   
   ----------- coverage: platform darwin, python 3.11.5 -----------
   Name                    Stmts   Miss  Cover   Missing
   -----------------------------------------------------
   src/app.py                 45      0   100%
   src/routes/auth.py         67      2    97%   142-143
   src/routes/users.py        89      0   100%
   src/models/user.py         34      0   100%
   -----------------------------------------------------
   TOTAL                     235      2    99%
   
   ============== 6 passed in 3.45s ==============
   ✅ All tests passed - 99% coverage
   ```

3. **Review Results**
   - Green checkmarks for passing tests
   - Coverage report inline
   - Click on missed lines to jump to code
   - Yantra suggests: "Would you like me to add tests for the 2 uncovered lines?"

**Time:** 3-5 seconds (instant feedback)

---

### Workflow 9: 🆕 Install Dependencies On-Demand (Week 9-10)

**Scenario:** Code needs a new library

**Steps:**

1. **Yantra Detects Missing Dependency**
   ```
   [Terminal Output Panel shows:]
   
   $ python src/payment_processor.py
   Traceback (most recent call last):
     File "src/payment_processor.py", line 5, in <module>
       import stripe
   ModuleNotFoundError: No module named 'stripe'
   ❌ Execution failed
   ```

2. **Yantra Auto-Installs**
   ```
   Yantra: "🔄 Detected missing module: stripe"
   Yantra: "🔄 Installing stripe..."
   
   [Terminal Output Panel shows:]
   
   $ pip install stripe
   Collecting stripe
     Using cached stripe-7.4.0-py2.py3-none-any.whl (243 kB)
   Installing collected packages: stripe
   Successfully installed stripe-7.4.0
   ✅ Installation complete
   
   $ python src/payment_processor.py
   Stripe payment processor initialized
   ✅ Execution successful
   ```

3. **Yantra Updates Dependencies**
   ```
   Yantra: "✓ Added stripe==7.4.0 to requirements.txt"
   Yantra: "✓ Code committed: chore: Add stripe dependency"
   ```

**Time:** 10-15 seconds (fully automatic)

**Benefits of Terminal Integration:**

1. **No Context Switching:** Everything happens in one window
2. **Full Transparency:** See exactly what commands Yantra runs
3. **Real-Time Feedback:** Watch progress as it happens
4. **Error Visibility:** Immediately see what went wrong
5. **Learning Tool:** Understand what commands Yantra uses
6. **Trust Building:** Verify Yantra's actions in real-time

---

## End User Workflows (Original)

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

### Status: � Partially Implemented (Terminal shortcuts in Week 9-10)

**Planned Shortcuts:**

| Action | macOS | Windows/Linux |
|--------|-------|---------------|
| New chat | ⌘N | Ctrl+N |
| Open project | ⌘O | Ctrl+O |
| Settings | ⌘, | Ctrl+, |
| Focus chat | ⌘1 | Ctrl+1 |
| Focus code | ⌘2 | Ctrl+2 |
| Focus preview | ⌘3 | Ctrl+3 |
| **🆕 Focus terminal** | **⌘4** | **Ctrl+4** |
| Run tests | ⌘T | Ctrl+T |
| **🆕 Run code** | **⌘R** | **Ctrl+R** |
| **🆕 Clear terminal** | **⌘K** | **Ctrl+K** |
| **🆕 Copy terminal output** | **⌘C** | **Ctrl+C** |
| **🆕 Stop execution** | **⌘.** | **Ctrl+.** |
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

6. **🆕 Watch the Terminal Output**
   - Monitor real-time execution
   - Learn what commands Yantra uses
   - Verify installations and deployments
   - Catch errors early

7. **🆕 Let Yantra Handle Execution**
   - Don't switch to external terminal
   - Don't manually run commands
   - Let Yantra manage the full lifecycle
   - Intervene only if Yantra asks

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

### 🆕 Q: Can I run my own terminal commands?
**A:** Yantra handles all execution automatically. For custom commands, ask Yantra to run them for you. This ensures security and proper context management.

### 🆕 Q: What commands can Yantra run?
**A:** Yantra can run:
- Python: `python`, `pip`, `pytest`
- Node: `node`, `npm`, `yarn`, `jest`
- Rust: `cargo build/test/run`
- Docker: `docker build/run/ps/stop`
- Cloud: `aws`, `gcloud`, `kubectl`, `terraform`
- Git: `git` (managed via MCP protocol)

**Security:** Dangerous commands like `rm -rf`, `sudo`, `eval` are blocked.

### 🆕 Q: Can I stop a running command?
**A:** Yes, press ⌘. (macOS) or Ctrl+. (Windows/Linux), or click the "Stop" button in the terminal panel.

### 🆕 Q: How do I see past execution output?
**A:** All terminal output is preserved in the session. Scroll up in the terminal panel to see history. You can also copy all output using ⌘C or the copy button.

### 🆕 Q: What if deployment fails?
**A:** Yantra automatically rolls back failed deployments and attempts to fix the issue. You'll see the rollback process in the terminal output.

### 🆕 Q: Does Yantra remember environment variables?
**A:** Yes, Yantra maintains environment context (venv, env vars, working directory) across all commands in a session.

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
