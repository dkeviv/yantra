# Yantra - Manual Testing Guide for Beta Testers

**Version:** 1.0  
**Last Updated:** November 23, 2025  
**Target Audience:** Amateur testers (no coding experience required)  
**Estimated Time:** 2-3 hours for complete testing

---

## 🎯 Purpose

This guide helps you test every feature of Yantra systematically. Follow step-by-step instructions to verify the application works correctly on your machine.

---

## 📋 Pre-Testing Setup

### System Requirements

**Minimum:**
- **Operating System**: macOS 10.15+, Windows 10+, or Ubuntu 20.04+
- **RAM**: 4GB (8GB recommended)
- **Disk Space**: 2GB free
- **Internet**: Required for AI features

**Software Prerequisites:**
- **Python**: 3.8 or higher ([Download](https://www.python.org/downloads/))
- **Node.js**: 16 or higher ([Download](https://nodejs.org/))
- **Git**: 2.0 or higher ([Download](https://git-scm.com/downloads))

### Installation Steps

**Step 1: Install Yantra**
```
macOS: Double-click Yantra.dmg → Drag to Applications
Windows: Run Yantra_Setup.exe → Follow wizard
Linux: sudo dpkg -i yantra_0.1.0_amd64.deb
```

**Step 2: First Launch**
```
1. Open Yantra application
2. You should see:
   - Left sidebar: File tree (empty initially)
   - Center: Chat interface "How can I help you?"
   - Right: Code viewer (empty)
   - Bottom: Terminal (ready)
3. ✅ PASS if all panels visible
4. ❌ FAIL if app crashes or panels missing
```

**Step 3: Create Test Project**
```
1. Click "File" menu → "Open Folder"
2. Create new folder: "yantra-test-project"
3. Select this folder
4. ✅ PASS if file tree shows "yantra-test-project" folder
5. ❌ FAIL if folder doesn't appear
```

---

## 🧪 Feature Testing Checklist

### Test 1: Basic Code Generation

**Feature:** Autonomous code generation  
**Time:** 5 minutes

**Steps:**
1. In chat, type: "Create a Python function that calculates fibonacci numbers"
2. Press Enter or click Send
3. Wait for response (should take 5-10 seconds)

**Expected Results:**
```
✅ PASS if you see:
- Agent status shows: "🤖 Generate" phase
- Confidence score appears (e.g., "85%")
- Code appears in code viewer
- Code contains a fibonacci function
- Success notification appears

❌ FAIL if:
- Nothing happens after 30 seconds
- Error message appears
- No code is generated
```

**Test Results:**
- [ ] PASS
- [ ] FAIL (describe issue): ___________________________

---

### Test 2: Dependency Validation (GNN)

**Feature:** GNN-based dependency checking  
**Time:** 5 minutes

**Steps:**
1. In chat, type: "Create a Python function that uses the requests library to fetch data from an API"
2. Wait for code generation
3. In chat, type: "Now create another function that calls the first function"

**Expected Results:**
```
✅ PASS if:
- First function is generated with "import requests"
- Second function imports and uses first function
- No import errors reported
- Dependencies are correctly tracked

❌ FAIL if:
- Second function doesn't import first function
- Import errors appear
- Code doesn't compile
```

**Test Results:**
- [ ] PASS
- [ ] FAIL (describe issue): ___________________________

---

### Test 3: Code Execution

**Feature:** Autonomous code execution  
**Time:** 5 minutes

**Steps:**
1. In chat, type: "Execute the fibonacci function with input 10"
2. Wait for execution

**Expected Results:**
```
✅ PASS if:
- Agent status shows: "⚡ Execute" phase
- Terminal shows output: "55" (10th fibonacci number)
- Success notification appears
- No errors in terminal

❌ FAIL if:
- Code doesn't execute
- Wrong output appears
- Error messages appear
```

**Test Results:**
- [ ] PASS
- [ ] FAIL (describe issue): ___________________________

---

### Test 4: Missing Dependency Auto-Fix

**Feature:** Automatic dependency installation  
**Time:** 10 minutes

**Steps:**
1. In chat, type: "Create a script that uses pandas to analyze a CSV file"
2. Wait for code generation
3. In chat, type: "Execute this script"
4. If pandas is not installed, watch what happens

**Expected Results:**
```
✅ PASS if:
- Yellow notification: "⚠️ Import error: pandas missing"
- Blue notification: "🔄 Auto-installing pandas..."
- Terminal shows: "pip install pandas"
- Green notification: "✅ Dependency installed"
- Script executes successfully

❌ FAIL if:
- Execution fails without auto-fix attempt
- pandas doesn't get installed
- Script remains broken
```

**Test Results:**
- [ ] PASS
- [ ] FAIL (describe issue): ___________________________

---

### Test 5: Auto-Retry on Errors

**Feature:** Intelligent auto-retry with fixing  
**Time:** 10 minutes

**Steps:**
1. In chat, type: "Create a function that divides two numbers"
2. Wait for generation
3. In chat, type: "Execute it with inputs 10 and 0"
4. Watch error handling

**Expected Results:**
```
✅ PASS if:
- Division by zero error detected
- Agent analyzes error
- Yellow notification: "⚠️ Runtime error detected"
- Agent attempts fix (adds zero check)
- Blue notification: "🔄 Retrying with fix..."
- Fixed version executes successfully

❌ FAIL if:
- Error not detected
- No retry attempt
- Same error repeats
```

**Test Results:**
- [ ] PASS
- [ ] FAIL (describe issue): ___________________________

---

### Test 6: Security Scanning

**Feature:** Automatic security vulnerability detection  
**Time:** 10 minutes

**Steps:**
1. In chat, type: "Create a database query function that takes user input and queries a users table"
2. Wait for generation (might generate vulnerable code)
3. In chat, type: "Scan for security vulnerabilities"

**Expected Results:**
```
✅ PASS if:
- Notification: "🔍 Security scan started"
- If SQL injection found:
  - Red notification: "🚨 CRITICAL: SQL injection vulnerability"
  - Blue notification: "🔄 Auto-fixing..."
  - Code is updated to use parameterized queries
  - Green notification: "✅ Security fix applied"
- Else: "✅ No vulnerabilities found"

❌ FAIL if:
- Scan doesn't run
- Vulnerability not detected
- Fix not applied
```

**Test Results:**
- [ ] PASS
- [ ] FAIL (describe issue): ___________________________

---

### Test 7: Git Integration

**Feature:** AI-powered Git commits  
**Time:** 10 minutes

**Prerequisites:** Git must be installed

**Steps:**
1. In terminal (bottom panel), type: `git init`
2. In chat, type: "Commit all changes with an AI-generated message"
3. Wait for commit

**Expected Results:**
```
✅ PASS if:
- Blue notification: "📝 Generating commit message..."
- Terminal shows: "git add ."
- Terminal shows: "git commit -m 'feat: ...'"
- Green notification: "✅ Changes committed"
- Commit message is descriptive (not just "update")

❌ FAIL if:
- Git commands fail
- Commit message is generic
- Files not committed
```

**Test Results:**
- [ ] PASS
- [ ] FAIL (describe issue): ___________________________

---

### Test 8: Browser Validation (Web Apps Only)

**Feature:** Real browser testing with CDP  
**Time:** 15 minutes

**Prerequisites:** Chrome browser installed

**Steps:**
1. In chat, type: "Create a simple HTML page with a button that shows an alert"
2. Wait for generation (creates index.html)
3. In chat, type: "Validate this in a browser"

**Expected Results:**
```
✅ PASS if:
- Chrome opens automatically
- Page loads (might be file:// URL)
- No console errors shown
- Green notification: "✅ Browser validation passed"
- Metrics shown: load time, DOM size

❌ FAIL if:
- Chrome doesn't open
- Page has errors
- Validation fails
```

**Test Results:**
- [ ] PASS
- [ ] FAIL (describe issue): ___________________________
- [ ] SKIP (not applicable - no web code)

---

### Test 9: Package Building

**Feature:** Multi-format package generation  
**Time:** 15 minutes

**Steps:**
1. In chat, type: "Build a Python wheel package for this project"
2. Wait for build process

**Expected Results:**
```
✅ PASS if:
- setup.py file is created
- Blue notification: "🔨 Building package..."
- Terminal shows: "python -m build"
- dist/ folder appears in file tree
- .whl file created
- Green notification: "✅ Package built successfully"

❌ FAIL if:
- Build fails
- No .whl file created
- Errors in terminal
```

**Test Results:**
- [ ] PASS
- [ ] FAIL (describe issue): ___________________________

---

### Test 10: Real-Time UI Updates

**Feature:** Live agent status and progress  
**Time:** 5 minutes

**Steps:**
1. In chat, type: "Create a complex Python class with multiple methods"
2. Watch the UI during generation

**Expected Results:**
```
✅ PASS if you see:
- Agent status changes:
  - "⚙️ Context" → "🤖 Generate" → "✓ Validate"
- Confidence score updates in real-time:
  - Starts around 70-80%
  - Increases to 85-95%
- Progress bar moves (0% → 100%)
- Phase icons change colors
- Notifications appear at each step

❌ FAIL if:
- UI doesn't update
- Agent status stays static
- No progress indication
```

**Test Results:**
- [ ] PASS
- [ ] FAIL (describe issue): ___________________________

---

### Test 11: Multi-LLM Fallback

**Feature:** Automatic failover between AI models  
**Time:** 10 minutes

**Setup:** Intentionally cause primary LLM to fail (requires API key configuration)

**Steps:**
1. Go to Settings → LLM Config
2. Enter invalid API key for primary model (Claude)
3. Ensure secondary model (GPT-4) has valid key
4. In chat, type: "Create a simple hello world function"

**Expected Results:**
```
✅ PASS if:
- Red notification: "⚠️ Primary model failed"
- Blue notification: "🔄 Switching to GPT-4..."
- Code is generated successfully
- Agent continues without user intervention

❌ FAIL if:
- Generation completely fails
- No fallback attempt
- User must manually intervene
```

**Test Results:**
- [ ] PASS
- [ ] FAIL (describe issue): ___________________________

---

### Test 12: Confidence Scoring

**Feature:** Multi-factor confidence calculation  
**Time:** 5 minutes

**Steps:**
1. In chat, type: "Create a simple add function"
2. Note confidence score
3. In chat, type: "Create a complex machine learning model training pipeline"
4. Note confidence score

**Expected Results:**
```
✅ PASS if:
- Simple function: Confidence 90-95% (green)
- Complex task: Confidence 70-85% (yellow/green)
- Confidence scores make sense
- Lower confidence for complex tasks

❌ FAIL if:
- All confidence scores are same
- Unrealistic scores (e.g., 100% always)
- No confidence shown
```

**Test Results:**
- [ ] PASS
- [ ] FAIL (describe issue): ___________________________

---

### Test 13: Context Compression

**Feature:** Intelligent context size management  
**Time:** 10 minutes

**Steps:**
1. Create a large project (10+ files)
2. In chat, type: "Explain the entire project structure"
3. Watch how Yantra handles large context

**Expected Results:**
```
✅ PASS if:
- Response includes all files mentioned
- No "context limit exceeded" error
- Response is coherent
- Notification: "📉 Context compressed by XX%"

❌ FAIL if:
- Error about token limits
- Response is truncated
- Missing important files
```

**Test Results:**
- [ ] PASS
- [ ] FAIL (describe issue): ___________________________

---

### Test 14: Terminal Integration

**Feature:** Secure terminal command execution  
**Time:** 10 minutes

**Steps:**
1. In chat, type: "Install pytest using pip"
2. Wait for execution
3. In terminal, type: `pytest --version`

**Expected Results:**
```
✅ PASS if:
- Command runs in terminal panel
- Output is shown in real-time
- Command completes successfully
- pytest version is displayed

❌ FAIL if:
- Command doesn't execute
- Terminal freezes
- No output shown
```

**Test Results:**
- [ ] PASS
- [ ] FAIL (describe issue): ___________________________

---

### Test 15: File Operations

**Feature:** Create, read, update, delete files  
**Time:** 10 minutes

**Steps:**
1. In chat, type: "Create a new file called config.json with sample configuration"
2. In chat, type: "Read the contents of config.json"
3. In chat, type: "Update config.json to change the port to 8080"
4. In chat, type: "Delete config.json"

**Expected Results:**
```
✅ PASS if:
- config.json appears in file tree
- Contents are shown in code viewer
- File is updated correctly
- File disappears from tree after delete
- Notifications at each step

❌ FAIL if:
- File operations fail
- Content is wrong
- File tree not updated
```

**Test Results:**
- [ ] PASS
- [ ] FAIL (describe issue): ___________________________

---

### Test 16: Error Recovery

**Feature:** State recovery after errors  
**Time:** 10 minutes

**Steps:**
1. In chat, type: "Create a function with intentional syntax error"
2. Watch error detection
3. In chat, type: "Fix the syntax error"

**Expected Results:**
```
✅ PASS if:
- Syntax error is detected
- Red notification: "🚨 Syntax error found"
- Agent attempts fix automatically
- Fixed code is valid Python
- Green notification: "✅ Error fixed"

❌ FAIL if:
- Error not detected
- No fix attempted
- Fixed code still has errors
```

**Test Results:**
- [ ] PASS
- [ ] FAIL (describe issue): ___________________________

---

### Test 17: Integration Tests (Technical)

**Feature:** End-to-end pipeline validation  
**Time:** 5 minutes (automated)

**Steps:**
1. Open terminal
2. Navigate to application directory
3. Run: `cargo test --test integration`

**Expected Results:**
```
✅ PASS if:
- All 32 integration tests pass
- Test run completes in <1 minute
- Output shows: "32 passed; 0 failed"

❌ FAIL if:
- Any test fails
- Test run hangs
- Errors during execution
```

**Test Results:**
- [ ] PASS
- [ ] FAIL (describe issue): ___________________________

---

### Test 18: Performance Benchmarks

**Feature:** Meet performance targets  
**Time:** 10 minutes

**Steps:**
1. In chat, type: "Create a REST API with 5 endpoints"
2. Time the complete cycle (use stopwatch)

**Expected Results:**
```
✅ PASS if:
- Context assembly: <5 seconds
- Code generation: <10 seconds
- Validation: <2 seconds
- Total cycle: <2 minutes

❌ FAIL if:
- Any step takes >2× expected time
- Total >5 minutes
```

**Test Results:**
- [ ] PASS
- [ ] FAIL (describe issue): ___________________________
- Context: ___ seconds
- Generation: ___ seconds
- Validation: ___ seconds
- Total: ___ seconds

---

## 📊 Summary Report

### Overall Test Results

| Feature | Status | Notes |
|---------|--------|-------|
| 1. Code Generation | ⬜ | |
| 2. Dependency Validation | ⬜ | |
| 3. Code Execution | ⬜ | |
| 4. Auto-Fix Dependencies | ⬜ | |
| 5. Auto-Retry | ⬜ | |
| 6. Security Scanning | ⬜ | |
| 7. Git Integration | ⬜ | |
| 8. Browser Validation | ⬜ | |
| 9. Package Building | ⬜ | |
| 10. Real-Time UI | ⬜ | |
| 11. LLM Fallback | ⬜ | |
| 12. Confidence Scoring | ⬜ | |
| 13. Context Compression | ⬜ | |
| 14. Terminal Integration | ⬜ | |
| 15. File Operations | ⬜ | |
| 16. Error Recovery | ⬜ | |
| 17. Integration Tests | ⬜ | |
| 18. Performance | ⬜ | |

**Legend:** ✅ PASS | ❌ FAIL | ⏭️ SKIP

### Pass Rate
- Total Tests: 18
- Passed: ___ / 18
- Failed: ___ / 18
- Skipped: ___ / 18
- **Pass Rate: ____%**

### Critical Issues Found
1. ___________________________________________
2. ___________________________________________
3. ___________________________________________

### Minor Issues Found
1. ___________________________________________
2. ___________________________________________
3. ___________________________________________

### Suggestions for Improvement
1. ___________________________________________
2. ___________________________________________
3. ___________________________________________

---

## 🐛 Bug Reporting

If you find bugs, please report them with:

**Bug Report Template:**
```
Title: [Brief description]

Steps to Reproduce:
1. [Step 1]
2. [Step 2]
3. [Step 3]

Expected Behavior:
[What should happen]

Actual Behavior:
[What actually happened]

Screenshots:
[If applicable]

System Info:
- OS: [macOS 13.0 / Windows 11 / Ubuntu 22.04]
- Yantra Version: [0.1.0]
- Python Version: [3.10.0]

Additional Context:
[Any other relevant information]
```

**Where to Report:**
- GitHub Issues: [Repository URL]
- Email: support@yantra.dev
- Discord: [Community Server]

---

## ✅ Testing Complete Checklist

Before submitting your test results:

- [ ] All 18 features tested
- [ ] Summary report filled out
- [ ] Pass rate calculated
- [ ] Critical issues documented
- [ ] Screenshots captured (if bugs found)
- [ ] System information noted
- [ ] Test duration recorded: ___ hours

**Thank you for testing Yantra! Your feedback helps make it better for everyone.** 🙏

---

**Document Version:** 1.0  
**Last Updated:** November 23, 2025  
**Maintained By:** Yantra Team
