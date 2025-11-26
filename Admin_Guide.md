# Yantra - Admin Guide

**Version:** 1.0  
**Last Updated:** November 25, 2025  
**Audience:** System Administrators and DevOps

---

## Overview

This guide provides essential information for administrators maintaining the Yantra platform. It covers system requirements, configuration, monitoring, troubleshooting, backup/recovery, and maintenance procedures.

---

## System Requirements

### Development Environment

**Operating Systems:**
- macOS 12+ (primary development platform)
- Linux (Ubuntu 20.04+, Debian 11+)
- Windows 10+ (WSL2 recommended)

**Required Software:**
- **Rust:** 1.75+ with Cargo
- **Node.js:** 18+ with npm/pnpm
- **Python:** 3.9+ (for code generation targets)
- **Git:** 2.30+
- **SQLite:** 3.44+ (bundled with Rust dependencies)

**Development Tools:**
- **VS Code:** Recommended with Rust Analyzer, Tauri, SolidJS extensions
- **Chrome/Chromium:** For browser automation testing
- **Tauri CLI:** Install via `cargo install tauri-cli`

### Runtime Requirements

**Desktop Application:**
- **Memory:** 4GB minimum, 8GB recommended
- **Disk Space:** 500MB for application + variable for user projects
- **CPU:** x86_64 or ARM64 (Apple Silicon supported)
- **Display:** 1280x720 minimum, 1920x1080 recommended

**Database:**
- SQLite database stored in user data directory
- Typical size: 10-100MB depending on project size
- Auto-grows as needed

---

## Installation & Setup

### First-Time Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/dkeviv/yantra.git
   cd yantra
   ```

2. **Install Rust Dependencies**
   ```bash
   cd src-tauri
   cargo build --release
   ```

3. **Install Frontend Dependencies**
   ```bash
   cd ../src-ui
   npm install
   # or
   pnpm install
   ```

4. **Configure LLM API Keys**
   
   Create `.env` file in project root:
   ```bash
   ANTHROPIC_API_KEY=sk-ant-...
   OPENAI_API_KEY=sk-...
   ```
   
   **Security:** Never commit `.env` to version control

5. **Run Development Build**
   ```bash
   # From project root
   npm run tauri dev
   # or
   pnpm tauri dev
   ```

6. **Build Production Release**
   ```bash
   npm run tauri build
   # or
   pnpm tauri build
   ```

### Python Environment Setup

For testing generated Python code:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install test dependencies
pip install pytest pytest-cov pytest-json-report
```

---

## Configuration Management

### Application Configuration

**Location:** `~/.yantra/config.json` (user-specific)

**Structure:**
```json
{
  "llm": {
    "primary_provider": "claude",
    "claude_model": "claude-sonnet-4-20250514",
    "openai_model": "gpt-4-turbo-2024-04-09",
    "max_retries": 3,
    "timeout_seconds": 60
  },
  "gnn": {
    "incremental_updates": true,
    "cache_enabled": true,
    "max_cache_size_mb": 100
  },
  "testing": {
    "auto_generate_tests": true,
    "success_threshold": 0.9,
    "timeout_seconds": 30
  }
}
```

**Editing Configuration:**
- Use the application's Settings UI (preferred)
- Or manually edit JSON file (restart required)

### Database Location

**SQLite Databases:**
- **GNN Graph:** `<workspace>/.yantra/graph.db`
- **User Settings:** `~/.yantra/settings.db`
- **Cache:** `~/.yantra/cache.db`

**Management:**
- Databases auto-created on first use
- Safe to delete for clean slate (will rebuild)
- Backup before major updates

### Environment Variables

**Development:**
```bash
RUST_LOG=debug              # Enable debug logging
RUST_BACKTRACE=1           # Full backtraces on panic
TAURI_DEBUG=true           # Tauri debug mode
```

**Production:**
```bash
ANTHROPIC_API_KEY=...      # Claude API key
OPENAI_API_KEY=...         # OpenAI API key
RUST_LOG=info              # Info-level logging
```

---

## Monitoring & Logging

### Log Locations

**Development:**
- Console output (stdout/stderr)
- Rust logs: `RUST_LOG=debug cargo run`

**Production:**
- **macOS:** `~/Library/Logs/Yantra/app.log`
- **Linux:** `~/.local/share/yantra/logs/app.log`
- **Windows:** `%APPDATA%\Yantra\logs\app.log`

### Log Levels

```bash
# Trace (most verbose)
RUST_LOG=trace

# Debug (development default)
RUST_LOG=debug

# Info (production default)
RUST_LOG=info

# Warn (errors and warnings only)
RUST_LOG=warn

# Error (errors only)
RUST_LOG=error
```

### Key Metrics to Monitor

1. **Performance Metrics:**
   - GNN build time: <5s for 10K LOC
   - Incremental updates: <50ms per file
   - Test execution: <30s typical
   - LLM response time: 3-10s
   - Memory usage: <500MB typical

2. **Error Rates:**
   - LLM API failures: <5%
   - Test failures: Target 0% (code should pass tests)
   - Parse errors: <1%
   - Database errors: 0%

3. **Cache Statistics:**
   - Cache hit rate: >80% after warmup
   - Cache size: <100MB typical
   - Cache evictions: Monitor for performance impact

### Health Checks

**Manual Health Check:**
```bash
# Check Rust installation
cargo --version

# Check Node.js installation
node --version

# Check Python installation
python3 --version

# Check database integrity
sqlite3 ~/.yantra/graph.db "PRAGMA integrity_check;"

# Run test suite
cd src-tauri
cargo test --lib --quiet
```

---

## Backup & Recovery

### What to Backup

**Critical Data:**
1. **Workspace Projects:** User code (user responsibility)
2. **GNN Databases:** `<workspace>/.yantra/graph.db`
3. **User Settings:** `~/.yantra/config.json`
4. **API Keys:** `.env` file (secure storage)

**Not Critical (Can Rebuild):**
- Cache databases
- Log files
- Build artifacts

### Backup Procedures

**Manual Backup:**
```bash
#!/bin/bash
# backup_yantra.sh

BACKUP_DIR="$HOME/yantra-backups/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup user settings
cp ~/.yantra/config.json "$BACKUP_DIR/"

# Backup workspace databases (example)
cp -r ~/Projects/my-project/.yantra "$BACKUP_DIR/workspace-db"

# Backup environment variables (encrypted)
cp .env "$BACKUP_DIR/" # Ensure this is stored securely!

echo "Backup complete: $BACKUP_DIR"
```

**Automated Backup (Recommended):**
- Use Time Machine (macOS)
- Use system backup tools (Linux/Windows)
- Include `~/.yantra` and workspace directories

### Recovery Procedures

**Restore from Backup:**
```bash
# 1. Restore settings
cp backup/config.json ~/.yantra/

# 2. Restore workspace database
cp -r backup/workspace-db/* ~/Projects/my-project/.yantra/

# 3. Restart application
# Database will auto-repair if corrupted
```

**Database Corruption Recovery:**
```bash
# Check database integrity
sqlite3 ~/.yantra/graph.db "PRAGMA integrity_check;"

# If corrupted, delete and rebuild
rm <workspace>/.yantra/graph.db
# Reopen project in Yantra - will rebuild automatically
```

---

## Maintenance Procedures

### Regular Maintenance (Weekly)

1. **Check for Updates**
   ```bash
   git pull origin main
   cargo update
   npm update
   ```

2. **Run Test Suite**
   ```bash
   cd src-tauri
   cargo test --lib --quiet
   cargo test --test '*' --quiet
   ```

3. **Check Disk Space**
   ```bash
   du -sh ~/.yantra
   du -sh <workspace>/.yantra
   ```

4. **Review Logs**
   ```bash
   tail -100 ~/Library/Logs/Yantra/app.log | grep -i error
   ```

### Monthly Maintenance

1. **Database Optimization**
   ```bash
   sqlite3 ~/.yantra/graph.db "VACUUM;"
   sqlite3 ~/.yantra/cache.db "VACUUM;"
   ```

2. **Clear Old Logs**
   ```bash
   find ~/Library/Logs/Yantra -type f -mtime +30 -delete
   ```

3. **Review Performance Metrics**
   - Check average GNN build times
   - Monitor incremental update performance
   - Review LLM API usage and costs

4. **Update Dependencies**
   ```bash
   cargo update
   npm audit fix
   ```

### Quarterly Maintenance

1. **Full Backup Verification**
   - Test restore procedure
   - Verify backup completeness

2. **Security Audit**
   - Rotate API keys
   - Review access logs
   - Update dependencies for security patches

3. **Performance Tuning**
   - Analyze slow queries
   - Optimize cache settings
   - Review database indexes

---

## Troubleshooting

### Common Issues

#### 1. Application Won't Start

**Symptoms:** App crashes on launch or won't open

**Solutions:**
```bash
# Check logs for errors
tail -50 ~/Library/Logs/Yantra/app.log

# Verify Rust installation
cargo --version

# Rebuild from clean state
cd src-tauri
cargo clean
cargo build --release

# Check database integrity
sqlite3 ~/.yantra/graph.db "PRAGMA integrity_check;"

# Last resort: Reset settings
rm -rf ~/.yantra
# App will recreate on next launch
```

#### 2. LLM API Failures

**Symptoms:** "Failed to generate code" errors

**Solutions:**
```bash
# Check API keys
cat .env | grep API_KEY

# Test API connectivity
curl -H "x-api-key: $ANTHROPIC_API_KEY" \
  https://api.anthropic.com/v1/messages

# Check rate limits
# Review API usage dashboard

# Switch to backup provider
# Settings → LLM → Primary Provider → OpenAI
```

#### 3. Slow Performance

**Symptoms:** GNN build >10s, incremental updates >100ms

**Solutions:**
```bash
# Check system resources
top | grep yantra

# Optimize database
sqlite3 <workspace>/.yantra/graph.db "VACUUM; ANALYZE;"

# Clear cache
rm -rf ~/.yantra/cache.db

# Reduce project size
# Exclude node_modules, .git, venv in .yantraignore

# Enable incremental updates
# Settings → GNN → Incremental Updates → Enable
```

#### 4. Test Execution Failures

**Symptoms:** "pytest not found" or tests failing

**Solutions:**
```bash
# Check Python environment
which python3
python3 --version

# Verify pytest installed
pip list | grep pytest

# Install missing dependencies
pip install pytest pytest-cov pytest-json-report

# Check virtual environment
source .venv/bin/activate

# Run tests manually
cd <workspace>
python -m pytest tests/ -v
```

#### 5. Database Locked Errors

**Symptoms:** "database is locked" errors

**Solutions:**
```bash
# Check for other processes
lsof | grep graph.db

# Kill hung processes
killall yantra

# Remove lock file
rm <workspace>/.yantra/*.db-shm
rm <workspace>/.yantra/*.db-wal

# Restart application
```

### Debug Mode

**Enable Full Debug Output:**
```bash
RUST_LOG=trace RUST_BACKTRACE=full npm run tauri dev
```

**Collect Debug Information:**
```bash
# System info
uname -a
rustc --version
cargo --version
node --version
python3 --version

# Disk space
df -h

# Memory usage
free -h  # Linux
vm_stat  # macOS

# Recent logs
tail -200 ~/Library/Logs/Yantra/app.log > debug-log.txt

# Test suite results
cargo test --lib 2>&1 | tee test-results.txt
```

---

## Performance Optimization

### GNN Performance

**Target:** <5s build for 10K LOC, <50ms incremental updates

**Optimization:**
1. Enable incremental updates (Settings → GNN)
2. Exclude large directories (.git, node_modules, venv)
3. Use SSD storage for `.yantra` directory
4. Increase cache size if RAM available

**Configuration:**
```json
{
  "gnn": {
    "incremental_updates": true,
    "cache_enabled": true,
    "max_cache_size_mb": 200,
    "exclude_patterns": [
      "node_modules/**",
      ".git/**",
      "venv/**",
      "__pycache__/**"
    ]
  }
}
```

### Memory Management

**Typical Usage:**
- Idle: 100-200MB
- Active coding: 300-500MB
- Large projects: 500MB-1GB

**Reduce Memory:**
- Decrease cache size
- Close unused projects
- Restart application periodically

### LLM Cost Optimization

**Strategies:**
1. Use Claude for primary (cheaper for most tasks)
2. Enable response caching
3. Optimize context assembly (send only relevant code)
4. Set reasonable token limits

**Cost Monitoring:**
- Track API usage in provider dashboards
- Review monthly bills
- Set up billing alerts

---

## Security Best Practices

### API Key Management

**DO:**
- Store in `.env` file (not in code)
- Use environment variables in production
- Rotate keys quarterly
- Use separate keys for dev/prod
- Restrict key permissions (API provider settings)

**DON'T:**
- Commit `.env` to version control
- Share keys in chat/email
- Use production keys in development
- Store in plain text (use keychain/vault)

### Data Privacy

**User Code:**
- Never stored on Yantra servers
- Only sent to LLM APIs (encrypted HTTPS)
- Respects `.yantraignore` file

**Sensitive Data:**
- Add sensitive files to `.yantraignore`
- Use environment variables for secrets
- Review code before sending to LLM

**Compliance:**
- GDPR: User data stays local
- SOC2: Encrypted transmission to LLM APIs
- HIPAA: Not recommended for PHI data

### Network Security

**Outbound Connections:**
- LLM APIs (api.anthropic.com, api.openai.com)
- Package registries (crates.io, npmjs.org, pypi.org)
- Git remotes (user-configured)

**Firewall Rules:**
- Allow HTTPS (443) to LLM providers
- Allow HTTP/HTTPS to package registries
- Allow SSH (22) for Git operations

---

## Scaling Considerations

### Large Projects

**10K-50K LOC:**
- Works well with default settings
- GNN build: 2-10s
- Incremental updates: <50ms

**50K-100K LOC:**
- Increase cache size to 200MB
- Consider breaking into modules
- GNN build: 10-30s
- May need more RAM (8GB+)

**100K+ LOC:**
- Monitor performance closely
- Use aggressive `.yantraignore`
- Consider monorepo tools
- May need 16GB+ RAM

### Team Usage

**Multiple Users:**
- Each user has own `~/.yantra` directory
- Workspaces shared via Git
- `.yantra/` directory in `.gitignore`
- Each user rebuilds GNN locally

**CI/CD Integration:**
- Not yet supported (MVP phase)
- Planned for Phase 2

---

## Upgrade Procedures

### Minor Version Upgrades (1.x → 1.y)

```bash
# 1. Backup current installation
cp -r ~/.yantra ~/.yantra.backup

# 2. Pull latest code
git pull origin main

# 3. Update dependencies
cd src-tauri
cargo update
cd ../src-ui
npm update

# 4. Rebuild
npm run tauri build

# 5. Test
cargo test --lib --quiet

# 6. Deploy
# Install new .dmg/.exe/.deb
```

### Major Version Upgrades (1.x → 2.x)

**Pre-Upgrade:**
1. Review CHANGELOG.md for breaking changes
2. Full backup of `~/.yantra` and workspaces
3. Test on non-production environment

**Upgrade:**
1. Follow minor upgrade steps
2. Run migration scripts (if provided)
3. Verify database compatibility

**Rollback (if needed):**
```bash
# 1. Restore backup
rm -rf ~/.yantra
cp -r ~/.yantra.backup ~/.yantra

# 2. Reinstall previous version
git checkout v1.x.x
npm run tauri build
```

---

## Disaster Recovery

### Complete System Failure

**Recovery Steps:**

1. **Reinstall Application**
   ```bash
   git clone https://github.com/dkeviv/yantra.git
   cd yantra
   npm install
   npm run tauri build
   ```

2. **Restore User Settings**
   ```bash
   cp backup/config.json ~/.yantra/
   ```

3. **Restore API Keys**
   ```bash
   cp backup/.env .env
   ```

4. **Rebuild GNN Databases**
   ```bash
   # Open each workspace in Yantra
   # GNN will automatically rebuild
   ```

5. **Verify Functionality**
   ```bash
   cargo test --lib --quiet
   ```

### Data Loss Prevention

**Strategy:**
- **3-2-1 Rule:** 3 copies, 2 different media, 1 offsite
- **Automated Backups:** Daily backups of `~/.yantra`
- **Version Control:** All code in Git
- **Cloud Sync:** Settings in cloud storage (optional)

---

## Support & Escalation

### Getting Help

1. **Check Documentation:**
   - This Admin Guide
   - Technical_Guide.md
   - Known_Issues.md

2. **Search Known Issues:**
   - Review Known_Issues.md
   - Check GitHub Issues

3. **Community Support:**
   - GitHub Discussions
   - Discord channel (if available)

4. **Report Bugs:**
   - GitHub Issues with:
     - System info (OS, Rust version, etc.)
     - Debug logs
     - Steps to reproduce
     - Expected vs actual behavior

### Collecting Diagnostic Info

**Run Diagnostic Script:**
```bash
#!/bin/bash
# diagnose.sh

echo "=== System Info ==="
uname -a
rustc --version
cargo --version
node --version
python3 --version

echo -e "\n=== Disk Space ==="
df -h | grep -E 'Filesystem|/$|/home'

echo -e "\n=== Memory ==="
free -h 2>/dev/null || vm_stat

echo -e "\n=== Yantra Files ==="
ls -lah ~/.yantra/

echo -e "\n=== Recent Errors ==="
tail -50 ~/Library/Logs/Yantra/app.log | grep -i error

echo -e "\n=== Test Results ==="
cd src-tauri
cargo test --lib --quiet 2>&1 | tail -20
```

---

## Appendix

### File Locations Reference

| Item | macOS | Linux | Windows |
|------|-------|-------|---------|
| Config | `~/.yantra/config.json` | `~/.yantra/config.json` | `%APPDATA%\yantra\config.json` |
| Logs | `~/Library/Logs/Yantra/` | `~/.local/share/yantra/logs/` | `%APPDATA%\Yantra\logs\` |
| Cache | `~/.yantra/cache.db` | `~/.yantra/cache.db` | `%APPDATA%\yantra\cache.db` |
| Workspace DB | `<workspace>/.yantra/graph.db` | Same | Same |

### Port Reference

| Service | Port | Usage |
|---------|------|-------|
| Tauri Dev Server | 1420 | Development frontend |
| Vite Dev Server | 5173 | Development hot reload |
| Browser CDP | 9222 | Chrome DevTools Protocol |

### Command Reference

```bash
# Development
npm run tauri dev              # Run development build
npm run tauri build            # Build production release

# Testing
cargo test --lib               # Unit tests
cargo test --test '*'          # Integration tests
cargo clippy                   # Linting
cargo fmt                      # Code formatting

# Database
sqlite3 <db> .schema           # Show schema
sqlite3 <db> "VACUUM;"         # Optimize
sqlite3 <db> "PRAGMA integrity_check;" # Check integrity

# Logs
tail -f ~/Library/Logs/Yantra/app.log  # Watch logs
grep -i error app.log          # Find errors
```

---

## Document Maintenance

**Update Frequency:**
- After major releases
- When procedures change
- After significant incidents

**Responsibility:**
- Primary: DevOps team
- Review: Tech lead
- Approval: System architect

**Version History:**
- 1.0 (Nov 25, 2025): Initial version

---

**Questions or Issues?**  
Contact: [Your support channel]  
Documentation: [Your docs site]  
GitHub: https://github.com/dkeviv/yantra
