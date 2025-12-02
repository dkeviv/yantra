// File: src-tauri/src/terminal/executor.rs
// Purpose: Terminal execution management with process detection and reuse
// Created: November 29, 2025

use std::collections::HashMap;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Terminal state tracking
#[derive(Debug, Clone, PartialEq)]
pub enum TerminalState {
    Idle,
    Busy,
    Closed,
}

/// Terminal session info
#[derive(Debug, Clone)]
pub struct TerminalInfo {
    pub id: String,
    pub state: TerminalState,
    pub last_command: Option<String>,
    pub last_used: Instant,
    pub pid: Option<u32>,
}

/// Terminal executor with smart reuse and process detection
pub struct TerminalExecutor {
    terminals: Arc<Mutex<HashMap<String, TerminalInfo>>>,
    idle_timeout: Duration,
}

impl TerminalExecutor {
    /// Create a new terminal executor
    pub fn new() -> Self {
        Self {
            terminals: Arc::new(Mutex::new(HashMap::new())),
            idle_timeout: Duration::from_secs(300), // 5 minutes
        }
    }

    /// Register a terminal with the executor
    pub fn register_terminal(&self, id: String, pid: Option<u32>) {
        let mut terminals = self.terminals.lock().unwrap();
        terminals.insert(
            id.clone(),
            TerminalInfo {
                id,
                state: TerminalState::Idle,
                last_command: None,
                last_used: Instant::now(),
                pid,
            },
        );
    }

    /// Mark a terminal as busy
    pub fn mark_busy(&self, id: &str, command: String) {
        if let Some(info) = self.terminals.lock().unwrap().get_mut(id) {
            info.state = TerminalState::Busy;
            info.last_command = Some(command);
            info.last_used = Instant::now();
        }
    }

    /// Mark a terminal as idle
    pub fn mark_idle(&self, id: &str) {
        if let Some(info) = self.terminals.lock().unwrap().get_mut(id) {
            info.state = TerminalState::Idle;
            info.last_used = Instant::now();
        }
    }

    /// Mark a terminal as closed
    pub fn mark_closed(&self, id: &str) {
        if let Some(info) = self.terminals.lock().unwrap().get_mut(id) {
            info.state = TerminalState::Closed;
        }
    }

    /// Check if a terminal is busy using platform-specific process detection
    pub fn is_terminal_busy(&self, id: &str) -> Result<bool, String> {
        let terminals = self.terminals.lock().unwrap();
        let info = terminals
            .get(id)
            .ok_or_else(|| format!("Terminal {} not found", id))?;

        // If we marked it as closed, it's not busy
        if info.state == TerminalState::Closed {
            return Ok(false);
        }

        // If we have a PID, check if there's an active process
        if let Some(pid) = info.pid {
            self.check_process_busy(pid)
        } else {
            // Fallback to our tracked state
            Ok(info.state == TerminalState::Busy)
        }
    }

    /// Check if a process is busy using platform-specific methods
    fn check_process_busy(&self, pid: u32) -> Result<bool, String> {
        #[cfg(target_os = "macos")]
        {
            self.check_process_busy_macos(pid)
        }

        #[cfg(target_os = "linux")]
        {
            self.check_process_busy_linux(pid)
        }

        #[cfg(target_os = "windows")]
        {
            self.check_process_busy_windows(pid)
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            // Fallback for unsupported platforms
            Ok(false)
        }
    }

    /// Check if a process is busy on macOS using ps
    #[cfg(target_os = "macos")]
    fn check_process_busy_macos(&self, pid: u32) -> Result<bool, String> {
        // Use ps to check process state
        // stat column: R (running), S (sleeping), T (stopped), Z (zombie)
        // A sleeping shell is idle, a running process means busy
        let output = Command::new("ps")
            .args(&["-p", &pid.to_string(), "-o", "stat="])
            .output()
            .map_err(|e| format!("Failed to run ps: {}", e))?;

        if !output.status.success() {
            // Process doesn't exist anymore
            return Ok(false);
        }

        let stat = String::from_utf8_lossy(&output.stdout).trim().to_string();
        
        // If the process has a '+' it's in foreground
        // If stat starts with 'R', it's running
        // If stat starts with 'S' and has no '+', it's idle (background sleep)
        let is_busy = stat.starts_with('R') || stat.contains('+');
        
        Ok(is_busy)
    }

    /// Check if a process is busy on Linux using /proc filesystem
    #[cfg(target_os = "linux")]
    fn check_process_busy_linux(&self, pid: u32) -> Result<bool, String> {
        use std::fs;

        // Check if process exists
        let proc_path = format!("/proc/{}", pid);
        if !std::path::Path::new(&proc_path).exists() {
            return Ok(false);
        }

        // Read process status
        let status_path = format!("/proc/{}/stat", pid);
        let stat = fs::read_to_string(&status_path)
            .map_err(|e| format!("Failed to read process stat: {}", e))?;

        // Parse state (third field after process name in parentheses)
        // State: R (running), S (sleeping), T (stopped), Z (zombie)
        let parts: Vec<&str> = stat.split_whitespace().collect();
        if parts.len() < 3 {
            return Ok(false);
        }

        let state = parts[2];
        
        // Running state means busy
        let is_busy = state == "R";
        
        Ok(is_busy)
    }

    /// Check if a process is busy on Windows using tasklist
    #[cfg(target_os = "windows")]
    fn check_process_busy_windows(&self, pid: u32) -> Result<bool, String> {
        // Use tasklist to check if process exists and get info
        let output = Command::new("tasklist")
            .args(&["/FI", &format!("PID eq {}", pid), "/FO", "CSV", "/NH"])
            .output()
            .map_err(|e| format!("Failed to run tasklist: {}", e))?;

        if !output.status.success() {
            return Ok(false);
        }

        let result = String::from_utf8_lossy(&output.stdout);
        
        // If tasklist finds the process, we consider it potentially busy
        // Windows doesn't easily expose foreground/background state without WMI
        // For simplicity, if the process exists, assume it might be busy
        Ok(!result.trim().is_empty() && !result.contains("INFO: No tasks"))
    }

    /// Find an idle terminal that can be reused
    pub fn find_idle_terminal(&self) -> Option<String> {
        let terminals = self.terminals.lock().unwrap();
        
        for (id, info) in terminals.iter() {
            // Skip closed terminals
            if info.state == TerminalState::Closed {
                continue;
            }

            // Check if terminal is idle
            if info.state == TerminalState::Idle {
                // Also verify it's actually idle (not just marked as idle)
                if let Ok(false) = self.is_terminal_busy(id) {
                    // Check if it's not too old (within idle timeout)
                    if info.last_used.elapsed() < self.idle_timeout {
                        return Some(id.clone());
                    }
                }
            }
        }

        None
    }

    /// Get terminal state
    pub fn get_state(&self, id: &str) -> Option<TerminalState> {
        self.terminals
            .lock()
            .unwrap()
            .get(id)
            .map(|info| info.state.clone())
    }

    /// Get all terminal info
    pub fn list_terminals(&self) -> Vec<TerminalInfo> {
        self.terminals
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect()
    }

    /// Clean up closed terminals
    pub fn cleanup_closed(&self) {
        self.terminals
            .lock()
            .unwrap()
            .retain(|_, info| info.state != TerminalState::Closed);
    }

    /// Clean up idle terminals that have exceeded timeout
    pub fn cleanup_idle_timeout(&self) {
        let now = Instant::now();
        self.terminals
            .lock()
            .unwrap()
            .retain(|_, info| {
                if info.state == TerminalState::Idle {
                    info.last_used.elapsed() < self.idle_timeout
                } else {
                    true
                }
            });
    }
}

impl Default for TerminalExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_terminal() {
        let executor = TerminalExecutor::new();
        executor.register_terminal("term-1".to_string(), Some(12345));
        
        let state = executor.get_state("term-1");
        assert_eq!(state, Some(TerminalState::Idle));
    }

    #[test]
    fn test_mark_busy_and_idle() {
        let executor = TerminalExecutor::new();
        executor.register_terminal("term-1".to_string(), None);
        
        executor.mark_busy("term-1", "npm start".to_string());
        assert_eq!(executor.get_state("term-1"), Some(TerminalState::Busy));
        
        executor.mark_idle("term-1");
        assert_eq!(executor.get_state("term-1"), Some(TerminalState::Idle));
    }

    #[test]
    fn test_find_idle_terminal() {
        let executor = TerminalExecutor::new();
        executor.register_terminal("term-1".to_string(), None);
        executor.register_terminal("term-2".to_string(), None);
        
        executor.mark_busy("term-1", "test".to_string());
        executor.mark_idle("term-2");
        
        let idle = executor.find_idle_terminal();
        assert!(idle.is_some());
        assert_eq!(idle.unwrap(), "term-2");
    }

    #[test]
    fn test_cleanup_closed() {
        let executor = TerminalExecutor::new();
        executor.register_terminal("term-1".to_string(), None);
        executor.register_terminal("term-2".to_string(), None);
        
        executor.mark_closed("term-1");
        executor.cleanup_closed();
        
        assert_eq!(executor.get_state("term-1"), None);
        assert_eq!(executor.get_state("term-2"), Some(TerminalState::Idle));
    }
}
