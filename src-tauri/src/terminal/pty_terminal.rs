// File: src-tauri/src/terminal/pty_terminal.rs
// Purpose: Real PTY-based terminal implementation with full shell support
// Last Updated: November 28, 2025

use portable_pty::{CommandBuilder, NativePtySystem, PtyPair, PtySize, PtySystem};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};
use tauri::Window;
use tokio::task;

/// Terminal session managing a PTY
pub struct TerminalSession {
    pub id: String,
    pub name: String,
    pty_pair: PtyPair,
    writer: Box<dyn Write + Send>,
}

impl TerminalSession {
    /// Create a new terminal session with a shell
    pub fn new(id: String, name: String, shell: Option<String>) -> Result<Self, String> {
        let pty_system = NativePtySystem::default();
        
        // Create PTY with reasonable size
        let pty_pair = pty_system
            .openpty(PtySize {
                rows: 24,
                cols: 80,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|e| format!("Failed to create PTY: {}", e))?;

        // Determine shell to use
        let shell_cmd = shell.unwrap_or_else(|| {
            std::env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".to_string())
        });

        // Determine working directory - use HOME instead of current_dir to avoid src-tauri folder
        let working_dir = std::env::var("HOME")
            .ok()
            .and_then(|h| std::path::PathBuf::from(h).canonicalize().ok())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

        // Spawn shell process
        let mut cmd = CommandBuilder::new(&shell_cmd);
        cmd.cwd(working_dir);
        
        pty_pair
            .slave
            .spawn_command(cmd)
            .map_err(|e| format!("Failed to spawn shell: {}", e))?;

        // Get writer for sending input (use the master PTY directly)
        let writer = pty_pair.master.take_writer()
            .map_err(|e| format!("Failed to get writer: {}", e))?;

        Ok(Self {
            id,
            name,
            pty_pair,
            writer,
        })
    }

    /// Write input to the terminal
    pub fn write_input(&mut self, data: &[u8]) -> Result<(), String> {
        self.writer
            .write_all(data)
            .map_err(|e| format!("Failed to write to terminal: {}", e))?;
        self.writer
            .flush()
            .map_err(|e| format!("Failed to flush terminal: {}", e))?;
        Ok(())
    }

    /// Resize the terminal
    pub fn resize(&mut self, rows: u16, cols: u16) -> Result<(), String> {
        self.pty_pair
            .master
            .resize(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|e| format!("Failed to resize terminal: {}", e))?;
        Ok(())
    }

    /// Get a reader for streaming output
    pub fn get_reader(&mut self) -> Result<Box<dyn Read + Send>, String> {
        self.pty_pair.master.try_clone_reader()
            .map_err(|e| format!("Failed to get reader: {}", e))
    }
}

/// Manager for multiple terminal sessions
pub struct TerminalManager {
    sessions: Arc<Mutex<HashMap<String, Arc<Mutex<TerminalSession>>>>>,
}

impl TerminalManager {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a new terminal session
    pub fn create_terminal(
        &self,
        id: String,
        name: String,
        shell: Option<String>,
        window: Window,
    ) -> Result<(), String> {
        let mut session = TerminalSession::new(id.clone(), name, shell)?;
        
        // Get reader before moving session
        let mut reader = session.get_reader()?;
        
        // Store session
        self.sessions
            .lock()
            .unwrap()
            .insert(id.clone(), Arc::new(Mutex::new(session)));

        // Start output streaming in background task
        let terminal_id = id.clone();
        task::spawn_blocking(move || {
            let mut buffer = [0u8; 8192];
            
            loop {
                match reader.read(&mut buffer) {
                    Ok(0) => {
                        // EOF - terminal closed
                        let _ = window.emit("terminal-closed", serde_json::json!({
                            "terminal_id": terminal_id,
                        }));
                        break;
                    }
                    Ok(n) => {
                        // Got data - send to frontend
                        let data = &buffer[..n];
                        // Convert to base64 to safely transmit binary data
                        use base64::Engine;
                        let encoded = base64::engine::general_purpose::STANDARD.encode(data);
                        let _ = window.emit("terminal-data", serde_json::json!({
                            "terminal_id": terminal_id,
                            "data": encoded,
                        }));
                    }
                    Err(e) => {
                        eprintln!("Error reading from terminal {}: {}", terminal_id, e);
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    /// Write input to a terminal
    pub fn write_input(&self, id: &str, data: &[u8]) -> Result<(), String> {
        let sessions = self.sessions.lock().unwrap();
        let session_arc = sessions
            .get(id)
            .ok_or_else(|| format!("Terminal {} not found", id))?;
        let mut session = session_arc.lock().unwrap();
        session.write_input(data)
    }

    /// Resize a terminal
    pub fn resize(&self, id: &str, rows: u16, cols: u16) -> Result<(), String> {
        let sessions = self.sessions.lock().unwrap();
        let session_arc = sessions
            .get(id)
            .ok_or_else(|| format!("Terminal {} not found", id))?;
        let mut session = session_arc.lock().unwrap();
        session.resize(rows, cols)
    }

    /// Close a terminal
    pub fn close_terminal(&self, id: &str) -> Result<(), String> {
        let mut sessions = self.sessions.lock().unwrap();
        sessions.remove(id);
        Ok(())
    }

    /// Get list of terminal IDs
    pub fn list_terminals(&self) -> Vec<String> {
        self.sessions.lock().unwrap().keys().cloned().collect()
    }
}
