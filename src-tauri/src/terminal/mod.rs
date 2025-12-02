// File: src-tauri/src/terminal/mod.rs
// Purpose: Terminal module exports
// Last Updated: November 29, 2025

pub mod pty_terminal;
pub mod executor;

pub use pty_terminal::{TerminalManager, TerminalSession};
pub use executor::{TerminalExecutor, TerminalState, TerminalInfo};
