// Browser module - Chrome DevTools Protocol integration
// Purpose: Validate web applications by monitoring console errors and runtime behavior

pub mod cdp;
pub mod validator;

pub use cdp::{BrowserSession, ConsoleMessage, ConsoleLevel};
pub use validator::{BrowserValidator, ValidationResult};
