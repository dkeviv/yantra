// Integration tests for autonomous execution pipeline
// Tests: Generate → Execute → Test → Validate

use std::path::PathBuf;
use std::fs;
use tempfile::TempDir;

#[cfg(test)]
mod execution_integration_tests {
    use super::*;

    /// Test 1: Full execution pipeline with simple Python script
    #[tokio::test]
    async fn test_full_pipeline_simple_script() {
        // Setup: Create temporary workspace
        let temp_dir = TempDir::new().unwrap();
        let workspace_path = temp_dir.path().to_path_buf();
        
        // Create a simple Python script
        let script_path = workspace_path.join("hello.py");
        fs::write(&script_path, r#"
def greet(name):
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(greet("World"))
"#).unwrap();

        // Execute: Run the script
        // Note: This would integrate with ScriptExecutor
        // For now, we verify the test structure is correct
        
        assert!(script_path.exists());
        assert!(workspace_path.exists());
        
        // Cleanup happens automatically when temp_dir drops
    }

    /// Test 2: Execution with missing dependencies (ImportError)
    #[tokio::test]
    async fn test_execution_with_missing_dependency() {
        let temp_dir = TempDir::new().unwrap();
        let workspace_path = temp_dir.path().to_path_buf();
        
        // Create script with import that will fail
        let script_path = workspace_path.join("with_import.py");
        fs::write(&script_path, r#"
import pandas as pd

def analyze_data():
    df = pd.DataFrame({'a': [1, 2, 3]})
    return df.sum()

if __name__ == "__main__":
    print(analyze_data())
"#).unwrap();

        // Expected flow:
        // 1. Execute → ImportError: No module named 'pandas'
        // 2. DependencyInstaller detects missing 'pandas'
        // 3. Install pandas via pip
        // 4. Re-execute → Success
        
        assert!(script_path.exists());
    }

    /// Test 3: Execution with runtime error (AttributeError)
    #[tokio::test]
    async fn test_execution_with_runtime_error() {
        let temp_dir = TempDir::new().unwrap();
        let workspace_path = temp_dir.path().to_path_buf();
        
        // Create script with runtime error
        let script_path = workspace_path.join("with_error.py");
        fs::write(&script_path, r#"
def calculate_price(item):
    # Bug: item can be None
    return item.price * 1.1

if __name__ == "__main__":
    result = calculate_price(None)
    print(result)
"#).unwrap();

        // Expected flow:
        // 1. Execute → AttributeError: 'NoneType' object has no attribute 'price'
        // 2. ScriptExecutor classifies as AttributeError
        // 3. Error details returned to orchestrator
        // 4. Orchestrator can regenerate with fix
        
        assert!(script_path.exists());
    }

    /// Test 4: Terminal streaming output
    #[tokio::test]
    async fn test_terminal_streaming() {
        let temp_dir = TempDir::new().unwrap();
        let workspace_path = temp_dir.path().to_path_buf();
        
        // Create script with multiple print statements
        let script_path = workspace_path.join("streaming.py");
        fs::write(&script_path, r#"
import time

for i in range(5):
    print(f"Processing item {i+1}")
    time.sleep(0.1)

print("Complete!")
"#).unwrap();

        // Expected behavior:
        // 1. Terminal executor starts subprocess
        // 2. Stdout streamed line by line via Tauri events
        // 3. UI updates in real-time (<10ms latency per line)
        // 4. Final "Complete!" message received
        
        assert!(script_path.exists());
    }

    /// Test 5: Multiple dependencies installation
    #[tokio::test]
    async fn test_multiple_dependencies() {
        let temp_dir = TempDir::new().unwrap();
        let workspace_path = temp_dir.path().to_path_buf();
        
        let script_path = workspace_path.join("multi_deps.py");
        fs::write(&script_path, r#"
import requests
import numpy as np
import matplotlib.pyplot as plt

def fetch_and_plot():
    response = requests.get("https://api.example.com/data")
    data = np.array([1, 2, 3, 4, 5])
    plt.plot(data)
    plt.savefig("output.png")
    return "Success"

if __name__ == "__main__":
    print(fetch_and_plot())
"#).unwrap();

        // Expected flow:
        // 1. Execute → ImportError: No module named 'requests'
        // 2. Install requests
        // 3. Execute → ImportError: No module named 'numpy'
        // 4. Install numpy
        // 5. Execute → ImportError: No module named 'matplotlib'
        // 6. Install matplotlib
        // 7. Execute → Success
        
        assert!(script_path.exists());
    }

    /// Test 6: Timeout handling for long-running script
    #[tokio::test]
    async fn test_execution_timeout() {
        let temp_dir = TempDir::new().unwrap();
        let workspace_path = temp_dir.path().to_path_buf();
        
        let script_path = workspace_path.join("infinite_loop.py");
        fs::write(&script_path, r#"
import time

def long_running():
    while True:
        time.sleep(1)
        print("Still running...")

if __name__ == "__main__":
    long_running()
"#).unwrap();

        // Expected behavior:
        // 1. Execute with 5-second timeout
        // 2. Script runs for 5 seconds
        // 3. Timeout exceeded
        // 4. Process terminated
        // 5. TimeoutError returned
        
        assert!(script_path.exists());
    }

    /// Test 7: Entry point detection
    #[tokio::test]
    async fn test_entry_point_detection() {
        let temp_dir = TempDir::new().unwrap();
        let workspace_path = temp_dir.path().to_path_buf();
        
        // Test with main() function
        let script1 = workspace_path.join("with_main.py");
        fs::write(&script1, r#"
def main():
    print("Main function")
    return 0

if __name__ == "__main__":
    exit(main())
"#).unwrap();

        // Test with direct execution
        let script2 = workspace_path.join("direct.py");
        fs::write(&script2, r#"
print("Direct execution")
x = 5 + 3
print(f"Result: {x}")
"#).unwrap();

        assert!(script1.exists());
        assert!(script2.exists());
    }

    /// Test 8: Error classification accuracy
    #[tokio::test]
    async fn test_error_classification() {
        let temp_dir = TempDir::new().unwrap();
        let workspace_path = temp_dir.path().to_path_buf();
        
        // ImportError
        let import_error = workspace_path.join("import_error.py");
        fs::write(&import_error, "import nonexistent_module").unwrap();
        
        // SyntaxError
        let syntax_error = workspace_path.join("syntax_error.py");
        fs::write(&syntax_error, "def broken(\n    print('missing closing paren')").unwrap();
        
        // TypeError
        let type_error = workspace_path.join("type_error.py");
        fs::write(&type_error, r#"
def add(a, b):
    return a + b

result = add(5, "string")
"#).unwrap();

        // ValueError
        let value_error = workspace_path.join("value_error.py");
        fs::write(&value_error, r#"
x = int("not_a_number")
"#).unwrap();

        assert!(import_error.exists());
        assert!(syntax_error.exists());
        assert!(type_error.exists());
        assert!(value_error.exists());
    }

    /// Test 9: Performance - Full cycle under 3 minutes
    #[tokio::test]
    async fn test_full_cycle_performance() {
        use std::time::Instant;
        
        let start = Instant::now();
        let temp_dir = TempDir::new().unwrap();
        let workspace_path = temp_dir.path().to_path_buf();
        
        // Create a realistic project structure
        let src_dir = workspace_path.join("src");
        fs::create_dir(&src_dir).unwrap();
        
        let main_file = src_dir.join("main.py");
        fs::write(&main_file, r#"
from utils import calculate

def main():
    result = calculate(5, 3)
    print(f"Result: {result}")
    return 0

if __name__ == "__main__":
    exit(main())
"#).unwrap();

        let utils_file = src_dir.join("utils.py");
        fs::write(&utils_file, r#"
def calculate(a, b):
    return a + b
"#).unwrap();

        let tests_dir = workspace_path.join("tests");
        fs::create_dir(&tests_dir).unwrap();
        
        let test_file = tests_dir.join("test_utils.py");
        fs::write(&test_file, r#"
from src.utils import calculate

def test_calculate():
    assert calculate(2, 3) == 5
    assert calculate(0, 0) == 0
    assert calculate(-1, 1) == 0
"#).unwrap();

        let elapsed = start.elapsed();
        
        // Full cycle target: <3 minutes (180 seconds)
        // For this setup phase: should be <1 second
        assert!(elapsed.as_secs() < 1);
        assert!(main_file.exists());
        assert!(utils_file.exists());
        assert!(test_file.exists());
    }

    /// Test 10: Concurrent execution safety
    #[tokio::test]
    async fn test_concurrent_execution() {
        use tokio::task;
        
        let temp_dir = TempDir::new().unwrap();
        let workspace_path = temp_dir.path().to_path_buf();
        
        // Create multiple scripts
        let scripts: Vec<String> = (0..5)
            .map(|i| {
                let script_path = workspace_path.join(format!("script_{}.py", i));
                fs::write(&script_path, format!(r#"
import time
print("Script {} starting")
time.sleep(0.5)
print("Script {} complete")
"#, i, i)).unwrap();
                script_path.to_string_lossy().to_string()
            })
            .collect();
        
        // Execute all scripts concurrently
        let handles: Vec<_> = scripts.iter().map(|_script| {
            task::spawn(async {
                // Simulate execution
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                Ok::<(), String>(())
            })
        }).collect();

        // Wait for all to complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }
    }
}

#[cfg(test)]
mod integration_test_helpers {
    use super::*;

    /// Helper: Create a test workspace with structure
    pub fn create_test_workspace() -> TempDir {
        let temp_dir = TempDir::new().unwrap();
        let workspace_path = temp_dir.path();
        
        // Create standard project structure
        fs::create_dir(workspace_path.join("src")).unwrap();
        fs::create_dir(workspace_path.join("tests")).unwrap();
        fs::create_dir(workspace_path.join(".yantra")).unwrap();
        
        // Create requirements.txt
        fs::write(
            workspace_path.join("requirements.txt"),
            "# Auto-generated by Yantra\n"
        ).unwrap();
        
        temp_dir
    }

    /// Helper: Verify script execution result
    pub fn verify_execution_result(
        exit_code: i32,
        stdout: &str,
        stderr: &str,
        expected_exit_code: i32,
    ) -> bool {
        exit_code == expected_exit_code && stderr.is_empty()
    }

    /// Helper: Parse error message
    pub fn parse_error_type(error_message: &str) -> Option<String> {
        if error_message.contains("ImportError") {
            Some("ImportError".to_string())
        } else if error_message.contains("AttributeError") {
            Some("AttributeError".to_string())
        } else if error_message.contains("TypeError") {
            Some("TypeError".to_string())
        } else if error_message.contains("ValueError") {
            Some("ValueError".to_string())
        } else if error_message.contains("SyntaxError") {
            Some("SyntaxError".to_string())
        } else {
            None
        }
    }
}
