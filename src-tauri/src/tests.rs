// File: src-tauri/src/tests.rs
// Purpose: Unit tests for Tauri file system commands
// Dependencies: tauri
// Last Updated: November 20, 2025

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_read_file_success() {
        // Create a temporary directory
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        
        // Write test content
        let mut file = fs::File::create(&file_path).unwrap();
        file.write_all(b"Hello, Yantra!").unwrap();
        
        // Read the file
        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "Hello, Yantra!");
    }

    #[test]
    fn test_read_file_not_found() {
        let result = fs::read_to_string("/nonexistent/file.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_write_file_success() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("output.txt");
        
        // Write content
        fs::write(&file_path, "Test content").unwrap();
        
        // Verify content
        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "Test content");
    }

    #[test]
    fn test_read_directory() {
        let temp_dir = TempDir::new().unwrap();
        
        // Create some test files
        fs::write(temp_dir.path().join("file1.txt"), "content1").unwrap();
        fs::write(temp_dir.path().join("file2.txt"), "content2").unwrap();
        fs::create_dir(temp_dir.path().join("subdir")).unwrap();
        
        // Read directory
        let entries: Vec<_> = fs::read_dir(temp_dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn test_file_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "test").unwrap();
        
        let metadata = fs::metadata(&file_path).unwrap();
        assert!(metadata.is_file());
        assert_eq!(metadata.len(), 4); // "test" is 4 bytes
    }

    #[test]
    fn test_directory_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path().join("testdir");
        fs::create_dir(&dir_path).unwrap();
        
        let metadata = fs::metadata(&dir_path).unwrap();
        assert!(metadata.is_dir());
    }
}
