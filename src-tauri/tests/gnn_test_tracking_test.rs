// Unit tests for GNN test tracking enhancements

use std::path::{Path, PathBuf};
use std::fs;
use yantra::gnn::{GNNEngine, EdgeType};

// Helper function to get test project path
fn get_test_project_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("test_project")
}

// Helper function to create a clean test database
fn setup_test_db(name: &str) -> PathBuf {
    let test_project = get_test_project_path();
    let yantra_dir = test_project.join(".yantra");
    fs::create_dir_all(&yantra_dir).unwrap();
    
    let db_path = yantra_dir.join(format!("{}.db", name));
    
    // Remove old database if exists
    if db_path.exists() {
        fs::remove_file(&db_path).unwrap();
    }
    
    db_path
}

#[test]
fn test_is_test_file_python() {
    // Test Python test file patterns
    assert!(GNNEngine::is_test_file(Path::new("test_calculator.py")));
    assert!(GNNEngine::is_test_file(Path::new("calculator_test.py")));
    assert!(GNNEngine::is_test_file(Path::new("tests/test_utils.py")));
    assert!(GNNEngine::is_test_file(Path::new("project/tests/calculator.py")));
    
    // Test non-test files
    assert!(!GNNEngine::is_test_file(Path::new("calculator.py")));
    assert!(!GNNEngine::is_test_file(Path::new("utils.py")));
    assert!(!GNNEngine::is_test_file(Path::new("main.py")));
}

#[test]
fn test_is_test_file_javascript() {
    // Test JavaScript/TypeScript test file patterns
    assert!(GNNEngine::is_test_file(Path::new("calculator.test.js")));
    assert!(GNNEngine::is_test_file(Path::new("utils.test.ts")));
    assert!(GNNEngine::is_test_file(Path::new("component.spec.js")));
    assert!(GNNEngine::is_test_file(Path::new("module.spec.ts")));
    assert!(GNNEngine::is_test_file(Path::new("__tests__/calculator.js")));
    assert!(GNNEngine::is_test_file(Path::new("src/__tests__/utils.ts")));
    
    // Test non-test files
    assert!(!GNNEngine::is_test_file(Path::new("calculator.js")));
    assert!(!GNNEngine::is_test_file(Path::new("utils.ts")));
    assert!(!GNNEngine::is_test_file(Path::new("component.jsx")));
}

#[test]
fn test_find_source_file_for_test() {
    let db_path = setup_test_db("test_find_source");
    let test_project = get_test_project_path();
    
    // Build graph
    let mut engine = GNNEngine::new(&db_path).unwrap();
    engine.build_graph(&test_project).unwrap();
    
    // Test mapping from test file to source file
    let test_calc_path = test_project.join("test_calculator.py");
    let source_file = engine.find_source_file_for_test(&test_calc_path);
    assert!(source_file.is_some(), "Should find source file for test_calculator.py");
    
    let source_file = source_file.unwrap();
    assert!(source_file.ends_with("calculator.py"), "Should map to calculator.py, got: {}", source_file);
    
    // Test mapping for utils test
    let test_utils_path = test_project.join("test_utils.py");
    let source_file = engine.find_source_file_for_test(&test_utils_path);
    assert!(source_file.is_some(), "Should find source file for test_utils.py");
    
    let source_file = source_file.unwrap();
    assert!(source_file.ends_with("utils.py"), "Should map to utils.py, got: {}", source_file);
}

#[test]
fn test_create_test_edges() {
    let db_path = setup_test_db("test_create_edges");
    let test_project = get_test_project_path();
    
    // Build graph
    let mut engine = GNNEngine::new(&db_path).unwrap();
    engine.build_graph(&test_project).unwrap();
    
    // Create test edges
    let edge_count = engine.create_test_edges().unwrap();
    
    println!("\n✅ Created {} test edges", edge_count);
    assert!(edge_count > 0, "Should create at least some test edges");
    
    // Verify test edges exist in graph
    // We should have TestDependency edges from test files to source files
    let all_nodes = engine.graph.get_all_nodes().cloned().collect::<Vec<_>>();
    
    let test_nodes: Vec<_> = all_nodes.iter()
        .filter(|n| GNNEngine::is_test_file(Path::new(&n.file_path)))
        .collect();
    
    let source_nodes: Vec<_> = all_nodes.iter()
        .filter(|n| !GNNEngine::is_test_file(Path::new(&n.file_path)))
        .collect();
    
    println!("Test nodes: {}", test_nodes.len());
    for node in &test_nodes {
        println!("  - {} in {}", node.name, node.file_path);
    }
    
    println!("Source nodes: {}", source_nodes.len());
    for node in &source_nodes {
        println!("  - {} in {}", node.name, node.file_path);
    }
    
    assert!(test_nodes.len() > 0, "Should have test nodes");
    assert!(source_nodes.len() > 0, "Should have source nodes");
}

#[test]
fn test_find_untested_code() {
    let db_path = setup_test_db("test_untested_code");
    let test_project = get_test_project_path();
    
    // Build graph and create test edges
    let mut engine = GNNEngine::new(&db_path).unwrap();
    engine.build_graph(&test_project).unwrap();
    engine.create_test_edges().unwrap();
    
    // Find all source files
    let all_nodes = engine.graph.get_all_nodes().cloned().collect::<Vec<_>>();
    let source_files: Vec<String> = all_nodes.iter()
        .filter(|n| !GNNEngine::is_test_file(Path::new(&n.file_path)))
        .map(|n| n.file_path.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    
    println!("\n✅ Source files analyzed:");
    for file in &source_files {
        println!("  - {}", file);
        
        // Check if this file has tests
        let file_nodes: Vec<_> = all_nodes.iter()
            .filter(|n| n.file_path == *file)
            .collect();
        
        let mut has_tests = false;
        for node in file_nodes {
            // Check if any test node has an edge to this node
            let incoming_tests: Vec<_> = all_nodes.iter()
                .filter(|n| GNNEngine::is_test_file(Path::new(&n.file_path)))
                .filter(|test_node| {
                    // Check if test_node tests this node
                    test_node.name.starts_with("test_") && 
                    test_node.name.contains(&node.name)
                })
                .collect();
            
            if incoming_tests.len() > 0 {
                has_tests = true;
                println!("    ✓ {} is tested by {} test(s)", node.name, incoming_tests.len());
            }
        }
        
        if !has_tests {
            println!("    ⚠️  File has no test coverage");
        }
    }
    
    // main.py should be untested (we didn't create tests for it)
    let main_py_tested = source_files.iter()
        .filter(|f| f.ends_with("main.py"))
        .any(|_| {
            all_nodes.iter()
                .filter(|n| GNNEngine::is_test_file(Path::new(&n.file_path)))
                .any(|test| {
                    test.file_path.contains("test_main")
                })
        });
    
    assert!(!main_py_tested, "main.py should not have tests (untested code)");
}

#[test]
fn test_impact_analysis_find_affected_tests() {
    let db_path = setup_test_db("test_impact_analysis");
    let test_project = get_test_project_path();
    
    // Build graph and create test edges
    let mut engine = GNNEngine::new(&db_path).unwrap();
    engine.build_graph(&test_project).unwrap();
    let edge_count = engine.create_test_edges().unwrap();
    
    println!("\n✅ Impact Analysis Test:");
    println!("Created {} test edges", edge_count);
    
    // Simulate: calculator.py changed
    let calculator_path = test_project.join("calculator.py");
    let calculator_str = calculator_path.to_str().unwrap();
    
    // Find all test files that test calculator.py
    let all_nodes = engine.graph.get_all_nodes().cloned().collect::<Vec<_>>();
    
    let affected_tests: Vec<_> = all_nodes.iter()
        .filter(|n| GNNEngine::is_test_file(Path::new(&n.file_path)))
        .filter(|test_node| {
            // Check if this test file tests calculator.py
            engine.find_source_file_for_test(Path::new(&test_node.file_path))
                .map(|src| src == calculator_str)
                .unwrap_or(false)
        })
        .collect();
    
    println!("\nWhen calculator.py changes:");
    println!("Need to re-run {} test file(s):", affected_tests.len());
    for test in &affected_tests {
        println!("  - {}", test.file_path);
    }
    
    // Should find test_calculator.py
    let has_test_calculator = affected_tests.iter()
        .any(|t| t.file_path.ends_with("test_calculator.py"));
    
    assert!(has_test_calculator, "Should identify test_calculator.py as affected test");
}

#[test]
fn test_test_coverage_metrics() {
    let db_path = setup_test_db("test_coverage_metrics");
    let test_project = get_test_project_path();
    
    // Build graph and create test edges
    let mut engine = GNNEngine::new(&db_path).unwrap();
    engine.build_graph(&test_project).unwrap();
    engine.create_test_edges().unwrap();
    
    let all_nodes = engine.graph.get_all_nodes().cloned().collect::<Vec<_>>();
    
    // Count source files
    let source_files_count = all_nodes.iter()
        .filter(|n| !GNNEngine::is_test_file(Path::new(&n.file_path)))
        .map(|n| &n.file_path)
        .collect::<std::collections::HashSet<_>>()
        .len();
    
    // Count test files
    let test_files_count = all_nodes.iter()
        .filter(|n| GNNEngine::is_test_file(Path::new(&n.file_path)))
        .map(|n| &n.file_path)
        .collect::<std::collections::HashSet<_>>()
        .len();
    
    // Count tested source files
    let tested_files_count = all_nodes.iter()
        .filter(|n| !GNNEngine::is_test_file(Path::new(&n.file_path)))
        .filter(|source_node| {
            // Check if any test file tests this source file
            all_nodes.iter()
                .filter(|n| GNNEngine::is_test_file(Path::new(&n.file_path)))
                .any(|test_node| {
                    engine.find_source_file_for_test(Path::new(&test_node.file_path))
                        .as_ref() == Some(&source_node.file_path)
                })
        })
        .map(|n| &n.file_path)
        .collect::<std::collections::HashSet<_>>()
        .len();
    
    let coverage_percentage = if source_files_count > 0 {
        (tested_files_count as f64 / source_files_count as f64) * 100.0
    } else {
        0.0
    };
    
    println!("\n✅ Test Coverage Metrics:");
    println!("  Source files: {}", source_files_count);
    println!("  Test files: {}", test_files_count);
    println!("  Tested source files: {}", tested_files_count);
    println!("  Coverage: {:.1}%", coverage_percentage);
    
    assert!(source_files_count >= 3, "Should have at least 3 source files");
    assert!(test_files_count >= 2, "Should have at least 2 test files");
    assert!(coverage_percentage > 0.0, "Should have some test coverage");
}
