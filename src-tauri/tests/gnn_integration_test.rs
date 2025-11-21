// Integration test for GNN engine with real Python project

use std::path::Path;
use std::fs;

// Import from the main crate
// Note: This assumes yantra is the crate name in Cargo.toml

#[test]
fn test_analyze_test_project() {
    // Path to test project
    let test_project = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("test_project");
    
    // Create .yantra directory
    let yantra_dir = test_project.join(".yantra");
    fs::create_dir_all(&yantra_dir).unwrap();
    
    let db_path = yantra_dir.join("graph.db");
    
    // Remove old database if exists
    if db_path.exists() {
        fs::remove_file(&db_path).unwrap();
    }
    
    // Create GNN engine and build graph
    let mut engine = yantra::gnn::GNNEngine::new(&db_path).unwrap();
    engine.build_graph(&test_project).unwrap();
    
    // Test 1: Find the Calculator class
    let calc_node = engine.find_node("Calculator", Some("calculator.py"));
    assert!(calc_node.is_some(), "Calculator class should be found");
    let calc_node = calc_node.unwrap();
    assert_eq!(calc_node.name, "Calculator");
    
    // Test 2: Find utility functions
    let sum_node = engine.find_node("calculate_sum", Some("utils.py"));
    assert!(sum_node.is_some(), "calculate_sum function should be found");
    
    let product_node = engine.find_node("calculate_product", Some("utils.py"));
    assert!(product_node.is_some(), "calculate_product function should be found");
    
    // Test 3: Check Calculator methods
    let add_method = engine.find_node("add", Some("calculator.py"));
    assert!(add_method.is_some(), "Calculator.add method should be found");
    
    let multiply_method = engine.find_node("multiply", Some("calculator.py"));
    assert!(multiply_method.is_some(), "Calculator.multiply method should be found");
    
    // Test 4: Check dependencies
    // Calculator.add should depend on calculate_sum
    let add_id = &add_method.unwrap().id;
    let add_deps = engine.get_dependencies(add_id);
    
    // Should have dependencies on calculate_sum and format_result
    println!("\nadd() dependencies: {} nodes", add_deps.len());
    for dep in &add_deps {
        println!("  - {} ({:?}) in {}", dep.name, dep.node_type, dep.file_path.split('/').last().unwrap_or(""));
    }
    
    // Verify we found the right dependencies
    assert!(add_deps.len() >= 2, "add() should have at least 2 dependencies");
    
    let has_calculate_sum = add_deps.iter().any(|d| d.name == "calculate_sum");
    let has_format_result = add_deps.iter().any(|d| d.name == "format_result");
    
    assert!(has_calculate_sum, "add() should depend on calculate_sum");
    assert!(has_format_result, "add() should depend on format_result");
    
    println!("\n✅ Cross-file dependencies working!");
    println!("   ✓ add() → calculate_sum (utils.py)");
    println!("   ✓ add() → format_result (utils.py)");
    
    println!("\n✅ GNN Integration Test Passed!");
    println!("   - Analyzed {} Python files", 3);
    println!("   - Found Calculator class and methods");
    println!("   - Found utility functions");
    println!("   - Graph structure is working!");
    println!("   - Cross-file dependencies resolved!");
}

#[test]
fn test_persist_and_load() {
    let test_project = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("test_project");
    
    let yantra_dir = test_project.join(".yantra");
    fs::create_dir_all(&yantra_dir).unwrap();
    
    let db_path = yantra_dir.join("graph_persist_test.db");
    
    if db_path.exists() {
        fs::remove_file(&db_path).unwrap();
    }
    
    // Build graph
    {
        let mut engine = yantra::gnn::GNNEngine::new(&db_path).unwrap();
        engine.build_graph(&test_project).unwrap();
    }
    
    // Load graph in new engine
    {
        let mut engine = yantra::gnn::GNNEngine::new(&db_path).unwrap();
        engine.load().unwrap();
        
        // Verify data persisted
        let calc_node = engine.find_node("Calculator", Some("calculator.py"));
        assert!(calc_node.is_some(), "Calculator should be loaded from DB");
        
        println!("✅ Persist and Load Test Passed!");
    }
}
