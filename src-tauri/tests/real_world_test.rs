// Real-World GNN Testing Suite
use std::fs;
use std::path::Path;
use std::time::Instant;
use tempfile::TempDir;
use yantra::gnn::GNNEngine;

fn create_test_project(dir: &Path, files: Vec<(&str, &str)>) -> Vec<String> {
    let mut paths = Vec::new();
    for (path, content) in files {
        let full_path = dir.join(path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&full_path, content).unwrap();
        paths.push(full_path.to_string_lossy().to_string());
    }
    paths
}

fn measure_build_time(engine: &mut GNNEngine, files: &[String]) -> u128 {
    let start = Instant::now();
    for file in files {
        engine.parse_file(Path::new(file)).unwrap();
    }
    start.elapsed().as_millis()
}

#[test]
fn test_simple_python() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("gnn.db");
    let mut engine = GNNEngine::new(&db_path).unwrap();

    let files = create_test_project(
        temp_dir.path(),
        vec![("main.py", "def greet():\n    return 'Hello'\n")]
    );

    let build_time = measure_build_time(&mut engine, &files);
    println!("Simple Python: {}ms, {} nodes", build_time, engine.node_count());
    assert!(build_time < 100);
}

#[test]
fn test_simple_javascript() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("gnn.db");
    let mut engine = GNNEngine::new(&db_path).unwrap();

    let files = create_test_project(
        temp_dir.path(),
        vec![("app.js", "function hello() { return 'hi'; }")]
    );

    let build_time = measure_build_time(&mut engine, &files);
    println!("Simple JS: {}ms, {} nodes", build_time, engine.node_count());
    assert!(build_time < 100);
}

#[test]
fn test_python_classes() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("gnn.db");
    let mut engine = GNNEngine::new(&db_path).unwrap();

    let code = "class Person:\n    def __init__(self, name):\n        self.name = name\n    def greet(self):\n        return self.name\n";
    let files = create_test_project(temp_dir.path(), vec![("person.py", code)]);

    let build_time = measure_build_time(&mut engine, &files);
    println!("Python Classes: {}ms, {} nodes", build_time, engine.node_count());
    assert!(build_time < 100);
    assert!(engine.node_count() >= 2);
}

#[test]
fn test_python_imports() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("gnn.db");
    let mut engine = GNNEngine::new(&db_path).unwrap();

    let files = create_test_project(
        temp_dir.path(),
        vec![
            ("utils.py", "def add(a, b):\n    return a + b\n"),
            ("main.py", "from utils import add\nresult = add(1, 2)\n"),
        ],
    );

    let build_time = measure_build_time(&mut engine, &files);
    println!("Python Imports: {}ms, {} nodes", build_time, engine.node_count());
    assert!(build_time < 150);
    assert!(engine.node_count() >= 2);
}

#[test]
fn test_javascript_classes() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("gnn.db");
    let mut engine = GNNEngine::new(&db_path).unwrap();

    let code = "class User {\n    constructor(name) {\n        this.name = name;\n    }\n    getName() {\n        return this.name;\n    }\n}\n";
    let files = create_test_project(temp_dir.path(), vec![("user.js", code)]);

    let build_time = measure_build_time(&mut engine, &files);
    println!("JS Classes: {}ms, {} nodes", build_time, engine.node_count());
    assert!(build_time < 100);
}

#[test]
fn test_typescript() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("gnn.db");
    let mut engine = GNNEngine::new(&db_path).unwrap();

    let code = "interface Config {\n    port: number;\n}\nclass Server {\n    config: Config;\n    start() {}\n}\n";
    let files = create_test_project(temp_dir.path(), vec![("server.ts", code)]);

    let build_time = measure_build_time(&mut engine, &files);
    println!("TypeScript: {}ms, {} nodes", build_time, engine.node_count());
    assert!(build_time < 100);
}

#[test]
fn test_react_tsx() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("gnn.db");
    let mut engine = GNNEngine::new(&db_path).unwrap();

    let code = "export const Button = ({ onClick }) => {\n    return <button onClick={onClick}>Click</button>;\n};\n";
    let files = create_test_project(temp_dir.path(), vec![("Button.tsx", code)]);

    let build_time = measure_build_time(&mut engine, &files);
    println!("React TSX: {}ms, {} nodes", build_time, engine.node_count());
    assert!(build_time < 100);
}

#[test]
fn test_python_decorators() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("gnn.db");
    let mut engine = GNNEngine::new(&db_path).unwrap();

    let code = "def decorator(func):\n    return func\n@decorator\ndef test():\n    pass\n";
    let files = create_test_project(temp_dir.path(), vec![("deco.py", code)]);

    let build_time = measure_build_time(&mut engine, &files);
    println!("Python Decorators: {}ms, {} nodes", build_time, engine.node_count());
    assert!(build_time < 100);
}

#[test]
fn test_typescript_generics() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("gnn.db");
    let mut engine = GNNEngine::new(&db_path).unwrap();

    let code = "function id<T>(x: T): T { return x; }\nclass Box<T> {\n    value: T;\n    get(): T { return this.value; }\n}\n";
    let files = create_test_project(temp_dir.path(), vec![("gen.ts", code)]);

    let build_time = measure_build_time(&mut engine, &files);
    println!("TS Generics: {}ms, {} nodes", build_time, engine.node_count());
    assert!(build_time < 100);
}

#[test]
fn test_large_python_module() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("gnn.db");
    let mut engine = GNNEngine::new(&db_path).unwrap();

    let mut code = String::new();
    for i in 0..20 {
        code.push_str(&format!("def func_{}(x):\n    return x * {}\n", i, i));
        code.push_str(&format!("class Class{}:\n    def method(self):\n        return {}\n", i, i));
    }

    let files = create_test_project(temp_dir.path(), vec![("large.py", &code)]);

    let build_time = measure_build_time(&mut engine, &files);
    let loc = code.lines().count();
    println!("Large Python: {} LOC, {}ms, {} nodes", loc, build_time, engine.node_count());
    
    // Project to 10K LOC
    let time_per_loc = build_time as f64 / loc as f64;
    let projected_10k = time_per_loc * 10_000.0;
    println!("  Projected 10K LOC: {:.0}ms ({:.2}s)", projected_10k, projected_10k / 1000.0);
    
    assert!(projected_10k < 5_000.0, "Should meet <5s target for 10K LOC");
}

#[test]
fn test_incremental_update() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("gnn.db");
    let mut engine = GNNEngine::new(&db_path).unwrap();

    let original = "def func1():\n    pass\n";
    let files = create_test_project(temp_dir.path(), vec![("test.py", original)]);
    
    // Initial build
    measure_build_time(&mut engine, &files);

    // Incremental update
    let modified = "def func1():\n    pass\ndef func2():\n    pass\n";
    fs::write(&files[0], modified).unwrap();
    
    let start = Instant::now();
    engine.parse_file(Path::new(&files[0])).unwrap();
    let update_time = start.elapsed().as_micros();

    println!("Incremental update: {}µs ({:.2}ms)", update_time, update_time as f64 / 1000.0);
    assert!(update_time < 50_000, "Should be <50ms (was {}µs)", update_time);
}

#[test]
fn test_edge_case_empty_file() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("gnn.db");
    let mut engine = GNNEngine::new(&db_path).unwrap();

    let files = create_test_project(temp_dir.path(), vec![("empty.py", "")]);
    let result = engine.parse_file(Path::new(&files[0]));
    assert!(result.is_ok(), "Should handle empty files");
    println!("Empty file: {} nodes", engine.node_count());
}

#[test]
fn test_edge_case_comments_only() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("gnn.db");
    let mut engine = GNNEngine::new(&db_path).unwrap();

    let files = create_test_project(temp_dir.path(), vec![("comments.py", "# Comment\n# Another\n")]);
    let result = engine.parse_file(Path::new(&files[0]));
    assert!(result.is_ok(), "Should handle comment-only files");
    println!("Comments only: {} nodes", engine.node_count());
}

#[test]
fn test_edge_case_unicode() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("gnn.db");
    let mut engine = GNNEngine::new(&db_path).unwrap();

    let code = "def greet():\n    return \"Hello 世界 🌍\"\n";
    let files = create_test_project(temp_dir.path(), vec![("unicode.py", code)]);
    let result = engine.parse_file(Path::new(&files[0]));
    assert!(result.is_ok(), "Should handle unicode");
    println!("Unicode: {} nodes", engine.node_count());
}

#[test]
fn test_mixed_languages() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("gnn.db");
    let mut engine = GNNEngine::new(&db_path).unwrap();

    let files = create_test_project(
        temp_dir.path(),
        vec![
            ("script.py", "def py_func(): pass"),
            ("module.js", "function jsFunc() {}"),
            ("comp.tsx", "export const C = () => <div />;"),
        ],
    );

    for file in &files {
        let result = engine.parse_file(Path::new(file));
        assert!(result.is_ok(), "Should handle: {}", file);
    }

    println!("Mixed languages: {} nodes total", engine.node_count());
    assert!(engine.node_count() >= 1, "Should have at least 1 node");
}
