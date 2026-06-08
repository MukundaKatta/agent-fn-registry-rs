//! Integration tests that exercise `agent-fn-registry` through its public API,
//! the same way a downstream crate would consume it.

use agent_fn_registry::{Registry, ToolEntry, ToolNotFoundError};
use serde_json::{json, Value};

fn build() -> Registry {
    let mut reg = Registry::new();
    reg.register(
        "echo",
        |args: Value| args.get("msg").cloned().unwrap_or(Value::Null),
        json!({
            "description": "Echo a message.",
            "input_schema": {
                "type": "object",
                "properties": { "msg": { "type": "string" } },
                "required": ["msg"]
            }
        }),
        &["read"],
        Some(json!({ "msg": "default" })),
    );
    reg.register(
        "write_file",
        |args: Value| json!({ "wrote": args.get("path").cloned().unwrap_or(Value::Null) }),
        json!({
            "description": "Write a file.",
            "input_schema": {
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }
        }),
        &["write", "fs"],
        None,
    );
    reg
}

#[test]
fn full_round_trip() {
    let reg = build();

    // Names come out sorted.
    assert_eq!(reg.tool_names(), vec!["echo", "write_file"]);
    assert_eq!(reg.len(), 2);
    assert!(!reg.is_empty());

    // Dispatch with an override.
    let out = reg.dispatch("echo", Some(json!({ "msg": "hi" }))).unwrap();
    assert_eq!(out, json!("hi"));

    // Dispatch falling back to the registered default.
    let out = reg.dispatch("echo", None).unwrap();
    assert_eq!(out, json!("default"));

    // Dispatch a tool that consumes a real argument.
    let out = reg
        .dispatch("write_file", Some(json!({ "path": "/tmp/x" })))
        .unwrap();
    assert_eq!(out, json!({ "wrote": "/tmp/x" }));
}

#[test]
fn dispatch_unknown_tool_is_typed_error() {
    let reg = build();
    let err: ToolNotFoundError = reg.dispatch("nope", None).unwrap_err();
    assert_eq!(err.name, "nope");
    assert!(err.to_string().contains("nope"));
}

#[test]
fn schema_name_is_injected_from_key() {
    let reg = build();
    // The registrant did not set "name" in the schema; the registry fills it in.
    let schema = reg.get_schema("echo").unwrap();
    assert_eq!(schema["name"], json!("echo"));
    let schema = reg.get_schema("write_file").unwrap();
    assert_eq!(schema["name"], json!("write_file"));
}

#[test]
fn anthropic_and_openai_exports_agree_on_names() {
    let reg = build();

    let anthropic = reg.anthropic_tools();
    let openai = reg.openai_functions();
    assert_eq!(anthropic.len(), openai.len());

    let anthropic_names: Vec<&Value> = anthropic.iter().map(|s| &s["name"]).collect();
    let openai_names: Vec<&Value> = openai.iter().map(|f| &f["function"]["name"]).collect();
    assert_eq!(anthropic_names, openai_names);

    // OpenAI wrapper carries the schema's input_schema as `parameters`.
    let echo_fn = &openai[0]; // "echo" sorts first
    assert_eq!(echo_fn["type"], json!("function"));
    assert!(echo_fn["function"]["parameters"]["properties"]["msg"].is_object());
}

#[test]
fn side_effect_filtering_partitions_the_registry() {
    let reg = build();

    let writers: Vec<&str> = reg
        .with_side_effect("write")
        .iter()
        .map(|e| e.name.as_str())
        .collect();
    assert_eq!(writers, vec!["write_file"]);

    let non_writers: Vec<&str> = reg
        .without_side_effect("write")
        .iter()
        .map(|e| e.name.as_str())
        .collect();
    assert_eq!(non_writers, vec!["echo"]);
}

#[test]
fn entries_expose_full_metadata_sorted() {
    let reg = build();
    let entries: Vec<&ToolEntry> = reg.entries();
    assert_eq!(entries.len(), 2);

    assert_eq!(entries[0].name, "echo");
    assert!(entries[0].side_effects.contains("read"));
    assert_eq!(entries[0].defaults["msg"], json!("default"));

    assert_eq!(entries[1].name, "write_file");
    assert!(entries[1].side_effects.contains("write"));
    assert!(entries[1].defaults.is_empty());
}

#[test]
fn unregister_and_clear() {
    let mut reg = build();
    assert!(reg.unregister("echo"));
    assert!(!reg.has("echo"));
    assert_eq!(reg.tool_names(), vec!["write_file"]);

    reg.clear();
    assert!(reg.is_empty());
    assert!(reg.anthropic_tools().is_empty());
}

#[test]
fn direct_entry_call_bypasses_default_merge() {
    let reg = build();
    let entry = reg.get("echo").unwrap();
    // Calling the entry directly does NOT merge defaults; an empty object
    // yields Null because "msg" is absent.
    assert_eq!(entry.call(json!({})), Value::Null);
    assert_eq!(entry.call(json!({ "msg": "x" })), json!("x"));
}
