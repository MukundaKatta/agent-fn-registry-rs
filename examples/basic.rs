//! A small end-to-end example of `agent-fn-registry`.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example basic
//! ```
//!
//! It registers two tools, dispatches them, exports their schemas in both
//! Anthropic and OpenAI shapes, and filters the tools by side effect.

use agent_fn_registry::Registry;
use serde_json::{json, Value};

fn main() {
    let mut reg = Registry::new();

    // A pure, read-only tool with a default argument.
    reg.register(
        "echo",
        |args: Value| args.get("msg").cloned().unwrap_or(Value::Null),
        json!({
            "description": "Echo a message back to the caller.",
            "input_schema": {
                "type": "object",
                "properties": { "msg": { "type": "string" } },
                "required": ["msg"]
            }
        }),
        &["read"],
        Some(json!({ "msg": "hello from defaults" })),
    );

    // A compute tool with no defaults.
    reg.register(
        "add",
        |args: Value| {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            json!(a + b)
        },
        json!({
            "description": "Add two numbers.",
            "input_schema": {
                "type": "object",
                "properties": { "a": { "type": "number" }, "b": { "type": "number" } },
                "required": ["a", "b"]
            }
        }),
        &["read", "compute"],
        None,
    );

    println!("registered tools: {:?}", reg.tool_names());

    // Dispatch using caller-supplied args.
    let sum = reg
        .dispatch("add", Some(json!({ "a": 2, "b": 40 })))
        .unwrap();
    println!("add(2, 40)        = {sum}");

    // Dispatch relying on the registered default argument.
    let echoed = reg.dispatch("echo", None).unwrap();
    println!("echo() [default]  = {echoed}");

    // Dispatching an unknown tool yields a typed error.
    match reg.dispatch("missing", None) {
        Ok(_) => unreachable!(),
        Err(err) => println!("dispatch(missing) -> {err}"),
    }

    // Export schemas for the two major provider shapes.
    println!(
        "\nanthropic_tools:\n{}",
        serde_json::to_string_pretty(&reg.anthropic_tools()).unwrap()
    );
    println!(
        "\nopenai_functions:\n{}",
        serde_json::to_string_pretty(&reg.openai_functions()).unwrap()
    );

    // Pick out the tools you might want to gate behind confirmation.
    let compute_tools: Vec<&str> = reg
        .with_side_effect("compute")
        .iter()
        .map(|e| e.name.as_str())
        .collect();
    println!("\ntools with the `compute` side effect: {compute_tools:?}");
}
