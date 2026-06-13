# agent-fn-registry

A small, dependency-light Rust crate that keeps everything an LLM agent tool
needs — its **callable**, its **schema**, its **side-effect tags**, and its
**default arguments** — together in a single registry, instead of scattered
across parallel maps and structs.

When you wire tools into an agent loop you usually end up juggling several
parallel structures: one map of name to function, another of name to JSON
schema, a third tracking which tools touch the network or mutate state, and a
fourth holding default arguments. `agent-fn-registry` collapses all of that into
one `Registry` keyed by tool name, and exports the schemas in the shapes the
Anthropic Messages API and OpenAI function-calling API expect.

## Features

- **One entry per tool** — function, schema, side-effect tags, and defaults
  live together in a `ToolEntry`.
- **Dispatch with defaults** — `dispatch` merges registered defaults with
  caller-supplied args (caller args win) before invoking the function.
- **Schema export** — `anthropic_tools()` returns Anthropic-style schema
  objects; `openai_functions()` returns OpenAI `{"type": "function", ...}`
  wrappers. Both are sorted by tool name for stable output.
- **Side-effect filtering** — tag tools with strings like `"read"`,
  `"network"`, or `"write"`, then select them with `with_side_effect` /
  `without_side_effect` (handy for gating mutating tools behind confirmation).
- **Minimal dependencies** — only `serde_json`.

## Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
agent-fn-registry = "0.1"
serde_json = "1"
```

## Usage

```rust
use agent_fn_registry::Registry;
use serde_json::{json, Value};

let mut reg = Registry::new();

reg.register(
    "echo",
    |args: Value| args.get("msg").cloned().unwrap_or(Value::Null),
    json!({
        "name": "echo",
        "description": "Echo a message.",
        "input_schema": {
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "required": ["msg"]
        }
    }),
    &["read"],
    Some(json!({"msg": "hello"})),
);

// Defaults fill in missing keys; caller-supplied args override them.
let result = reg.dispatch("echo", Some(json!({"msg": "world"}))).unwrap();
assert_eq!(result, json!("world"));

let with_default = reg.dispatch("echo", None).unwrap();
assert_eq!(with_default, json!("hello"));

// Export schemas for whichever API you target.
let anthropic = reg.anthropic_tools();
assert_eq!(anthropic[0]["name"], json!("echo"));

let openai = reg.openai_functions();
assert_eq!(openai[0]["type"], json!("function"));
```

### Filtering by side effect

```rust
// Tools that read but never mutate state:
for entry in reg.without_side_effect("write") {
    println!("safe tool: {}", entry.name);
}
```

## API overview

| Method | Purpose |
| --- | --- |
| `register(name, fn, schema, side_effects, defaults)` | Register a tool. The schema's `"name"` field is set to the registry key automatically. |
| `dispatch(name, args)` | Merge defaults with `args` and call the tool. Returns `ToolNotFoundError` if absent. |
| `get` / `get_schema` / `side_effects_of` / `defaults_of` | Inspect a registered tool. |
| `tool_names` | Sorted list of registered names. |
| `anthropic_tools` / `openai_functions` | Bulk schema export in each API's shape. |
| `with_side_effect` / `without_side_effect` | Filter tools by a side-effect tag. |
| `unregister` / `clear` | Remove one or all tools. |
| `has` / `len` / `is_empty` | Registry inspection. |

Lookups that fail return a `ToolNotFoundError` carrying the requested `name`.

## Tech stack

- **Language:** Rust (edition 2021)
- **Dependencies:** [`serde_json`](https://crates.io/crates/serde_json)

## Building and testing

```bash
cargo build
cargo test
```

## License

Licensed under the MIT License.
