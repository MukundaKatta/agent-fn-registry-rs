# agent-fn-registry

[![CI](https://github.com/MukundaKatta/agent-fn-registry-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/agent-fn-registry-rs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/agent-fn-registry.svg)](https://crates.io/crates/agent-fn-registry)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

One registry per LLM agent tool set.

When you give a tool to an LLM agent you end up tracking four things that
must stay in sync: the **callable** that actually runs, the **JSON schema**
the model sees, the **side-effect tags** you use to gate or audit the call,
and the **default arguments** to fill in. `agent-fn-registry` keeps all four
in one place instead of in scattered parallel maps, and exports the schemas in
both Anthropic Messages-API and OpenAI function-calling shapes.

It is a small, dependency-light library (only `serde_json`) with no async,
no global state, and no macros.

## Install

Add it to your `Cargo.toml`:

```toml
[dependencies]
agent-fn-registry = "0.1"
serde_json = "1"
```

## Quick start

```rust
use agent_fn_registry::Registry;
use serde_json::{json, Value};

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
    Some(json!({ "msg": "hello" })),
);

// Caller args win over the registered defaults.
let result = reg.dispatch("echo", Some(json!({ "msg": "world" }))).unwrap();
assert_eq!(result, json!("world"));

// Fall back to the registered default when no args are supplied.
let result = reg.dispatch("echo", None).unwrap();
assert_eq!(result, json!("hello"));

// Hand the whole tool set straight to the model.
let tools = reg.anthropic_tools();
assert_eq!(tools[0]["name"], json!("echo"));
```

A complete, runnable version of this lives in [`examples/basic.rs`](examples/basic.rs):

```sh
cargo run --example basic
```

## How it works

Each call to [`register`](#registry) stores a [`ToolEntry`] holding:

- **`name`** — the registry key. It is also written into `schema["name"]`
  automatically, so you never have to repeat it.
- **`schema`** — an Anthropic `input_schema`-style object. If `schema["name"]`
  is missing it is filled in from the key.
- **`side_effects`** — a set of free-form tags such as `read`, `write`,
  `network`, used to filter or gate tools (see
  [`with_side_effect`](#filtering)).
- **`defaults`** — a JSON object merged in before the caller's args on every
  dispatch.

`dispatch` performs a **shallow merge**: it starts from `defaults`, then applies
the caller-supplied keys on top (caller wins), and passes the merged object to
the callable. Calling [`ToolEntry::call`] directly bypasses the default merge.

## API

### Registry

| Method | Description |
| --- | --- |
| `Registry::new()` | Create an empty registry. |
| `register(name, fn, schema, side_effects, defaults)` | Register a tool; returns the stored `&ToolEntry`. |
| `unregister(name) -> bool` | Remove a tool; `true` if it existed. |
| `clear()` | Remove all tools. |
| `dispatch(name, args) -> Result<Value, ToolNotFoundError>` | Merge defaults with `args` and call the tool. |

### Inspection

| Method | Description |
| --- | --- |
| `has(name) -> bool` | Whether a tool is registered. |
| `len()` / `is_empty()` | Number of registered tools. |
| `tool_names() -> Vec<&str>` | All names, sorted. |
| `entries() -> Vec<&ToolEntry>` | All entries, sorted by name. |
| `get(name) -> Result<&ToolEntry, ToolNotFoundError>` | Look up one entry. |
| `get_schema(name)` | Clone of the tool's schema. |
| `side_effects_of(name)` | The tool's side-effect set. |
| `defaults_of(name)` | Clone of the tool's default args. |

### Schema export

| Method | Description |
| --- | --- |
| `anthropic_tools() -> Vec<Value>` | Schemas in Anthropic Messages-API shape, sorted by name. |
| `openai_functions() -> Vec<Value>` | Schemas wrapped in OpenAI function-calling shape, sorted by name. |

### Filtering

| Method | Description |
| --- | --- |
| `with_side_effect(effect) -> Vec<&ToolEntry>` | Tools carrying `effect`, sorted by name. |
| `without_side_effect(effect) -> Vec<&ToolEntry>` | Tools lacking `effect`, sorted by name. |

All bulk accessors return results in deterministic, name-sorted order so output
is stable across runs regardless of internal `HashMap` ordering.

### Errors

`dispatch`, `get`, `get_schema`, `side_effects_of`, and `defaults_of` return
`ToolNotFoundError { name }` when the requested tool is not registered. It
implements `std::error::Error` and `Display` (`"tool not found: <name>"`).

## Development

```sh
cargo build
cargo test            # unit + integration + doc tests
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo run --example basic
```

## License

Licensed under the [MIT License](LICENSE).
