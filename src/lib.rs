/*!
agent-fn-registry: one registry per LLM agent tool set.

Keeps the callable, schema, side-effect tags, and default args
in one place instead of scattered parallel structures.

```rust
use agent_fn_registry::Registry;
use serde_json::{json, Value};

let mut reg = Registry::new();

reg.register(
    "echo",
    |args: Value| args.get("msg").cloned().unwrap_or(Value::Null),
    json!({"name": "echo", "description": "Echo a message.",
           "input_schema": {"type": "object",
                            "properties": {"msg": {"type": "string"}},
                            "required": ["msg"]}}),
    &["read"],
    Some(json!({"msg": "hello"})),
);

let result = reg.dispatch("echo", Some(json!({"msg": "world"}))).unwrap();
assert_eq!(result, json!("world"));

let tools = reg.anthropic_tools();
assert_eq!(tools[0]["name"], json!("echo"));
```
*/

use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// ---- errors ---------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolNotFoundError {
    pub name: String,
}

impl std::fmt::Display for ToolNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tool not found: {}", self.name)
    }
}

impl std::error::Error for ToolNotFoundError {}

// ---- ToolEntry ------------------------------------------------------------

type ToolFn = Arc<dyn Fn(Value) -> Value + Send + Sync>;

/// One registered tool.
pub struct ToolEntry {
    pub name: String,
    pub schema: Value,
    pub side_effects: HashSet<String>,
    pub defaults: Map<String, Value>,
    fn_: ToolFn,
}

impl ToolEntry {
    /// Call the underlying function with `args` (a JSON object value).
    pub fn call(&self, args: Value) -> Value {
        (self.fn_)(args)
    }
}

impl std::fmt::Debug for ToolEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolEntry")
            .field("name", &self.name)
            .field("side_effects", &self.side_effects)
            .field("defaults", &self.defaults)
            .finish()
    }
}

// ---- Registry -------------------------------------------------------------

/// In-process registry of LLM agent tools.
#[derive(Default)]
pub struct Registry {
    tools: HashMap<String, ToolEntry>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    // ---- registration ----------------------------------------------------

    /// Register a tool.
    ///
    /// `fn_` receives the merged args object (`defaults` overridden by
    /// caller-supplied args) as a `serde_json::Value`.
    ///
    /// `schema` should be an Anthropic `input_schema`-style object with at
    /// least a `"name"` field. If `schema["name"]` is absent, `name` is
    /// inserted automatically.
    ///
    /// `side_effects` is an optional slice of tag strings like `&["read",
    /// "network"]`. `defaults` is an optional JSON object whose keys are
    /// merged in before the caller-supplied args.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        fn_: impl Fn(Value) -> Value + Send + Sync + 'static,
        schema: Value,
        side_effects: &[&str],
        defaults: Option<Value>,
    ) -> &ToolEntry {
        let name = name.into();
        let mut schema = schema;
        // ensure schema["name"] matches the registry key
        if let Some(obj) = schema.as_object_mut() {
            obj.insert("name".to_owned(), Value::String(name.clone()));
        }
        let defaults_map = defaults.and_then(|v| v.into_object()).unwrap_or_default();
        let entry = ToolEntry {
            name: name.clone(),
            schema,
            side_effects: side_effects.iter().map(|s| s.to_string()).collect(),
            defaults: defaults_map,
            fn_: Arc::new(fn_),
        };
        self.tools.insert(name.clone(), entry);
        self.tools.get(&name).unwrap()
    }

    pub fn unregister(&mut self, name: &str) -> bool {
        self.tools.remove(name).is_some()
    }

    pub fn clear(&mut self) {
        self.tools.clear();
    }

    // ---- inspection ------------------------------------------------------

    pub fn has(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Tool names sorted lexicographically.
    ///
    /// Used internally to give every bulk accessor a deterministic order
    /// regardless of the underlying `HashMap` iteration order.
    fn sorted_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tools.keys().map(String::as_str).collect();
        names.sort();
        names
    }

    /// Sorted list of registered tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.sorted_names()
    }

    /// All registered tool entries, sorted by name.
    ///
    /// Handy for inspecting or serializing the whole registry without
    /// looking each tool up by name.
    pub fn entries(&self) -> Vec<&ToolEntry> {
        self.sorted_names()
            .into_iter()
            .map(|n| &self.tools[n])
            .collect()
    }

    pub fn get(&self, name: &str) -> Result<&ToolEntry, ToolNotFoundError> {
        self.tools.get(name).ok_or_else(|| ToolNotFoundError {
            name: name.to_owned(),
        })
    }

    pub fn get_schema(&self, name: &str) -> Result<Value, ToolNotFoundError> {
        Ok(self.get(name)?.schema.clone())
    }

    pub fn side_effects_of(&self, name: &str) -> Result<&HashSet<String>, ToolNotFoundError> {
        Ok(&self.get(name)?.side_effects)
    }

    pub fn defaults_of(&self, name: &str) -> Result<Map<String, Value>, ToolNotFoundError> {
        Ok(self.get(name)?.defaults.clone())
    }

    // ---- dispatch --------------------------------------------------------

    /// Look up the tool by name and call it.
    ///
    /// Defaults are merged in first; caller-supplied keys in `args` win.
    pub fn dispatch(&self, name: &str, args: Option<Value>) -> Result<Value, ToolNotFoundError> {
        let entry = self.get(name)?;
        let mut merged = entry.defaults.clone();
        if let Some(a) = args {
            if let Some(obj) = a.into_object() {
                merged.extend(obj);
            }
        }
        Ok(entry.call(Value::Object(merged)))
    }

    // ---- bulk schema export ----------------------------------------------

    /// All schemas in Anthropic Messages-API shape (plain schema objects).
    pub fn anthropic_tools(&self) -> Vec<Value> {
        self.sorted_names()
            .iter()
            .map(|n| self.tools[*n].schema.clone())
            .collect()
    }

    /// All schemas in OpenAI function-calling shape.
    pub fn openai_functions(&self) -> Vec<Value> {
        self.sorted_names()
            .iter()
            .map(|n| {
                let schema = &self.tools[*n].schema;
                let parameters = schema
                    .get("input_schema")
                    .or_else(|| schema.get("parameters"))
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": schema.get("name").cloned().unwrap_or(Value::Null),
                        "description": schema.get("description").and_then(|v| v.as_str()).unwrap_or(""),
                        "parameters": parameters,
                    }
                })
            })
            .collect()
    }

    // ---- filtering -------------------------------------------------------

    /// All tools that carry `effect` in their side_effects set.
    pub fn with_side_effect(&self, effect: &str) -> Vec<&ToolEntry> {
        let mut entries: Vec<&ToolEntry> = self
            .tools
            .values()
            .filter(|e| e.side_effects.contains(effect))
            .collect();
        entries.sort_by_key(|e| e.name.as_str());
        entries
    }

    /// All tools that do NOT carry `effect` in their side_effects set.
    pub fn without_side_effect(&self, effect: &str) -> Vec<&ToolEntry> {
        let mut entries: Vec<&ToolEntry> = self
            .tools
            .values()
            .filter(|e| !e.side_effects.contains(effect))
            .collect();
        entries.sort_by_key(|e| e.name.as_str());
        entries
    }
}

// Helper: extract Map from Value::Object
trait IntoObject {
    fn into_object(self) -> Option<Map<String, Value>>;
}
impl IntoObject for Value {
    fn into_object(self) -> Option<Map<String, Value>> {
        match self {
            Value::Object(m) => Some(m),
            _ => None,
        }
    }
}

// ---- tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_registry() -> Registry {
        let mut reg = Registry::new();
        reg.register(
            "echo",
            |args| args.get("msg").cloned().unwrap_or(Value::Null),
            json!({"name": "echo", "description": "Echo a message.",
                   "input_schema": {"type": "object", "properties": {"msg": {"type": "string"}}}}),
            &["read"],
            Some(json!({"msg": "default"})),
        );
        reg.register(
            "add",
            |args| {
                let a = args["a"].as_f64().unwrap_or(0.0);
                let b = args["b"].as_f64().unwrap_or(0.0);
                json!(a + b)
            },
            json!({"name": "add", "description": "Add two numbers.",
                   "input_schema": {"type": "object",
                                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}}),
            &["read", "compute"],
            None,
        );
        reg
    }

    #[test]
    fn register_and_has() {
        let reg = make_registry();
        assert!(reg.has("echo"));
        assert!(reg.has("add"));
        assert!(!reg.has("nope"));
    }

    #[test]
    fn len() {
        let reg = make_registry();
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn tool_names_sorted() {
        let reg = make_registry();
        assert_eq!(reg.tool_names(), vec!["add", "echo"]);
    }

    #[test]
    fn entries_sorted_by_name() {
        let reg = make_registry();
        let names: Vec<&str> = reg.entries().iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["add", "echo"]);
    }

    #[test]
    fn entries_empty() {
        let reg = Registry::new();
        assert!(reg.entries().is_empty());
    }

    #[test]
    fn entries_carry_metadata() {
        let reg = make_registry();
        let entries = reg.entries();
        // first entry is "add" (sorted); it has the "compute" side effect.
        assert_eq!(entries[0].name, "add");
        assert!(entries[0].side_effects.contains("compute"));
        // "echo" carries a default arg.
        assert_eq!(entries[1].defaults["msg"], json!("default"));
    }

    #[test]
    fn dispatch_with_args() {
        let reg = make_registry();
        let result = reg.dispatch("echo", Some(json!({"msg": "hello"}))).unwrap();
        assert_eq!(result, json!("hello"));
    }

    #[test]
    fn dispatch_uses_defaults() {
        let reg = make_registry();
        let result = reg.dispatch("echo", None).unwrap();
        assert_eq!(result, json!("default"));
    }

    #[test]
    fn dispatch_args_override_defaults() {
        let reg = make_registry();
        let result = reg
            .dispatch("echo", Some(json!({"msg": "override"})))
            .unwrap();
        assert_eq!(result, json!("override"));
    }

    #[test]
    fn dispatch_not_found() {
        let reg = make_registry();
        let err = reg.dispatch("nope", None).unwrap_err();
        assert_eq!(err.name, "nope");
    }

    #[test]
    fn dispatch_add() {
        let reg = make_registry();
        let result = reg
            .dispatch("add", Some(json!({"a": 3.0, "b": 4.0})))
            .unwrap();
        assert!((result.as_f64().unwrap() - 7.0).abs() < 1e-9);
    }

    #[test]
    fn get_schema() {
        let reg = make_registry();
        let schema = reg.get_schema("echo").unwrap();
        assert_eq!(schema["name"], json!("echo"));
        assert_eq!(schema["description"], json!("Echo a message."));
    }

    #[test]
    fn schema_name_auto_set() {
        let mut reg = Registry::new();
        reg.register("myname", |_| json!(null), json!({}), &[], None);
        let schema = reg.get_schema("myname").unwrap();
        assert_eq!(schema["name"], json!("myname"));
    }

    #[test]
    fn side_effects_of() {
        let reg = make_registry();
        let fx = reg.side_effects_of("add").unwrap();
        assert!(fx.contains("read"));
        assert!(fx.contains("compute"));
    }

    #[test]
    fn defaults_of() {
        let reg = make_registry();
        let d = reg.defaults_of("echo").unwrap();
        assert_eq!(d["msg"], json!("default"));
    }

    #[test]
    fn defaults_of_empty() {
        let reg = make_registry();
        let d = reg.defaults_of("add").unwrap();
        assert!(d.is_empty());
    }

    #[test]
    fn anthropic_tools_sorted() {
        let reg = make_registry();
        let tools = reg.anthropic_tools();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["name"], json!("add"));
        assert_eq!(tools[1]["name"], json!("echo"));
    }

    #[test]
    fn openai_functions_shape() {
        let reg = make_registry();
        let fns = reg.openai_functions();
        assert_eq!(fns[0]["type"], json!("function"));
        assert_eq!(fns[0]["function"]["name"], json!("add"));
        assert!(fns[0]["function"]["parameters"].is_object());
    }

    #[test]
    fn with_side_effect() {
        let reg = make_registry();
        let read_tools = reg.with_side_effect("read");
        assert_eq!(read_tools.len(), 2);
        let compute_tools = reg.with_side_effect("compute");
        assert_eq!(compute_tools.len(), 1);
        assert_eq!(compute_tools[0].name, "add");
    }

    #[test]
    fn without_side_effect() {
        let reg = make_registry();
        let no_compute = reg.without_side_effect("compute");
        assert_eq!(no_compute.len(), 1);
        assert_eq!(no_compute[0].name, "echo");
    }

    #[test]
    fn unregister() {
        let mut reg = make_registry();
        assert!(reg.unregister("echo"));
        assert!(!reg.has("echo"));
        assert_eq!(reg.len(), 1);
        assert!(!reg.unregister("echo")); // already gone
    }

    #[test]
    fn clear() {
        let mut reg = make_registry();
        reg.clear();
        assert!(reg.is_empty());
    }

    #[test]
    fn get_not_found() {
        let reg = make_registry();
        let err = reg.get("nope").unwrap_err();
        assert_eq!(err.name, "nope");
        assert!(err.to_string().contains("nope"));
    }

    #[test]
    fn empty_registry() {
        let reg = Registry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.tool_names(), Vec::<&str>::new());
        assert!(reg.anthropic_tools().is_empty());
        assert!(reg.openai_functions().is_empty());
    }

    #[test]
    fn tool_entry_call_direct() {
        let reg = make_registry();
        let entry = reg.get("echo").unwrap();
        let result = entry.call(json!({"msg": "direct"}));
        assert_eq!(result, json!("direct"));
    }
}
