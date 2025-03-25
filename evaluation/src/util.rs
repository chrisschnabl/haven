/// Format a header for LLM prompts using the Llama 3 chat format
pub fn format_header(header_name: &str, content: &str) -> String {
    format!("<|start_header_id|>{}<|end_header_id|>{}<|eot_id|>", header_name, content)
}