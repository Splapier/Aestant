"""Schema viewer UI component.

This module provides a collapsible tree view for the master schema,
allowing users to view and edit leaf keys with their parent hierarchy.
Users can add new keys under lower keys and edit/delete their own additions.
"""

import gradio as gr
from copy import deepcopy

from chatbot.schema_manager import (
    load_master_schema,
    save_master_schema,
    flatten_schema_keys,
    parse_key_path,
    get_key_at_path,
    set_key_at_path,
    delete_key_at_path,
    validate_schema_change,
)


def _build_hierarchical_groups(
    schema: dict,
) -> dict[str, dict[str, dict]]:
    """Group leaf keys by their immediate parent.

    Args:
        schema: The master schema dictionary.

    Returns:
        Dict mapping parent key name to {path: field_def} mappings.
    """
    flat = flatten_schema_keys(schema)
    grouped = {}

    for path, field_def in flat.items():
        path_parts = parse_key_path(path)
        if len(path_parts) >= 2:
            immediate_parent = path_parts[-2]
            if immediate_parent not in grouped:
                grouped[immediate_parent] = {}
            grouped[immediate_parent][path] = field_def

    return grouped


def get_valid_parent_paths(schema: dict) -> list[str]:
    """Get all paths that can have child keys (level 0 or 1 with dict values).

    Args:
        schema: The master schema dictionary.

    Returns:
        List of dot-separated paths that can have children added under them.
    """
    parents = []
    for key, value in schema.items():
        if isinstance(value, dict) and value:
            parents.append(key)
            for child, child_val in value.items():
                if isinstance(child_val, dict) and child_val:
                    parents.append(f"{key}.{child}")
    return parents


def _is_user_added_key(
    schema: dict,
    path: str,
) -> bool:
    """Check if a key was added by the user (i.e., doesn't exist in original structure).

    For simplicity, this checks if a key is deeper than level 1.
    A more sophisticated version would track creation metadata.

    Args:
        schema: The current schema.
        path: Dot-separated path to the key.

    Returns:
        True if potentially user-added (level 2 or deeper).
    """
    path_parts = parse_key_path(path)
    return len(path_parts) >= 2


TYPE_OPTIONS = ["string", "boolean", "array", "number", "object"]

TYPE_DEFAULTS = {
    "string": "",
    "boolean": "false",
    "array": "[]",
    "number": "0",
    "object": "{}",
}


def _render_group_html(
    group_name: str,
    items: dict[str, dict],
    group_expanded: bool = True,
) -> str:
    """Render a collapsible group as HTML.

    Args:
        group_name: Name of the parent group (e.g., "hair", "eyes").
        items: Dict of {path: field_def} for keys under this group.
        group_expanded: Whether the group should start expanded.

    Returns:
        HTML string for the group accordion.
    """
    display_state = "expanded" if group_expanded else "collapsed"
    items_html = ""

    for path, field_def in sorted(items.items()):
        path_parts = parse_key_path(path)
        key_name = path_parts[-1]

        parent_path = " → ".join(path_parts[:-1])
        full_path = " → ".join(path_parts)

        field_type = field_def.get("type", "string")
        default_val = field_def.get("default")
        description = field_def.get("description", "")
        confidence = field_def.get("confidence_score", 0.0)

        can_delete = len(path_parts) >= 2

        type_options = "".join(
            f'<option value="{t}" {"selected" if t == field_type else ""}>{t}</option>'
            for t in TYPE_OPTIONS
        )

        default_str = str(default_val) if default_val is not None else ""

        items_html += f"""
        <div class="schema-item" data-path="{path}">
            <div class="schema-item-header">
                <span class="schema-key-name">{key_name}</span>
                <span class="schema-path">({parent_path})</span>
                {'<button class="delete-btn" onclick="deleteSchemaKey(\'' + path + "')\">🗑️</button>" if can_delete else ""}
            </div>
            <div class="schema-item-details">
                <div class="schema-field">
                    <label>Type:</label>
                    <select class="schema-type" data-path="{path}">{type_options}</select>
                </div>
                <div class="schema-field">
                    <label>Default:</label>
                    <input type="text" class="schema-default" data-path="{path}" value="{default_str}">
                </div>
                <div class="schema-field">
                    <label>Description:</label>
                    <input type="text" class="schema-description" data-path="{path}" value="{description}">
                </div>
                <div class="schema-field">
                    <label>Confidence:</label>
                    <input type="number" step="0.1" min="0" max="1" class="schema-confidence" data-path="{path}" value="{confidence}">
                </div>
            </div>
        </div>
        """

    return f"""
    <div class="schema-group" data-group="{group_name}">
        <details {display_state}>
            <summary class="group-header"><span class="group-header-right"><span>{group_name}</span> <span class="item-count">({len(items)} items)</span></span></summary>
            <div class="group-items">
                {items_html}
            </div>
        </details>
    </div>
    """


def _render_add_key_form(available_parents: list[str] | None = None) -> str:
    """Render the HTML for adding a new key (legacy - now uses Gradio inputs).

    Args:
        available_parents: List of parent paths to include in dropdown. Defaults to dynamic discovery.

    Returns:
        HTML string for the add key form (simplified).
    """
    return ""


_CSS = """
<style>
.schema-viewer {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px;
    max-height: calc(100vh - 200px);
    min-height: 400px;
    overflow-y: auto;
    padding: 8px;
    overflow-x: hidden;
}
.schema-viewer-container {
    min-width: 300px;
}
.schema-group {
    margin-bottom: 8px;
    border: 1px solid #444;
    border-radius: 4px;
    overflow: hidden;
}
.schema-group summary {
    background: #2a2a2a;
    color: #e0e0e0;
    padding: 8px 12px;
    cursor: pointer;
    font-weight: 600;
    user-select: none;
    list-style: none;
}
.schema-group summary::-webkit-details-marker {
    display: none;
}
.schema-group summary::before {
    content: "▶";
    display: inline-block;
    margin-right: 8px;
    font-size: 10px;
    transition: transform 0.2s;
}
.schema-group details[open] summary::before {
    transform: rotate(90deg);
}
.schema-group summary:hover {
    background: #3a3a3a;
}
.group-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.group-header-right {
    display: flex;
    align-items: center;
    gap: 8px;
}
.item-count {
    font-weight: normal;
    color: #999;
    font-size: 12px;
}
.schema-item {
    border-top: 1px solid #444;
    padding: 8px 12px;
    background: #1e1e1e;
}
.schema-item-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.schema-key-name {
    font-weight: 600;
    color: #e0e0e0;
}
.schema-path {
    color: #888;
    font-size: 12px;
}
.schema-item-details {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 4px 8px;
    font-size: 12px;
}
.schema-item-details label {
    color: #999;
    text-align: right;
}
.schema-item-details input,
.schema-item-details select {
    padding: 4px 8px;
    border: 1px solid #555;
    border-radius: 3px;
    font-size: 12px;
    background: #2a2a2a;
    color: #e0e0e0;
}
.schema-item-details input:focus,
.schema-item-details select:focus {
    outline: none;
    border-color: #007bff;
}
.schema-item-details select {
    cursor: pointer;
}
.schema-add-form {
    border: 1px solid #007bff;
    border-radius: 4px;
    padding: 12px;
    background: #1a2a3a;
    margin-top: 12px;
}
.schema-add-form .add-form-header {
    font-weight: 600;
    margin-bottom: 12px;
    color: #4da6ff;
}
.schema-add-form .schema-field {
    margin-bottom: 8px;
}
.schema-add-form label {
    display: block;
    color: #aaccee;
    font-size: 12px;
    margin-bottom: 4px;
}
.schema-add-form input,
.schema-add-form select {
    width: 100%;
    padding: 6px 8px;
    border: 1px solid #446688;
    border-radius: 3px;
    font-size: 12px;
    background: #2a3a4a;
    color: #e0e0e0;
    box-sizing: border-box;
}
.schema-add-form input:focus,
.schema-add-form select:focus {
    outline: none;
    border-color: #007bff;
}
.schema-add-form button {
    margin-top: 8px;
    padding: 8px 16px;
    background: #007bff;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
}
.schema-add-form button:hover {
    background: #0056b3;
}
.schema-add-form button.delete-btn {
    background: #dc3545;
    padding: 4px 8px;
    font-size: 12px;
}
.schema-add-form button.delete-btn:hover {
    background: #bd2130;
}
.schema-item .delete-btn {
    padding: 2px 6px;
    font-size: 11px;
    background: #dc3545;
    color: white;
    border: none;
    border-radius: 3px;
    cursor: pointer;
    margin-left: auto;
}
.schema-item .delete-btn:hover {
    background: #bd2130;
}
#schema-viewer-container details {
    margin-bottom: 4px;
}
#schema-viewer-container details[open] summary {
    border-bottom: 1px solid #444;
}
#schema-viewer-container details[open] {
    background: #1e1e1e;
}
</style>
"""

_INIT_JS = """
<script>
function initSchemaViewer() {
    // Set up auto-default when type changes
    document.addEventListener('change', function(e) {
        if (e.target.classList.contains('new-key-type')) {
            const type = e.target.value;
            const defaults = {string: '', boolean: 'false', array: '[]', number: '0', object: '{}'};
            const defaultInput = e.target.closest('.schema-add-form').querySelector('.new-key-default');
            if (defaultInput && defaults[type] !== undefined) {
                defaultInput.value = defaults[type];
            }
        }
        
        // Also handle existing items
        if (e.target.classList.contains('schema-type')) {
            const type = e.target.value;
            const row = e.target.closest('.schema-item');
            const defaultInput = row.querySelector('.schema-default');
            const defaults = {string: '', boolean: 'false', array: '[]', number: '0', object: '{}'};
            if (defaultInput && defaults[type] !== undefined) {
                defaultInput.value = defaults[type];
            }
        }
    });
    
    // Set up blur handlers for saving fields
    document.addEventListener('change', function(e) {
        if (e.target.classList.contains('schema-default') || 
            e.target.classList.contains('schema-description') ||
            e.target.classList.contains('schema-confidence')) {
            const path = e.target.dataset.path;
            const field = e.target.classList.contains('schema-default') ? 'default' :
                         e.target.classList.contains('schema-description') ? 'description' : 'confidence_score';
            const value = e.target.value;
            // Trigger Gradio update via hidden input
            const hiddenInput = document.getElementById('schema-update-input');
            if (hiddenInput) {
                hiddenInput.value = JSON.stringify({path, field, value});
                hiddenInput.dispatchEvent(new Event('change'));
            }
        }
    });
}
document.addEventListener('DOMContentLoaded', initSchemaViewer);
</script>
"""


def _load_schema_html(schema: dict, available_parents: list[str] | None = None) -> str:
    """Render the full schema viewer as HTML.

    Args:
        schema: The master schema dictionary.
        available_parents: List of parent paths to include in add form.

    Returns:
        Complete HTML string for the schema viewer.
    """
    groups = _build_hierarchical_groups(schema)

    groups_html = ""
    for group_name, items in sorted(groups.items()):
        if items:
            groups_html += _render_group_html(group_name, items)

    if available_parents is None:
        available_parents = get_valid_parent_paths(schema)

    return f"""
<div id="schema-viewer" class="schema-viewer">
    <style>{_CSS}</style>
    <script>{_INIT_JS}</script>
    <div id="schema-viewer-container" class="schema-viewer-container">
        {groups_html if groups_html else "<p>No schema keys defined. Add keys below.</p>"}
    </div>
    {_render_add_key_form(available_parents)}
</div>
"""


def add_schema_key(
    parent_path: str,
    key_name: str,
    field_type: str,
    description: str,
) -> tuple[gr.update, str]:
    """Add a new key to the schema.

    Args:
        parent_path: Parent path (e.g., "head.hair").
        key_name: Name of new key to add.
        field_type: Type of the field (string, boolean, array).
        description: Description of the field.

    Returns:
        Tuple of (html_update, status_message).
    """
    new_field = {
        "type": field_type,
        "default": None,
        "confidence_score": 0.0,
        "description": description,
    }

    full_path = f"{parent_path}.{key_name}"
    allowed, msg = validate_schema_change({}, {}, "add", full_path)

    if not allowed:
        gr.Warning(msg)
        return gr.update(), f"Cannot add: {msg}"

    try:
        schema = load_master_schema()
        set_key_at_path(schema, full_path, new_field)
        save_master_schema(schema)
        gr.Info(f"Added {key_name} under {parent_path}")
        return gr.update(value=_load_schema_html(schema)), f"Added {key_name}"
    except Exception as e:
        gr.Error(f"Failed to add key: {str(e)}")
        return gr.update(), f"Failed: {str(e)}"


def update_schema_field(
    path: str,
    field: str,
    value: str | float,
) -> tuple[gr.update, str]:
    """Update a field value.

    Args:
        path: Dot-separated path to the key.
        field: Field name to update (type, default, description, confidence_score).
        value: New value for the field.

    Returns:
        Tuple of (html_update, status_message).
    """
    try:
        schema = load_master_schema()
        current = get_key_at_path(schema, path)

        if current is None:
            gr.Warning(f"Key not found: {path}")
            return gr.update(), f"Key not found: {path}"

        if field == "confidence_score":
            value = float(value)
        elif field == "default":
            if value == "" or value == "null":
                value = None
            elif current.get("type") == "boolean":
                value = value.lower() in ("true", "1", "yes")

        current[field] = value
        set_key_at_path(schema, path, current)
        save_master_schema(schema)

        return gr.update(value=_load_schema_html(schema)), "Updated"
    except Exception as e:
        gr.Error(f"Failed to update: {str(e)}")
        return gr.update(), f"Failed: {str(e)}"


def delete_schema_key(path: str) -> tuple[gr.update, str]:
    """Delete a key from the schema.

    Args:
        path: Dot-separated path to the key to delete.

    Returns:
        Tuple of (html_update, status_message).
    """
    try:
        schema = load_master_schema()
        path_parts = parse_key_path(path)

        if len(path_parts) <= 1:
            gr.Warning("Cannot delete top-level keys")
            return gr.update(), "Cannot delete protected keys"

        allowed, msg = validate_schema_change(schema, schema, "delete", path)
        if not allowed:
            gr.Warning(msg)
            return gr.update(), msg

        key_name = path_parts[-1]
        deleted = delete_key_at_path(schema, path)

        if deleted:
            save_master_schema(schema)
            gr.Info(f"Deleted {key_name}")
            return gr.update(value=_load_schema_html(schema)), f"Deleted {key_name}"
        else:
            gr.Warning(f"Key not found: {path}")
            return gr.update(), f"Key not found: {path}"
    except Exception as e:
        gr.Error(f"Failed to delete: {str(e)}")
        return gr.update(), f"Failed: {str(e)}"


def refresh_schema_viewer() -> str:
    """Refresh the schema viewer HTML.

    Returns:
        Updated HTML for the schema viewer.
    """
    schema = load_master_schema()
    return _load_schema_html(schema)


def create_schema_viewer_tab() -> tuple[
    gr.HTML,
    gr.Button,
    gr.Dropdown,
    gr.Textbox,
    gr.Dropdown,
    gr.Textbox,
    gr.Button,
    gr.Textbox,
    gr.Button,
]:
    """Create the schema viewer tab component.

    Must be called within a gr.Blocks context.

    Returns:
        Tuple of (schema_viewer_html, refresh_button, add_parent_input, add_key_input, add_type_input, add_desc_input, add_button, delete_path_input, delete_button).
    """
    schema = load_master_schema()
    viewer_html = _load_schema_html(schema)
    parent_choices = get_valid_parent_paths(schema)

    schema_viewer = gr.HTML(
        value=viewer_html,
        label="Schema Viewer",
        elem_id="schema-viewer-output",
    )

    refresh_button = gr.Button(
        "🔄 Refresh",
        variant="secondary",
        size="sm",
    )

    default_parent = parent_choices[0] if parent_choices else None

    with gr.Row():
        add_parent = gr.Dropdown(
            choices=parent_choices,
            label="Parent",
            value=default_parent,
            scale=1,
        )
        add_key = gr.Textbox(
            label="Key Name",
            placeholder="e.g., color, style",
            scale=1,
        )
        add_type = gr.Dropdown(
            choices=TYPE_OPTIONS,
            label="Type",
            value="string",
            scale=1,
        )
        add_desc = gr.Textbox(
            label="Description",
            placeholder="Field description",
            scale=2,
        )
        add_button = gr.Button("➕ Add Key", variant="primary", scale=1)

    with gr.Row():
        delete_path = gr.Textbox(
            label="Delete Key Path",
            placeholder="e.g., head.hairs.my_new_key",
            scale=3,
        )
        delete_button = gr.Button("🗑️ Delete", variant="stop", scale=1)

    refresh_button.click(
        fn=refresh_schema_viewer,
        inputs=None,
        outputs=[schema_viewer],
    )

    def _handle_add(
        parent: str,
        key_name: str,
        field_type: str,
        description: str,
    ) -> tuple[gr.update, str]:
        if not key_name or not key_name.strip():
            return gr.update(), "Please enter a key name"
        return add_schema_key(parent, key_name.strip(), field_type, description)

    add_button.click(
        fn=_handle_add,
        inputs=[add_parent, add_key, add_type, add_desc],
        outputs=[schema_viewer, add_key],
    ).then(
        fn=lambda: ("",),
        outputs=[add_key],
    )

    def _handle_delete(path: str) -> tuple[gr.update, str]:
        if not path or not path.strip():
            return gr.update(), "Please enter a key path"
        return delete_schema_key(path.strip())

    delete_button.click(
        fn=_handle_delete,
        inputs=[delete_path],
        outputs=[schema_viewer, delete_path],
    ).then(
        fn=lambda: ("",),
        outputs=[delete_path],
    )

    return (
        schema_viewer,
        refresh_button,
        add_parent,
        add_key,
        add_type,
        add_desc,
        add_button,
        delete_path,
        delete_button,
    )


__all__ = [
    "create_schema_viewer_tab",
    "add_schema_key",
    "update_schema_field",
    "delete_schema_key",
    "refresh_schema_viewer",
    "get_valid_parent_paths",
]
