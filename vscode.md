# "keybindings.json"
```json
// Place your key bindings in this file to override the defaults
[
    {
        "key": "shift+cmd+b",
        "command": "workbench.action.toggleSidebarVisibility"
    },
    {
        "key": "ctrl+right",
        "command": "cursorWordEndRight",
        "when": "editorTextFocus"
    },
    {
        "key": "ctrl+left",
        "command":"cursorWordEndLeft",
        "when":"editorTextFocus"
    },
    {
        "key": "ctrl+shift+right",
        "command": "cursorWordEndRightSelect",
        "when": "editorTextFocus"
    },
    {
        "key": "ctrl+shift+left",
        "command": "cursorWordEndLeftSelect",
        "when": "editorTextFocus"
    }
]
```