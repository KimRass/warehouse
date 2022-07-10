# "keybindings.json"
```json
// Place your key bindings in this file to override the defaults
[
    {
        "key": "shift+cmd+b",
        "command": "workbench.action.toggleSidebarVisibility"
    },
    {
        "key": "shift+cmd+'",
        "command": "workbench.action.terminal.toggleTerminal",
        "when": "terminal.active"
    },
    {
        "key": "ctrl+`",
        "command": "-workbench.action.terminal.toggleTerminal",
        "when": "terminal.active"
    },
    {
        "key": "shift+cmd+down",
        "command": "workbench.action.terminal.resizePaneDown",
        "when": "terminalFocus && terminalHasBeenCreated || terminalFocus && terminalProcessSupported"
    },
    {
        "key": "shift+cmd+left",
        "command": "workbench.action.terminal.resizePaneLeft",
        "when": "terminalFocus && terminalHasBeenCreated || terminalFocus && terminalProcessSupported"
    },
    {
        "key": "shift+cmd+right",
        "command": "workbench.action.terminal.resizePaneRight",
        "when": "terminalFocus && terminalHasBeenCreated || terminalFocus && terminalProcessSupported"
    },
    {
        "key": "shift+cmd+up",
        "command": "workbench.action.terminal.resizePaneUp",
        "when": "terminalFocus && terminalHasBeenCreated || terminalFocus && terminalProcessSupported"
    }
]
```