## **Agent Progress Reporting**

A generic, model-accessible tool that allows LLMs to create and maintain one or more progression structures to communicate their work to the user. The model can define arbitrary items, steps or milestones and send updates to modify their descriptions, statuses and progress as the task evolves. The harness displays this information through a compact indicator integrated into the existing TUI status line (e.g. `Current Task: Implementing parser · Step 5/10`), with an expandable view providing more detail. Multiple independent progression structures can coexist within the same session, supporting different tasks, parallel workflows and multi-agent systems.

## **Specification**

### **1. Model-accessible interface**

The model can:

- Create progression structures.
- Add, modify and remove items.
- Update item descriptions, statuses and progress.
- Publish events associated with progression items.
- Maintain multiple structures concurrently.

### **2. Progression structures**

- Completely flexible structure, defined by the model according to the task.
- No predefined workflow or mandatory hierarchy.
- Support steps, milestones, groups or arbitrary items.
- Structures can evolve dynamically during execution.
- Multiple structures can coexist within a session.

### **3. TUI integration**

**Compact view**

Display a short description of the currently relevant activity in the existing harness status line.

Example:

```
Current Task: Implementing parser · Step 5/10
```

**Expanded view**

An optional terminal-inspired interface displaying progression structures, their items, descriptions, statuses and updates.

### **4. General requirements**

- Support multiple concurrent structures.
- Reflect updates in real time.
- Keep the interface lightweight and non-intrusive.
- Allow different presentation layers to consume the same information.
- Remain independent of agent architecture, model and task type.

**Core principle:** The model decides what to report and how to structure its progression; the harness provides the mechanism to display it.


## **Basic Design**

The harness exposes a small set of tools to the model to manage progression structures dynamically.

| Tool                   | Description                                                        |
| ---------------------- | ------------------------------------------------------------------ |
| `create_progress`      | Create a new progression structure with a title and initial items. |
| `update_progress`      | Update progression metadata, such as its title or description.     |
| `add_progress_item`    | Add an item to an existing progression structure.                  |
| `update_progress_item` | Update an item's title, description, status or progress.           |
| `remove_progress_item` | Remove an item from a progression structure.                       |
| `delete_progress`      | Remove an entire progression structure.                            |

### **Example interface**

```
create_progress({
  title: "Implement authentication",
  items: [
    { id: "1", title: "Review existing architecture" },
    { id: "2", title: "Implement authentication flow" },
    { id: "3", title: "Add tests" }
  ]
});

update_progress_item({
  progress_id: "...",
  item_id: "2",
  status: "running",
  description: "Implementing OAuth integration"
});

update_progress_item({
  progress_id: "...",
  item_id: "2",
  status: "completed"
});
```

### **Basic data model**

```
interface Progress {
  id: string;
  title: string;
  description?: string;
  items: ProgressItem[];
}

interface ProgressItem {
  id: string;
  title: string;
  description?: string;
  status?: string;
  progress?: number;
}
```

### **Design considerations**

- **Flexible:** No mandatory workflow, hierarchy or predefined statuses.
- **Concurrent:** Multiple progression structures can coexist within a session.
- **Mutable:** The model can modify the structure at any point.
- **Lightweight:** Tools should be simple and inexpensive to invoke.
- **Presentation-independent:** The harness owns rendering; the model only manages the information.

The proposed interface is intentionally minimal. Additional capabilities, such as nested items, ordering, agent associations and event history, can be introduced later without changing the fundamental concept.