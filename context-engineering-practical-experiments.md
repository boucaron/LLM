# Context Engineering: Practical Experiments

An LLM has a **context**: the working information available to it while performing a task.

The **context is a scarce resource**. It has a limited capacity, and the larger it becomes, the more expensive and difficult it is to process effectively. You cannot simply keep ingesting more and more data.

The **core problem** is that a context has a finite capacity. As it approaches its limit, the system must compact, remove, or otherwise manage information to continue. In practice, performance can also degrade well before the hard limit because the model has to reason over increasingly large amounts of information.

There is a second, more subtle problem: **more context is not necessarily better context**.

Irrelevant or weakly relevant data increases the amount of information the model has to reason through. It can introduce ambiguities, conflicting instructions, misleading examples, or simply distract the model from the actual task. This applies both to the data you provide and to the prompts themselves: prompts can contain subtle ambiguities or overlapping rules that create conflicts in the model's reasoning.

In an agentic workflow this becomes even more important. An agent accumulates tool calls, files, intermediate results, decisions, errors, and observations over time. The context can grow continuously.

**Compaction** is therefore commonly used to summarize an existing context as it grows, in order to make room for further work. It is useful, but it is still a form of information loss: a large execution history has to be compressed into a smaller representation.

This suggests another strategy:

> **Instead of continuously making one context smaller, avoid putting unnecessary information into it in the first place.**

## What is Context Engineering ?

> **Context engineering is not only about fitting more information into a context; it is about deciding which information should exist in a context at all.**

## Core Context Engineering Techniques

Before looking at agentic workflows, it is useful to look at what can already be done at the chat level for instance. These techniques provide some useful principles for organizing context.

### Context as a working set

> **A context should be treated as a working set, not a database.**

A database stores information because it may be needed later.

A working set contains information because it is needed **now**.

For a llm/agent:

```
Persistent state
       │
       ▼
  select working set
       │
       ▼
    Context
       │
       ▼
     reason
       │
       ▼
   update state
       │
       ▼
  discard working set
```

This gives you a much stronger framing than "context window management."

The problem isn't:

> How do we fit the database into the context?

It's:

> **How do we construct the smallest useful working set for this operation?**

### Context selection / filtering

Before putting information into a context, decide whether it actually needs to be there.

Instead of:

```
Available information
        ↓
   Stuff everything
        ↓
     Context
```

use:

```
Available information
        ↓
  Select / filter
        ↓
Relevant information
        ↓
     Context
```

This includes:

* relevance filtering;
* deduplication;
* removing stale information;
* removing superseded observations;
* excluding information that is only potentially useful;
* selecting only the fields of a document that matter;
* filtering tool results before passing them to the model.

This is perhaps the most fundamental context-engineering technique because it prevents context growth **before it happens**.

A useful principle is:

> **Do not ask "Can this information fit?" Ask "Does this information need to exist here?"**

## Progressive disclosure

Don't provide information at the beginning merely because it *might* become useful later.

Instead, expose information only when the agent reaches the point where it needs it.

```
                    Available knowledge
                           │
                           │
                    ┌──────▼──────┐
                    │   Agent     │
                    └──────┬──────┘
                           │
                    "I need X"
                           │
                           ▼
                    Retrieve X
                           │
                           ▼
                       Context
```

This is especially powerful for:

* large codebases;
* documentation;
* databases;
* file systems;
* tool catalogs;
* APIs;
* knowledge bases.

The llm doesn't need the entire repository in its context. It needs enough information to **navigate toward the next relevant information**.

This creates an important distinction:

> **Context should often contain pointers to information rather than the information itself.**

For example:

```
Repository
    ↓
file index
    ↓
relevant files
    ↓
relevant functions
    ↓
relevant lines
```

rather than:

```
Repository → entire repository in context
```

## Context compression / representation transformation

There are actually several different operations:

### Summarization

```
100 observations
      ↓
5 observations
```

### Extraction

```
Large document
      ↓
Relevant facts
```

### Structuring

```
Unstructured conversation
      ↓
JSON / table / state representation
```

### Compilation

```
Long instructions
      ↓
Self-contained task specification
```

### Canonicalization

```
Different descriptions
      ↓
One normalized representation
```

These are not all "summarization."

A particularly important idea is:

> **The best representation of information depends on the operation that will consume it.**

For example, an LLM doesn't necessarily need:

```
"The user previously said X, then later clarified Y..."
```

It may need:

```
constraints:
  - X
  - Y

decision:
  - use Y when X and Y conflict
```

That's **representation engineering**, not merely compression.

## Fork is key

> **Fork** creates a new conversation or execution context from a given point, allowing work to continue independently without accumulating that work in the original context.

The important property is that you can continue working from that point **without polluting the original context**.

A fork can be used in several ways:

* explore a new hypothesis or direction;
* perform an ephemeral task and discard the result;
* perform a deeper analysis without accumulating it in the original conversation;
* generate a summary or prompt that will be used to perform another task;
* experiment with an alternative solution.

### Fork as exploration

Suppose you ask an LLM to analyze a dataset and identify several risks.

You now want to investigate each risk in much more detail, using additional data.

A common approach is to continue in the same conversation:

```text
Initial analysis
    ↓
Risk A → add data → analyze
    ↓
Risk B → add data → analyze
    ↓
Risk C → add data → analyze
    ↓
...
```

The problem is that the context progressively accumulates the work performed for every previous risk.

Instead, fork the conversation at the point where the relevant information is available:

```text
                  Initial analysis
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           Fork A      Fork B      Fork C
           Risk A      Risk B      Risk C
```

Each exploration gets its own context.

When Risk A is finished, you decide whether its result is useful:

* if not, discard the fork;
* if useful, extract the relevant result;
* then move to Risk B using another fork.

The important point is that the detailed exploration of Risk A does **not** become part of the context used for Risk B.

A natural approach is to perform these explorations sequentially in the same conversation. Forking is often simpler and more efficient because it prevents context from accumulating information that is no longer needed.

**The key point: you did not stockpile the exploration in the main context.**

## Context boundaries

Forking is one form of a broader concept: deliberately establishing **context boundaries**.

You can create boundaries around:

* users;
* tasks;
* subproblems;
* projects;
* agents;
* tools;
* experiments;
* security domains;
* time periods.

For example:

```
             Parent context
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
   Research      Coding      Planning
    context      context      context
       │           │           │
       └───────────┼───────────┘
                   ▼
                Results
```

The boundary prevents accidental information flow.

This has two benefits:

1. **context efficiency**
2. **information isolation**

The second is important enough that it deserves its own principle:

> **Context boundaries are also correctness boundaries.**

Irrelevant information isn't merely expensive. It can cause the model to make the wrong decision.

## Task specification, isolated execution and merge

Sometimes a fork is still not enough.

Suppose the original analysis tells you that you need to perform a large transformation on many work items, but each work item contains too much data to process all of them within the existing context.

You can use a fork to create a **task specification**.

The fork can be asked to produce a compact, self-contained description containing:

* the objective;
* the relevant rules and constraints;
* the transformation to perform;
* the validation criteria;
* any other information required to execute the task independently.

The result is effectively a **compiled task context** for the next stage.

Then, instead of continuing in the original conversation, create a fresh conversation for each work item:

```text
Task specification
       │
       ├──── Work item A → fresh context → result
       ├──── Work item B → fresh context → result
       ├──── Work item C → fresh context → result
       └──── Work item D → fresh context → result
```

Each context contains only:

```text
task specification
+
work item
+
required local information
```

The LLM performs the work, validates the result, and the context can then be discarded.

You move to the next work item with another fresh context.

Once all work items are finished—whether successfully or with failures—you return to the original conversation.

In this workflow, the human is performing **the merge** at this point.

The merge does not mean copying the worker conversations back into the original context. It means collecting their **outcomes, state changes, and failures** so that the parent workflow can continue.

For example:

```text
Original conversation
        │
        ├── Work A → success
        ├── Work B → success
        ├── Work C → failure
        └── Work D → success
        │
        ▼
     Merge state
        │
        ▼
Original conversation continues
```

The original context remains small because it never contained the detailed execution of A, B, C, and D.

### This is a fundamental scalability technique

The important property is not merely that the individual conversations are smaller.

It is that the amount of **working context** required for one independent work item does not need to grow with the number of other work items.

If you have 10, 100, or 10,000 independent work items, each worker can still operate with approximately the same local context:

```text
                    N work items
                         │
       ┌─────────────────┼─────────────────┐
       ▼                 ▼                 ▼
   Context A         Context B         Context C
   task + A          task + B          task + C
       │                 │                 │
       ▼                 ▼                 ▼
     result             result             result
```

This also means that the same runtime and the same model can potentially process the work more efficiently, because each inference operates on a much smaller working context.

This does not necessarily reduce the total number of tokens processed: the same information may still need to be processed across all workers. The gain is that each individual inference operates on a much smaller and more relevant working context, which can improve latency, memory usage, reasoning quality, and model scalability.

A larger model is not necessarily the solution to a large-context problem.

**Reducing the context required by each operation can be a scalability technique in its own right.**

And once this is implemented by an agent runtime rather than manually by a human, the same mechanism can support **fork → execute → validate → discard → merge** automatically, including parallel execution when appropriate.

> **Forking prevents irrelevant work from polluting a context. Fresh contexts allow individual work items to scale independently. The parent context becomes an orchestration context rather than an execution log.**

## Result passing instead of history passing

This is closely related to the `merge` concept, but I think it's worth making explicit.

When one context hands work to another, don't transfer its conversation.

Transfer its **output contract**.

Bad:

```
Worker B receives:
    entire Worker A conversation
```

Better:

```
Worker B receives:
    Worker A result
    + relevant state
    + unresolved issues
```

Even better:

```
Worker A
   ↓
validated artifact
   ↓
structured result
   ↓
Worker B
```

This makes agents behave more like composable functions:

```
input → operation → validated output
```

rather than:

```
conversation → conversation → conversation → ...
```

Even better, you can extend it with the description of the task, the requirements and so on.

## Information lifecycle management

This is probabyl something missing and really important concept.

Every piece of context can have a lifecycle:

```
Created
  ↓
Relevant
  ↓
Active
  ↓
Superseded
  ↓
Archived / discarded
```

Instead of treating context as a static collection of messages, treat information as having **state**.

For example:

```
Hypothesis A
    ↓
supported
    ↓
active

Hypothesis B
    ↓
contradicted
    ↓
discard

Decision X
    ↓
superseded by Decision Y
    ↓
replace
```

This gives you another technique:

### Supersession

When new information invalidates old information, don't retain both indefinitely.

```
old assumption
      ↓
new evidence
      ↓
new assumption
      ↓
replace old assumption
```

Otherwise the model has to reason about which version is authoritative.

# Context Engineering in Agent Systems

## Externalized state

Not all information needs to live in the context window.

Some information should live in an external state store and be retrieved when required.

```
                 ┌───────────────┐
                 │ External state│
                 │               │
                 │ files         │
                 │ database      │
                 │ artifacts     │
                 │ memory        │
                 │ checkpoints   │
                 └───────┬───────┘
                         │
                    retrieve
                         │
                         ▼
                    ┌─────────┐
                    │ Context │
                    └─────────┘
```

This is different from simply having a larger context.

The context becomes a **working set**, while the external store becomes the **source of persistent state**.

For example, an agent could maintain:

```
context:
    current objective
    current hypothesis
    current task
    relevant evidence
    next action

external state:
    complete execution history
    files
    intermediate artifacts
    previous results
    structured state
```

This resembles the distinction between **RAM and persistent storage** in conventional computing.

## Hierarchical context

Not all context needs the same level of detail.

You can represent information at multiple resolutions:

```
Level 0 — existence
    ↓
Level 1 — summary
    ↓
Level 2 — structured details
    ↓
Level 3 — raw information
```

For example:

```
Project
  ↓
Module summary
  ↓
Function description
  ↓
Function implementation
  ↓
Specific lines
```

The agent can descend the hierarchy only when necessary.

This is essentially **progressive refinement**.

It avoids the common failure mode of giving the model:

```
10,000 lines of source code
```

when the model initially only needed:

```
This module handles authentication.
```

## Validation before context promotion

Following the previous section, this one particularly good technique to distinguish between:

**information observed by a worker**

and

**information trusted by the system**.

For example:

```
Tool result
    ↓
Worker interpretation
    ↓
Validation
    ↓
Canonical state
    ↓
Promote to parent context
```

Instead of letting every intermediate observation enter the global context, only validated information is promoted.

This prevents the parent context from becoming polluted by:

* speculation;
* failed hypotheses;
* transient errors;
* contradictory observations;
* malformed tool output.

You could call this **context promotion**.

It fits beautifully with your fork → execute → validate → merge architecture.

## Compaction Alternatives

Context compaction is generally used to *summarize the current context*. However, I think this is often the wrong abstraction for long-running agentic tasks. A summary tries to preserve information from the conversation, while what we actually need is to preserve the **state of the work**.

I started using a simple alternative that I inject manually into my agent when I know the context is about to run out:

```text
It is time to finish; you are out of context.
Summarize what has been completed, what remains to be done, 
and provide a one-line prompt to continue the work in a new session.
```

It is deliberately simple, but I find it works surprisingly well.

Instead of attempting to compress the entire conversation, it asks the model to produce a short **handoff** containing:

* what has already been done;
* what remains;
* important decisions and constraints;
* anything that needs to be retrieved from external files or other persistent state;
* and, most importantly, a very short prompt that can be used to start the next session.

This also works particularly well when the agent has been instructed beforehand to persist important information in external files. At the handoff stage, the model will often finish or update those files and include hints about which files the next session should inspect.

So rather than thinking of this as **context compaction**, I think it is better understood as **task checkpointing** or **state handoff**.

The distinction is important:

> **Compaction tries to preserve the conversation.
> A handoff tries to preserve the ability to continue the work.**

For long-running agents, the second may be a much more useful objective.

This prompt is probably a bit better:

```plaintext
Context is ending. 
Create a continuation handoff: 
state what is completed, 
what remains, key decisions/constraints, 
and any files or external state to inspect. 
Then give a single short prompt that can resume 
the work in a new session 
without repeating completed steps.
```

> The objective of long-running context management is not to preserve history; it is to preserve sufficient state for correct continuation.

# ChangeLog

20/08/2026: First Draft and various iterations
18/09/2026: Add Handoff as alternative to Compaction
