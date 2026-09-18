# Context Engineering Primitives

## 1. The underlying problem

For an agent using a local/small LLM, the context window is a scarce resource.

A session may contain a large amount of useful history:

* previous reasoning
* tool calls/results
* intermediate debugging
* documents
* plans
* experiments
* failed attempts
* previous conclusions

But the LLM does **not** need all of that at every inference.

Therefore, two concepts are separated:

```text
Persistent history
       │
       │ projection
       ▼
Working context
       │
       ▼
      LLM
```

The history remains intact, while the model receives a **managed projection** of that history.

The guiding analogy is:

> **History is persistent storage; working context is RAM.**

This is particularly useful for affordable local LLMs, where context capacity is much more constrained than in large hosted models.

---

# 2. Immutable history vs mutable working context

The central architectural principle is:

```text
History = immutable
Context = mutable projection
```

If the agent removes something from its context, it does **not delete it from the session history**.

For example:

```text
History:
A B C D E F G H

Working context:
A B F G H
```

C, D and E still exist in history. They simply are not currently resident in the working context.

This means context manipulation is reversible in principle and does not destroy the actual trajectory.

---

# 3. Context GC / deletion

The first primitive is context garbage collection:

```text
context.gc(...)
```

Its semantics are:

> Remove things from the active working context while retaining them in persistent history.

A more ergonomic programming model uses marks:

```text
context.markForDeletion()

... do work ...

context.deleteMarked()
```

This effectively creates a temporary region.

For example:

```text
task
↓
mark
↓
debugging
↓
tool calls
↓
experiments
↓
deleteMarked
```

The debugging information remains in history, but no longer consumes the active context.

This is more ergonomic than manually tracking every item that should eventually be removed.

---

# 4. Markers are a fundamental primitive

The marker is more fundamental than `gc`.

Conceptually:

```text
mark()
   │
   ├── temporary work
   ├── temporary work
   └── temporary work
deleteMarked()
```

A marker defines a **context scope**.

Explicit marker handles are also possible in a later stage:

```text
m = context.mark()

...

context.delete(m)
```

But the simple stack-like form:

```text
mark()
...
deleteMarked()
```

is sufficient for an initial implementation.

---

# 5. `context.push`

The second primitive is `push`.

Its purpose is to deliberately add information to the working context:

```text
context.push(data)
```

For example:

```text
context.push(requirements)
context.push(constraints)
context.push(previous_analysis)
```

This is important because the context does not have to be a contiguous slice of the conversation.

The agent can **construct a context from selected pieces of information**.

This is a major part of the abstraction.

---

# 6. `context.call`

The key primitive is:

```text
context.call(prompt)
```

The idea is:

> Construct a context, execute an LLM computation over that context, and return the result.

The simplest semantic model is:

```text
Context + Prompt → Result
```

So:

```text
context.push(requirements)
context.push(constraints)

design = context.call("Design the solution")

context.push(design)

implementation = context.call("Implement it")
```

This makes `call()` analogous to a normal programming-language function call.

The important difference is that the **function argument is context**.

Instead of:

```text
function f(x)
```

the model is conceptually:

```text
Context + Prompt → Result
```

---

# 7. Why this is not fork/merge

The model is not necessarily:

```text
parent
 ├── fork
 │    └── child
 └── merge
```

because the child does not need to inherit the whole parent state.

Instead:

```text
parent context
      │
      │ explicitly construct
      ▼
child context
      │
      │ call
      ▼
result
      │
      ▼
parent pushes result
```

It is better described as:

> **A function call over an explicitly constructed context.**

The child receives only what the caller chooses to provide.

That makes the model more general and cleaner.

---

# 8. A debugging example

Suppose the agent does:

```text
task
↓
implementation
↓
verification
↓
failure
```

At that point it can mark the current context:

```text
context.mark()
```

Then perform a debugging computation:

```text
debugging
tool calls
experiments
analysis
fix
verification
```

Once the problem is solved:

```text
context.deleteMarked()
```

and retain only the useful result:

```text
context.push(fix_summary)
```

The main context is therefore kept small, while the complete debugging trajectory remains available in history.

---

# 9. `call()` is also a context transformation primitive

`call()` is not limited to "ask another agent something".

It can perform arbitrary context transformations:

```text
context.call("Summarize this")
context.call("Extract the requirements")
context.call("Find contradictions")
context.call("Critique this design")
context.call("Produce a structured representation")
context.call("Identify the remaining problems")
```

So operations like `summarize()` do not need to be separate primitives.

They are just particular calls:

```text
context.call("Summarize...")
```

This keeps the primitive set very small.

---

# 10. Sequential composition is sufficient initially

This is an important simplification.

Although `call()` can eventually be generalized to multiple contexts/prompts, for a normal affordable local LLM the **single sequential call is sufficient as the initial primitive**.

So the initial model can simply be:

```text
push
→ call
→ push
→ call
→ push
→ call
```

There is no initial need for:

* parallel workers
* explicit map/reduce
* complex agent orchestration
* distributed execution

The abstraction is already useful with one sequential LLM call at a time.

---

# 11. The natural generalization of `call`

The primitive naturally extends to multiple cases.

### Same context, different prompts

```text
C
├── call(P1)
├── call(P2)
└── call(P3)
```

### Different contexts, same prompt

```text
C1 ── call(P)
C2 ── call(P)
C3 ── call(P)
```

### Different contexts and different prompts

```text
(C1,P1)
(C2,P2)
(C3,P3)
```

This resembles `map`.

Then results can be combined:

```text
C
├── call(P1) → R1
├── call(P2) → R2
└── call(P3) → R3
             │
             ▼
       call([R1,R2,R3], "synthesize")
```

This gives a natural map/reduce-style reasoning model.

However, **this does not need to be part of version one**.

`call()` is the fundamental operation; batching and parallelism can be added later.

---

# 12. Hierarchical planning fits naturally

Hierarchical planning proceeds as follows:

1. Make a coarse master plan.
2. Take each task.
3. Analyze/decompose it further.
4. Continue until the task is small enough.
5. Execute the resulting tasks.
6. Verify.
7. If necessary, create another focused reasoning/debugging context.

This is not specifically a coding technique.

It is a general **hierarchical reasoning/information-flow architecture**.

The important observation is that planning does not only determine **what actions happen**.

It also determines:

> **what information is allowed to enter each computation.**

For example:

```text
Goal
├── analyze requirements → requirements
├── design              → design
├── implement           → implementation
├── verify              → report
└── fix                 → final result
```

Each computation can receive only the context it needs.

This is one of the strongest reasons for making context a first-class object.

---

# 13. Bottom-up reasoning also fits

The same model supports broader information-gathering patterns, such as looking across an entire codebase and collecting information bottom-up before doing a cross-cutting refactor.

```text
module A → analysis A ┐
module B → analysis B ├─→ synthesis → global understanding
module C → analysis C ┘
```

The abstraction is not tied to a particular planning strategy.

It supports:

* top-down decomposition
* bottom-up synthesis
* iterative hypothesis/experiment
* focused debugging
* cross-cutting analysis
* context summarization
* independent perspectives

---

# 14. Recursion is the main safety/runtime constraint

Once `context.call()` exists, a call can potentially make another call:

```text
call
 └── call
      └── call
           └── ...
```

Recursion therefore needs to be constrained.

The obvious initial solution is a depth limit:

```text
MAX_DEPTH = 2 or 3
```

Conceptually:

```text
context.call(prompt, depth)

if depth >= MAX_DEPTH:
    do not expose/allow another call
```

A call budget can eventually supplement that:

```text
max_depth
max_calls
max_tokens
```

But depth alone is enough for a first implementation.

This reinforces the fact that the initial system does not need complicated orchestration.

---

# 15. The difficult part: the projection

`context.call()` itself is relatively simple.

The difficult engineering problem is the **projection of the persistent session into the actual LLM request**.

The session should not be physically rewritten every time the context changes.

Instead, conceptually:

```text
Immutable session/history
          │
          │ projection
          ▼
   Working context
          │
          ▼
     provider API
```

The history remains the source of truth.

The working context is a materialized view/projection.

---

# 16. Event log + materialized view

The implementation follows the pattern:

> **append-only event log + materialized projection**

For example:

```text
History:
e1 e2 e3 e4 e5 e6 e7 e8
```

Then context operations:

```text
keep all
gc e3,e4,e5
push X
push Y
```

produce:

```text
Working context:
e1 e2 e6 e7 e8 X Y
```

The context operations themselves can be persisted.

The original history does not have to be modified.

The materialized projection can also be checkpointed periodically:

```text
checkpoint:
active = [e1,e2,e6,e7,e8,X,Y]
```

so that every operation does not need to be replayed from the beginning.

---

# 17. The resulting session architecture

The architecture becomes roughly:

```text
                 ┌─────────────────────┐
                 │ Immutable History   │
                 │                     │
                 │ e1 e2 e3 ... eN     │
                 └──────────┬──────────┘
                            │
                            │ projection
                            ▼
                 ┌─────────────────────┐
                 │ Working Context     │
                 │                     │
                 │ selected entries    │
                 │ + pushed data       │
                 └──────────┬──────────┘
                            │
                            ▼
                           LLM
```

The history is durable.

The working context is disposable and reconstructible.

---

# 18. The context operations themselves can be events

Operations such as:

```text
mark()
push(...)
gc(...)
call(...)
```

can themselves become part of the session/event stream.

The session can therefore conceptually contain:

```text
message
tool_call
tool_result
context.mark
context.push
context.call
context.result
context.deleteMarked
...
```

This provides persistence without making the context itself the authoritative storage.

---

# 19. How to expose the primitives to the LLM

A remaining design question is:

> How does the LLM actually get the ability to manipulate its context?

The simplest answer is: **make them tools**.

For example:

```text
context.mark()
context.deleteMarked()
context.push(data)
context.call(prompt)
```

From the LLM's perspective these are ordinary tool calls.

For example:

```text
assistant → context.mark()
tool      → success

assistant → context.push("...")
tool      → success

assistant → context.call("Analyze the problem")
tool      → result

assistant → context.deleteMarked()
tool      → success
```

The runtime handles the actual projection mechanics.

---

# 20. `context.call()` can look like an ordinary tool

This is a particularly useful property.

From the main model's perspective:

```text
assistant
    │
    │ context.call(...)
    ▼
  tool result
    │
    ▼
assistant continues
```

Internally, however, the runtime does:

```text
construct child context
        ↓
invoke LLM
        ↓
obtain result
        ↓
return result as tool result
```

So the model does not need to understand the internal implementation.

It simply has a tool whose implementation happens to invoke another LLM inference.

---

# 21. Tool visibility can be scoped

`context.call()` does not necessarily need to be exposed recursively without limit.

A child call can receive a restricted tool set.

For example:

```text
Main LLM
  └── context.call
       └── child LLM
            ├── context.push
            ├── context.mark
            └── context.deleteMarked
```

and, once the maximum depth is reached, no `context.call` at all.

This gives a clean mechanism for controlling recursion.

---

# 22. `push` should remain simple

Low-level session manipulation should not be exposed.

The model does not need to know internal history IDs for ordinary operations.

A simple interface is enough:

```text
context.push(data)
```

The runtime can assign internal IDs:

```text
push("API requires authentication")
       ↓
internal context item c17
```

Those IDs can remain an implementation detail unless precise manipulation is required.

---

# 23. Markers reduce the need for IDs

The marker abstraction is particularly useful because of this.

Instead of the model saying:

```text
delete e37
delete e38
delete e39
delete e40
...
```

it can simply do:

```text
context.mark()

... temporary work ...

context.deleteMarked()
```

The runtime knows which context items belong to that temporary scope.

This makes the API more natural for an LLM.

---

# 24. UI implications

The same abstraction can work interactively in a UI.

A user can see history/context entries and choose:

```text
Keep
Summarize
Remove from context
```

while the actual history remains intact.

The UI can also expose:

* current context
* context token usage
* selected/removed entries
* markers/scopes
* context produced by calls
* the savings obtained by removing material

The same underlying context projection mechanism can therefore serve both:

```text
LLM-controlled context management
```

and

```text
human-controlled context management
```

---

# 25. Relationship to existing systems

Existing work, particularly **Letta/MemGPT**, is the closest conceptual relative.

Letta separates active context from longer-term/archival information and supports editable memory blocks and retrieval.

This demonstrates that the general problem of separating persistent information from active model context is well established.

This formulation is more specifically centered around:

```text
immutable trajectory
        +
mutable context projection
        +
context.push()
context.call()
context.mark()
context.deleteMarked()
```

The idea is therefore better described as a **specific context-computation/programming abstraction**, rather than as a new general idea of managed context.

---

# 26. The minimal primitive set

The whole system can be expressed compactly.

### Context lifecycle

```text
mark()
deleteMarked()
```

### Context construction

```text
push(data)
```

### Context computation

```text
call(prompt)
```

The core composition is:

```text
push
  ↓
call
  ↓
push(result)
  ↓
call
  ↓
...
```

and temporary work can be wrapped with:

```text
mark
  ↓
work
  ↓
deleteMarked
```

---

# 27. The deeper abstraction

The strongest formulation is:

> **Agents have persistent history and a managed working context. Context functions perform computation over explicitly constructed context.**

More formally:

```text
History = immutable event log

Context = projection(History, operations)

call(Context, Prompt) → Result
```

The important point is that **context itself becomes a first-class computational resource**.

Instead of treating the prompt/context as an incidental blob that happens to be sent to an LLM, the runtime provides operations for constructing and transforming it.

---

# 28. Why this is particularly interesting for local LLMs

For a large hosted model with a huge context window, brute-force inclusion of history can sometimes work reasonably well.

For a small local model, every unnecessary token matters.

So instead of asking:

> "How can more history be fit into the context?"

the system asks:

> "What is the smallest useful context for this particular computation?"

That makes the architecture much more compatible with affordable local inference.

The LLM can effectively manage its own limited RAM.

---

# 29. The likely first implementation in Pi

The Pi integration can therefore remain relatively small.

A **ContextManager/projection layer** is introduced between Pi's persistent session and the provider request.

Something approximately like:

```text
Pi session
   │
   ▼
ContextManager
   ├── active entries
   ├── markers
   ├── pushed entries
   ├── call depth
   └── projection
   │
   ▼
provider messages
```

The underlying Pi history is not destructively modified.

Before each actual LLM request:

```text
session history
      ↓
context projection
      ↓
provider-compatible messages
      ↓
LLM
```

The biggest implementation subtlety is that the projection cannot blindly manipulate arbitrary raw messages. It needs to understand logical message/tool-call/tool-result structures so that it does not produce an invalid provider conversation.

---

# 30. What the first version deliberately does not need

The first version does **not** need:

* parallel calls
* map/reduce primitives
* complex agent graphs
* automatic planning
* sophisticated memory systems
* arbitrary context IDs exposed to the model
* distributed workers
* elaborate fork/merge semantics
* automatic summarization as a primitive
* complicated scheduling

These can all be built later on top of the basic abstraction.

The initial system can essentially be:

```text
persistent history
+
working context projection
+
push
+
mark/delete
+
call
+
recursion limit
```

That is enough to demonstrate the idea.

---

# 31. The conceptual progression

The design can be condensed into this progression:

```text
1. Context is scarce for local LLMs
             ↓
2. Keep history immutable
             ↓
3. Give the agent a mutable working projection
             ↓
4. Add context.push()
             ↓
5. Add context.mark()/deleteMarked()
             ↓
6. Let the agent compute over constructed context
             ↓
7. Add context.call(prompt)
             ↓
8. Treat the call as a function whose argument is context
             ↓
9. Nest calls with a depth limit
             ↓
10. Generalize later to multiple contexts/prompts
```

The central conceptual shift is:

> **Don't make the conversation itself the context. Make context a programmable projection of the conversation.**
