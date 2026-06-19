# Theory & Implementation

This section explains **how ROSS is built on the inside** — pairing the underlying
rotordynamic *theory* with the actual *implementation* in the source code.

Where the [User Guide](../user_guide/user_guide) shows you *how to use* ROSS and the
[API Reference](../references/api) documents *what* each function does, the pages here
answer a different question: ***how does the code work, and why is it written that way?***

Each walkthrough takes one part of ROSS and reads it end to end — the governing equations,
the solution strategy, the class/method structure, and runnable examples that open up the
internal state at every stage. The goal is that, after reading a page, you can open the
corresponding source file and understand every line.

These pages are aimed at:

- **Users** who want to understand the model behind a result (assumptions, limitations, signs).
- **Potential developers** who want to extend, debug, or contribute to ROSS and need to see how
  theory maps onto the implementation.

```{toctree}
:maxdepth: 1
:caption: Theory & Implementation
labyrinth_seal
```
