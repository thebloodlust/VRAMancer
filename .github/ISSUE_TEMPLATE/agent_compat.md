---
name: Agent client compatibility report
about: Tell us whether a coding agent (Cline, Continue, OpenWebUI, …) works against VRAMancer
title: "[compat] <agent name> — works / broken"
labels: compatibility
---

<!--
We only claim what we measured. Aider is the only client validated end-to-end here.
Your report is how the compatibility table in the README grows — a "it's broken"
report is as useful as a "it works".
-->

**Agent / client**: <!-- Cline, Continue, OpenWebUI, Zed, custom script… + version -->

**Result**: ☐ works  ☐ partially works  ☐ broken

**What you ran**
```bash
# the vramancer serve command, and how you pointed the client at it
```

**What happened**
<!-- For a failure: the client-side error AND the server log lines around it.
     For a partial: which feature broke (tool calls? streaming? system prompt?). -->

**Environment**
<!-- Paste the output of:  vramancer doctor --share
     It is anonymised: hardware and versions only, no hostname/user/paths. -->

```
<paste here>
```

**Streaming?**  ☐ streamed  ☐ `--no-stream`
<!-- Streaming of tool-calls is deliberately not implemented yet; a client that
     requires it is expected to fail, and that report is exactly what would
     unfreeze the feature. -->
