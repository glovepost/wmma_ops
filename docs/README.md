# Documentation map

The repository contains current operating guidance, architecture references,
and a long historical notebook. They do not carry equal authority.

## Start here

1. [Project README](../README.md) — build, API, testing, and benchmark entry
   points.
2. [Performance status](PERFORMANCE_STATUS.md) — current result ledger,
   numerical contracts, source audit, rejected paths, and promotion gate.
3. [Profiling guide](PROFILING.md) — how to separate timing, counters, and ISA
   analysis on gfx1151.

## Architecture references

- [WMMA fragment layout for gfx1151](wmma_fragment_layout_rdna3.md) derives
  the A/B/C/D lane and register mappings used by the helper code.
- [Annotated RDNA3.5/WMMA references](wmma_references.md) ranks primary AMD
  material and calls out misleading or architecture-specific guidance.
- [RDNA3.5 ISA reference conversion](rdna35_instruction_set_architecture.md)
  is a searchable conversion of AMD document 70649 with section numbering
  preserved.
- `rdna35_figures/` contains figure pages whose vector content was not present
  in the source PDF's text layer.

## Related investigation

[Decode-attention findings](decode_attention_gfx1151.md) documents a separate
bandwidth-bound attention experiment used by Ember. Its shapes, bottlenecks,
and measurements must not be presented as GEMM results from `wmma-ops`.

## Historical material

[WMMA development notes](WMMA_DEVELOPMENT_NOTES.md) is an append-only lab
notebook assembled across several optimization sessions. It intentionally
retains failed hypotheses, old filenames, superseded toolchains, and results
from different shapes. Use it to recover experiment history, not to answer
“what is current?”

When documents disagree, use this order:

1. generated code, full-output validation, and retained raw measurements;
2. `PERFORMANCE_STATUS.md`;
3. the project README and profiling guide;
4. the historical development notebook.

## Updating the documentation

- Date performance-ledger refreshes and identify the source commit.
- State matrix shape plus input, accumulation, and output types beside every
  throughput number.
- Label peak samples separately from fresh-process distributions.
- Keep profiler durations out of performance denominators.
- Link primary sources near architecture claims; record inferences as such.
- Move superseded investigations into the notebook instead of silently
  rewriting their historical result.
- Check relative links and command syntax before publishing.
