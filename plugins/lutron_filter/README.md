# Lutron Filter Plugin

Automatically create precomputed filters from 1d convolutions.
Workflow:

1. Convert your model into a precomputable one
2. Train and evaluate as usual
3. Generate vhdl sources
4. Use the [experiment-framework](https://github.com/es-ude/elastic-ai.experiment-framework) for synthesis
5. Use the experiment-framework to upload the bitstream and run the inference

## Installation

In your uv project run

```
  uv add "git+https://github.com/es-ude/elastic-ai.creator/plugins/lutron_filter"
```
