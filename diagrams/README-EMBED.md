# Embedding these in the README

Lead with the money diagram — drop `![Why it cannot put words in your mouth](diagrams/04-why-it-cannot-put-words-in-your-mouth.svg)` directly under the intro, above "The core invariant"; it makes the guarantee visible before anyone reads a line of prose.
Put `01-context-c4-l1.svg` and `03-gate-components-c4-l3.svg` in "Architecture sketch" (the L3 one replaces the ASCII flow), and `02-containers-c4-l2.svg` under "How to run", since it is the thing that explains the two deployments.
Each `.svg` has a matching `.mmd` for anywhere an image won't do — paste its contents into a ```mermaid fence and GitHub renders it natively; the SVGs are self-contained (no external fonts, scripts or assets) and are the higher-fidelity version.
