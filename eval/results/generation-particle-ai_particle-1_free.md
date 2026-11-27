# Generation eval

Model `particle-ai/particle-1:free` via OpenRouter, Python 3.12.10. Task: accounts and todo lists, backend to frontend, in a FastAPI + JS starter project.
Input tokens: whole prompt, o200k_base. Output tokens: as counted by the provider (mean over runs).

| Context | Input | Input saved | Output | Total | Total saved | Cut off | Complete | Python parses | Edits applied | Time, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 11556 | +0.0% | 16000 | 27556 | +0.0% | 1/1 | 83% | 4/4 | — | 56 |

## Missing parts

- original: frontend uses apiFetch, todos page
