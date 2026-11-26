# Generation eval

Model `nvidia/nemotron-3-ultra-550b-a55b:free` via OpenRouter, Python 3.12.10. Task: accounts and todo lists, backend to frontend, in a FastAPI + JS starter project.
Input tokens: whole prompt, o200k_base. Output tokens: as counted by the provider (mean over runs).

| Context | Input | Input saved | Output | Total | Total saved | Cut off | Complete | Python parses | Edits applied | Time, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 11556 | +0.0% | 8846 | 20402 | +0.0% | 0/3 | 100% | 12/12 | — | 402 |
| L1 | 11086 | -4.1% | 8606 | 19692 | -3.5% | 0/1 | 100% | 4/4 | — | 335 |
| L2 | 8148 | -29.5% | 8111 | 16259 | -20.3% | 0/1 | 100% | 4/4 | — | 196 |
| L3 | 5136 | -55.6% | 10556 | 15692 | -23.1% | 1/3 | 92% | 12/12 | — | 1546 |
| L3+query | 8606 | -25.5% | 8916 | 17522 | -14.1% | 0/1 | 100% | 4/4 | — | 316 |
| L3+brief (stale) | 5273 | -54.4% | 7703 | 12976 | -36.4% | 0/3 | 97% | 12/12 | — | 212 |
| L3+no-docs+brief (stale) | 4343 | -62.4% | 9659 | 14002 | -31.4% | 0/3 | 67% | 10/10 | — | 165 |

(stale): the answer was given for an older prompt (shrink has changed since); run the eval again to replace it.

## Missing parts

- L3: frontend uses apiFetch, login page, todos page
- L3+brief: frontend uses apiFetch
- L3+no-docs+brief: frontend uses apiFetch, login page, todos page
- L3+no-docs+brief: login page, paginated list, routers registered, todo PATCH and DELETE, todos are per user, todos router file
- L3+no-docs+brief: frontend uses apiFetch, login page, todos page
