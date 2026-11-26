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
| L3+brief | 5200 | -55.0% | 7703 | 12903 | -36.8% | 0/3 | 97% | 12/12 | — | 212 |
| L3+no-docs+brief | 4270 | -63.0% | 9659 | 13929 | -31.7% | 0/3 | 67% | 10/10 | — | 165 |
| L3+brief+edits | 5288 | -54.2% | 10819 | 16107 | -21.1% | 1/3 | 86% | 9/9 | 10/21 | 293 |

## Missing parts

- L3: frontend uses apiFetch, login page, todos page
- L3+brief: frontend uses apiFetch
- L3+no-docs+brief: frontend uses apiFetch, login page, todos page
- L3+no-docs+brief: login page, paginated list, routers registered, todo PATCH and DELETE, todos are per user, todos router file
- L3+no-docs+brief: frontend uses apiFetch, login page, todos page
- L3+brief+edits: login page, routers registered
- L3+brief+edits: frontend uses apiFetch, routers registered
- L3+brief+edits: routers registered
