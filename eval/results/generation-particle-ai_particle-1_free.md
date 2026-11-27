# Generation eval

Model `particle-ai/particle-1:free` via OpenRouter, Python 3.12.10. Task: accounts and todo lists, backend to frontend, in a FastAPI + JS starter project.
Reasoning effort: none. Input tokens: whole prompt, o200k_base. Output tokens: as counted by the provider (mean over runs).

| Context | Input | Input saved | Output | Total | Total saved | Cut off | Complete | Works | Python parses | Edits applied | Restored | Time, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 11556 | +0.0% | 8784 | 20340 | +0.0% | 0/3 | 100% | 84% | 12/12 | — | — | 40 |
| L3+keep+dedent+brief+patch | 5416 | -53.1% | 4526 | 9942 | -51.1% | 0/3 | 100% | 95% | 12/12 | — | 11 files | 44 |

## Failed behaviour checks (one line per answer)

- original: audit log records login and login_failed; frontend: sign-in page with its script; frontend: todo page with its script
- original: audit log records login and login_failed; frontend: sign-in page with its script; frontend: todo page with its script
- original: audit log records login and login_failed; frontend: sign-in page with its script; frontend: todo page with its script; todos: title must be 1-200 characters after trimming
- L3+keep+dedent+brief+patch: audit log records login and login_failed
- L3+keep+dedent+brief+patch: audit log records login and login_failed
- L3+keep+dedent+brief+patch: audit log records login and login_failed
