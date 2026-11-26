# Quality eval

Model `nvidia/nemotron-3-ultra-550b-a55b:free` via OpenRouter, Python 3.12.10; prompt tokens counted with o200k_base.
Accuracy = share of questions answered correctly (keyword check), mean over runs.

| Context | Tokens | textproc | json | shrink | All | Structure questions | Body questions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 24784 | 100% | 100% | 100% | 100% | 100% | 100% |
| L1 | 22582 | 100% | 100% | 100% | 100% | 100% | 100% |
| L2 | 16680 | 100% | 100% | 100% | 100% | 100% | 100% |
| L3 | 6044 | 62% | 92% | 100% | 85% | 100% | 39% |
| L3+query | 9084 (partial) | 100% | 100% | — | 100% | 100% | 100% |

Not finished yet: L3+query (some corpora have no answers; run the same command again to continue).

## Wrong answers

| Context | Question | Needs | Answers |
| --- | --- | --- | --- |
| L3 | a4: What exactly does shlex.quote return when it is given an empty string? | body | 3x: unknown |
| L3 | a7: Which function does shlex.join apply to every argument before joining them? | body | 3x: unknown |
| L3 | a8: Which function does fnmatch.fnmatch apply to both the name and the pattern before matching? | body | 3x: unknown |
| L3 | b5: Complete the error message raised for an object that cannot be serialized: 'Object of type X ...' | body | 2x: unknown |
