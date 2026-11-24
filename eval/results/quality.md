# Quality eval

Model `nvidia/nemotron-3-ultra-550b-a55b:free` via OpenRouter, Python 3.12.10; context tokens are o200k_base.
Accuracy = share of questions answered correctly (keyword check), mean over runs.

| Context | Tokens | textproc | json | shrink | All | Structure questions | Body questions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 26019 | 100% | 100% | 100% | 100% | 100% | 100% |
| L1 | 23596 | 100% | 100% | 100% | 100% | 100% | 100% |
| L2 | 17304 | 100% | 100% | 100% | 100% | 100% | 100% |
| L3 | 5733 | 62% | 92% | 100% | 85% | 100% | 39% |
| L3+query | 10437 | 75% | 100% | 100% | 92% | 100% | 67% |

## Wrong answers

| Context | Question | Needs | Answers |
| --- | --- | --- | --- |
| L3 | a4: What exactly does shlex.quote return when it is given an empty string? | body | 3x: unknown |
| L3 | a7: Which function does shlex.join apply to every argument before joining them? | body | 3x: unknown |
| L3 | a8: Which function does fnmatch.fnmatch apply to both the name and the pattern before matching? | body | 3x: unknown |
| L3 | b5: Complete the error message raised for an object that cannot be serialized: 'Object of type X ...' | body | 2x: unknown |
| L3+query | a4: What exactly does shlex.quote return when it is given an empty string? | body | 3x: unknown |
| L3+query | a7: Which function does shlex.join apply to every argument before joining them? | body | 3x: unknown |
