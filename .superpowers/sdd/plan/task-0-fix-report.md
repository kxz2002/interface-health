# T0 guard repairs — 2026-10-14

## Scope and status

Completed narrow protocol/config/test/documentation repairs after T1 at a44e71d. No metrics implementation change, prepare, scoring, training, old pipeline or raw writes. Untracked AGENTS.md preserved. Current frozen configuration was valid: this repair does not claim actual training contamination.

## Changes

- Frozen exact case → role/anomaly/target identities; Normal injection/target null; ordered capture/injection times. KILL injection may end after capture ends.
- All feature statuses nonnull and allowed; OBSERVED iff finite value, other statuses NaN; nonnegative resources, integral process count; nonnull booleans/counts; resource_valid exactly matches R3 plus one unambiguous observed identity. Extra diagnostic ambiguity can invalidate otherwise observed R3.
- Nonnull independent_label requires INDEPENDENT_STATE, heldout, empty actual input dependencies and confirmed review. Instance termination remains distinct from Service unavailable.
- Corrected registry actual typo 64996ff28d8-gwdfs to 64996ff68-gwdfs; strict parser accepts 9/10 character ReplicaSet hashes, full candidate service names, five character Pod suffixes.
- Documentation now states nine confirmed evidence records, actual configured input independence, state semantic matching and S2 injection target retrieval.

## Red/green evidence

Added behavioral tests first. Command:

```bash
conda run -n interface python -m pytest tests/test_service_protocol.py -q
```

Red: 15 failed / 21 passed. Failures reproduced CPU→Normal mapping, target mutation, null status, OBSERVED NaN, negative resource/count, inconsistent valid flag, null bool, all three weak grades with independent label, changed 9-char hashes for security/execute/travel/user. Original tests passed.

After repair:

```bash
conda run -n interface python -m pytest tests/test_service_protocol.py tests/test_service_metrics.py -q
```

Green: 52 passed, then 55 passed in 2.29s after extra confirmed-review and legitimate Unknown/zero-limit/process-zero checks. Final three tests are additional boundary checks, not claimed red-first reproductions. Formatting: black on the two Python files. No full suite per explicit bounded task requirement.

## Bounded original identity and coverage check

Only two supplied Metric CSV files were streamed; direct identity selection uses pod/container/id/job/namespace. Both raw files confirm `ts-security-service-64996ff68-gwdfs`. No changed raw content or prepare outputs. Script below performs original identity read then direct in-memory T1 build, no model fitting.

|case|grid rows|valid R3 rows|source services|valid per service|
|---|---:|---:|---:|---:|
|Normal_20260728T161044Z|3000|3000|25|120|
|Lv_P_CPU_preserve_20260728T164711Z_em|3175|3175|25|127|

Historical T1 report counts were 2880/3048 valid rows before identity repair; corrected counts are 3000/3175, recovering security's 120/127 rows. Both retain all 25 candidate services. Coverage describes resource observations, not healthy state or model scores. Pandas identity read emitted DtypeWarning for mixed raw label columns; build completed using T1's explicitly typed Arrow reader. For reruns use dtype=str in the inspection read to avoid the warning.

Reproduction (repository root; original executed command `conda run -n interface python /tmp/t0_bounded.py`):

```python
from pathlib import Path
import json
import pandas as pd
from src.service.protocol import load_protocol, load_case_meta
from src.service.metrics import build_metric_windows
p=load_protocol(Path('configs/service/minimal_v1.yaml'),Path('configs/data/new_merge.yaml'))
results=[]
for c in p.config['cases'][:2]:
    d=Path('data/new_merge')/c['case_id']
    path=next((d/'metric_data').glob('*.csv'))
    pods=set()
    for chunk in pd.read_csv(path,usecols=['pod','container','id','job','namespace'],chunksize=100000):
        pods.update(chunk.loc[chunk.container.eq('ts-security-service'),'pod'].dropna())
    f,q=build_metric_windows([path],load_case_meta(d),p)
    result={'case_id':c['case_id'],'path':str(path),'security_pods':sorted(pods),'grid_rows':len(f),'r3_valid':int(f.resource_valid.sum()),'source_service_count':int(q.canonical_service_id.nunique()),'valid_by_service':f.groupby('canonical_service_id').resource_valid.sum().astype(int).to_dict()}
    results.append(result)
    print(json.dumps(result),flush=True)
Path('/tmp/t0-bounded-result.json').write_text(json.dumps(results,indent=2))
```

## Constraints/concerns

No KILL build or end-to-end prepare evaluation performed. Real metadata/config comparison remains T2. Missing `.claude/rules/preprocessors-and-raw-data.md` could not be read; followed supplied AGENTS.md and accessible Spec. Sandbox command startup failed (bwrap loopback Operation not permitted); authorized read/edit/test commands ran via escalation without rejected approval. Files outside scoped Service ownership untouched.
