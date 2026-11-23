# Synthetic RESULTS

## §10 Claim ledger

```bash
export BASE=/tmp
export WORK=$BASE/work
```

| # | claim | evidence | regeneration command | verifier verdict |
|---|---|---|---|---|
| 1 | abc hashes to sha256 `ba7816bf…` | none | `printf 'abc' \| sha256sum` | |
| 2 | wrong hash expected sha256 `deadbeef00` | none | `printf 'abc' \| sha256sum` | |
| 3 | gpu row | none | `CUDA_VISIBLE_DEVICES=0 uv run python measure_rho.py` | |
| 4 | writes row | none | `echo hi > $WORK/out.txt` | |
| 5 | no hash | none | `echo $WORK` | |

## §11 Next
