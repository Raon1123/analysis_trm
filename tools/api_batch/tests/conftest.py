import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

WITHHELD_FIXTURE = """version: 1
families:
  - id: exp013-fixture
    status: withheld
    exp_ids: [EXP-013]
    patterns:
      - '(?<![A-Za-z0-9])fig1_(?:mlp|tf)_(?:z|noz)_(?:iter|noiter)_k\\d+_s(?:[2-9]|\\d{2,})(?![0-9])'
  - id: exp012-fixture
    status: withheld
    exp_ids: [EXP-012]
    patterns:
      - '(?<![A-Za-z0-9])clean[-_ ]?r1(?![0-9])\\w*'
  - id: released-fixture
    status: released
    exp_ids: [EXP-015]
    patterns: []
  - id: verdict-prose
    status: withheld
    exp_ids: []
    patterns:
      - 'falsified|refuted|C-LTF|verdict|판독'
"""
API_FIXTURE = """version: 1
families:
  - id: locked-fixture
    status: withheld
    kind: locked
    exp_ids: [EXP-016]
    patterns: []
  - id: per-call-fixture
    status: withheld
    kind: sealed-artifact
    patterns: ['per_call\\.csv']
    paths: ['**/per_call.csv', '**/sealed/**']
"""


@pytest.fixture
def deny_files(tmp_path):
    a, b = tmp_path / "WITHHELD_RUNS.yaml", tmp_path / "API_BATCH_DENY.yaml"
    a.write_text(WITHHELD_FIXTURE, encoding="utf-8")
    b.write_text(API_FIXTURE, encoding="utf-8")
    return [a, b]


@pytest.fixture
def lab(tmp_path):
    d = tmp_path / "labroot"
    (d / "11_governance/11_lab-root").mkdir(parents=True)
    return d


@pytest.fixture
def builder(deny_files, lab):
    from tools.api_batch.pack import PackBuilder
    return PackBuilder.from_files(deny_files, lab_root=lab, cap_tokens=60000)
