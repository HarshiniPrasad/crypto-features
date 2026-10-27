# crypto-features

Shared feature and target engineering for UTS 36120 AMLA AT3, Group 26. One file per
student, so each of us only commits our own feature logic, but everyone computes
identical values by importing the same package in the experiments repo, the 4 FastAPI
repos, and the Streamlit repo.

## Install

```
pip install git+https://github.com/<OWNER>/crypto-features.git
```

Replace `<OWNER>` with wherever this repo ends up living (a group member's account, or
a shared org). Pin to a commit or tag once the feature set stabilises, so a later push
doesn't silently change everyone's features mid-assignment:

```
pip install git+https://github.com/<OWNER>/crypto-features.git@<commit-or-tag>
```

## Usage

```python
import pandas as pd
from crypto_features import build_all_features
from crypto_features.targets import add_student_c_target, drop_unlabeled_tail

raw = pd.read_csv("crypto_daily_ohlcv.csv", parse_dates=["date"])
# raw must have columns: token, date, open, high, low, close, volume

features = build_all_features(raw)          # adds every student's features
features = add_student_c_target(features)   # adds TrendScore, TrendCategory, TrendCategory_t+7
data = drop_unlabeled_tail(features, "TrendCategory_t+7")
```

Each student can also call just their own target function
(`add_student_a_target`, `add_student_b_target`, `add_student_c_target`,
`add_student_d_target` in `crypto_features.targets`) on the output of
`build_all_features`.

## Structure

| File | Owner | Features |
|---|---|---|
| `student_a.py` | A | DailyReturn, Return5, Return20, Momentum |
| `student_b.py` | B | MA5, MA20, TrendStrength |
| `student_c.py` | C | Volatility20, MaxPrice20, MaxDrawdown, GapRisk |
| `student_d.py` | D | SMA20(Volume), RelativeVolume |
| `pipeline.py` | — | `build_all_features`, runs every student's `add_features` in dependency order (A, B before C; D independent) |
| `targets.py` | — | Each student's target formula and the t+7 shift, applied per coin |

**Rule:** only edit your own `student_x.py` file. If your target or feature needs a
column another student's file doesn't produce yet, say so instead of adding it
yourself — `pipeline.py`'s ordering is the single place dependencies are resolved.

## Why a separate package

Confirmed with Anthony (5 Oct): an installable shared package means every student's
model is trained on identical feature values, without copy-pasting feature code
between 6 repos (experiments, Streamlit, 4 FastAPI services) and without one student's
local edits silently diverging from another's.

## Status

Student A, B and D's files are reference implementations from the group's agreed
formulas (feature-ownership table), written so Student C's pipeline isn't blocked
while waiting on the owning students. **Each owner should replace their file with
their own implementation** — the function signature (`add_features(df) -> df`) and
output column names must stay the same so nothing downstream breaks. Diff your
version's output against this one on a few rows to confirm you get the same values
before replacing it.
