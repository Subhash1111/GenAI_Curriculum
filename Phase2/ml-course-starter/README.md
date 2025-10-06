# ML Course Starter (4 Weeks)

A GitHub-ready repository to **teach Machine Learning in a few weeks**. It includes a clear curriculum, hands-on notebooks, reusable scripts, tests, and CI. Use it for cohorts, bootcamps, or self-paced learning.

## What’s Inside
- **Week-by-week curriculum** with learning goals & checklists
- **Notebooks** for lessons + labs
- **Reusable training scripts** for regression/classification/clustering
- **Assignments & projects** with rubrics
- **Tests & CI** to keep solutions working
- **Environment files** for easy setup (pip or conda)

## Quick Start
```bash
# 1) Create environment
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r environment/requirements.txt

# 2) Launch Jupyter
jupyter lab

# 3) Run a demo training script
python scripts/train_sklearn.py --task classification --dataset iris --model random_forest
```

> Datasets are fetched automatically when possible; otherwise see `data/README.md`.

## Weekly Flow
- **Week 1 — Data Science Foundations:** pandas, numpy, EDA, viz
- **Week 2 — Statistics & SQL:** hypothesis testing, AB tests
- **Week 3 — Supervised Learning:** regression, classification, model metrics
- **Week 4 — Unsupervised + Capstone:** clustering, dimensionality reduction, end‑to‑end project

See `curriculum.md` for detailed outcomes, lesson plans, and assignments.

## Repo Structure
```
ml-course-starter/
├─ environment/          # pip/conda files
├─ notebooks/            # lessons & labs (Week1–Week4)
├─ scripts/              # reusable python scripts (train/evaluate)
├─ data/                 # local data folder (gitignored), fetch helpers
├─ tests/                # unit tests for exercises
├─ projects/             # capstone template & rubric
├─ .github/workflows/    # CI for tests & lint
└─ curriculum.md
```

## Teaching Tips
- Start each week with a **mini-lecture notebook**, then a **guided lab**, then a **small assignment**.
- Use the **scripts** to standardize model training across cohorts.
- Encourage **PR-based submissions**; use GitHub Issues for Q&A.
