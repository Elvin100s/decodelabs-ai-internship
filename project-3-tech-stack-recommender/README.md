# Project 3 — AI Recommendation Logic: Tech Stack Recommender 🎯

**Goal:** Create a simple recommendation system based on user preferences — the
"Digital Matchmaker" that cures choice overload.

The user enters their skills; the engine maps them against a dataset of job roles
(`raw_skills.csv`) and returns the Top-3 most relevant career paths, ranked by
angular alignment.

## Approach: content-based filtering

Chosen over collaborative filtering because it's driven by **item attributes**
(role → skills), needs no historical user data, and is inherently robust against
the *item cold start* problem.

## The 4-step ranking pipeline

| Step | What happens |
|---|---|
| 1. **Ingestion** | Accepts a minimum of **3 user skills** (sanitized to the shared vocabulary space). |
| 2. **Scoring** | Roles and the user profile become **TF-IDF vectors** — term frequency rewards representative tags, inverse document frequency (log-dampened) penalizes generic ones. Each role is scored with **cosine similarity**: `cos(θ) = A·B / (‖A‖‖B‖)`, invariant to vector magnitude (unlike Euclidean distance, which fails at scale). |
| 3. **Sorting** | Descending by similarity score. |
| 4. **Filtering** | Truncates to the **Top-3** list to prevent information overload. |

**Cold start handling:** if the user's skills share nothing with the vocabulary
(all scores 0), the engine falls back to popular roles instead of recommending noise.

## Run it

```bash
python3 recommender.py
```

Example session:

```
Skill 1: python
Skill 2: cloud computing
Skill 3: automation

Top 3 recommended career paths:
  1. DevOps Engineer               match:  34.4%
  2. Cloud Architect               match:  32.1%
  3. Site Reliability Engineer     match:  28.9%
```

## Files

- `recommender.py` — the engine
- `raw_skills.csv` — 20 job roles ("items") with their skill stacks

## Key skills demonstrated

Logic building, pattern matching, vector mapping, TF-IDF feature extraction,
cosine similarity, ranking pipelines, and cold-start strategies.
