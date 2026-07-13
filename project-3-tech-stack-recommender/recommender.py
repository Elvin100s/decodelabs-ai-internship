"""
Project 3 — AI Recommendation Logic: Tech Stack Recommender (DecodeLabs)

Content-based filtering engine following the 4-step ranking pipeline:
  1. INGESTION  -> capture the user state (minimum 3 skills)
  2. SCORING    -> TF-IDF vectors + cosine similarity against every job role
  3. SORTING    -> descending order by similarity score
  4. FILTERING  -> truncate to the Top-N list (Top 3) to prevent choice overload
"""

import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET = os.path.join(os.path.dirname(__file__), "raw_skills.csv")
TOP_N = 3
MIN_SKILLS = 3


def ingest_user_skills() -> list[str]:
    """Step 1: Ingestion — capture at least MIN_SKILLS user inputs."""
    print("Enter your skills one per line (at least 3). Press Enter on an empty line to finish.")
    skills: list[str] = []
    while True:
        raw = input(f"Skill {len(skills) + 1}: ")
        clean = raw.lower().strip()  # sanitization, same vocabulary space as the dataset
        if clean:
            skills.append(clean)
        elif len(skills) >= MIN_SKILLS:
            return skills
        else:
            print(f"Please enter at least {MIN_SKILLS} skills for accurate matching.")


def recommend(user_skills: list[str], items: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """Steps 2–4: Scoring, Sorting, Filtering."""
    # Vector mapping: items and the user profile must share the exact same
    # vocabulary space, so we vectorize them together.
    documents = items["skills"].tolist() + [" ".join(user_skills)]

    # TF-IDF: rewards specific, descriptive tags; penalizes generic high-frequency terms.
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(documents)
    item_vectors, user_vector = matrix[:-1], matrix[-1]

    # Cosine similarity: measures the angle between vectors (orientation of
    # preferences), invariant to magnitude. With non-negative TF-IDF weights the
    # score sits between 0 and 1 — an intuitive percentage match.
    scores = cosine_similarity(user_vector, item_vectors).flatten()

    ranked = items.assign(match=scores).sort_values("match", ascending=False)
    return ranked.head(top_n)


def main() -> None:
    items = pd.read_csv(DATASET)
    print(f"=== DecodeLabs Tech Stack Recommender ===")
    print(f"Loaded {len(items)} job roles from raw_skills.csv\n")

    user_skills = ingest_user_skills()
    top = recommend(user_skills, items)

    print(f"\nYour profile: {user_skills}")

    if top["match"].max() == 0:
        # Cold start: no overlap between the profile and any item vector.
        print("\nNo matches found (user cold start). Popular fallback roles:")
        for role in items["role"].head(TOP_N):
            print(f"  - {role}")
        return

    print(f"\nTop {TOP_N} recommended career paths:")
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        print(f"  {rank}. {row['role']:<28} match: {row['match'] * 100:5.1f}%")
        print(f"     stack: {row['skills']}")


if __name__ == "__main__":
    main()
