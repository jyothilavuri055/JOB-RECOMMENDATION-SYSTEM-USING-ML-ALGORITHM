# ===============================
# RESUME BASED JOB RECOMMENDATION (MULTIPLE INPUT)
# ===============================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# JOB DATASET
# ===============================

jobs = {
    "job_title": [
        "Data Scientist", "Web Developer", "ML Engineer",
        "Data Analyst", "Software Engineer", "AI Engineer"
    ],
    "job_description": [
        "Python machine learning data analysis statistics",
        "HTML CSS JavaScript web development frontend backend",
        "Deep learning neural networks artificial intelligence python",
        "SQL Excel data visualization business analysis",
        "Java C++ software development coding programming",
        "Artificial intelligence machine learning deep learning python"
    ]
}

jobs_df = pd.DataFrame(jobs)

# ===============================
# LOOP FOR MULTIPLE INPUTS
# ===============================

while True:
    print("\nEnter your resume (or type 'exit' to stop):\n")
    resume_text = input()
 if resume_text.lower() == 'exit':
        print("Program Ended ✅")
        break

    # TF-IDF
    all_text = jobs_df['job_description'].tolist()
    all_text.append(resume_text)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(all_text)

    # Similarity
    resume_vector = tfidf_matrix[-1]
    job_vectors = tfidf_matrix[:-1]
    similarity = cosine_similarity(resume_vector, job_vectors)

    # Top jobs
    top_indices = similarity[0].argsort()[::-1][:3]

    print("\n🔹 TOP JOB RECOMMENDATIONS:\n")

    for i in top_indices:
        print("Job Title:", jobs_df.iloc[i]['job_title'])
        print("Match Score:", round(similarity[0][i], 2))
        print("-"*40)