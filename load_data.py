import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "resume_matcher.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

print("Loading CSV file...")
df = pd.read_csv('data/job_descriptions.csv', nrows=2000, on_bad_lines='skip', engine='python')

print("Cleaning columns...")
df_clean = pd.DataFrame()
df_clean['job_title']        = df['Job Title']
df_clean['company']          = df['Company']
df_clean['required_skills']  = df['skills']
df_clean['job_role']         = df['Role']
df_clean['experience_level'] = df['Experience']
df_clean['description']      = df['Job Description']

if 'application_url' in df.columns:
    df_clean['application_url'] = df['application_url'].fillna("Application link unavailable")
    # Replace empty strings with fallback
    df_clean['application_url'] = df_clean['application_url'].apply(lambda u: u.strip() if str(u).strip() else "Application link unavailable")
else:
    df_clean['application_url'] = "Application link unavailable"

# Ensure 1-indexed sequential integer ID
df_clean.insert(0, 'id', range(1, len(df_clean) + 1))

print("Saving to database...")
conn = get_connection()
df_clean.to_sql('jobs', conn, if_exists='replace', index=False)
conn.close()

print(f"Done! Loaded {len(df_clean)} jobs successfully!")