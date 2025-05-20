import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\hp\Desktop\StudentsPerformance.csv")

print(df.head(20))
print("Dataset Info:")
df.info()
print("\nSummary Statistics:\n", df.describe())

df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]


print("\nMissing Values:\n", df.isnull().sum())

df['average_score'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1)

def classify_performance(score):
    if score >= 90:
        return 'Excellent'
    elif score >= 75:
        return 'Good'
    elif score >= 60:
        return 'Average'
    else:
        return 'Poor'

df['performance_level'] = df['average_score'].apply(classify_performance)


print("\nAverage Score by Gender:\n", df.groupby('gender')['average_score'].mean())

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='gender', y='average_score')
plt.title('Average Score by Gender')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='performance_level', order=['Poor', 'Average', 'Good', 'Excellent'])
plt.title('Student Performance Levels')
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df[['math_score', 'reading_score', 'writing_score']].corr(), annot=True, cmap='Blues')
plt.title('Correlation Between Scores')
plt.show()

df.to_csv("cleaned_students_performance.csv", index=False)
print("\nCleaned data exported to 'cleaned_students_performance.csv'")
