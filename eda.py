# EDA for Habit Tracking Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("habit_tracking_synthetic.csv")


print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nPreview:\n", df.head())

# 3. Feature engineering

task_cols = ['leetcode','capstone','projects','misc']

# Total tasks done in a day
df['num_tasks_done'] = df[task_cols].sum(axis=1)

# Active day (at least 1 task)
df['is_active_day'] = (df['num_tasks_done'] >= 1).astype(int)

# Full day (all 3 core tasks)
df['is_full_day'] = ((df[['leetcode','capstone','projects']].sum(axis=1) == 3).astype(int))

# Perfect + bonus day (all 4 tasks)
df['is_bonus_day'] = (df['num_tasks_done'] == 4).astype(int)

# Convert date
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values("date")

# 4. Basic stats

print("\nTask completion counts:\n", df[task_cols].sum())
print("\nAverage sleep hours:", df['sleep_hours'].mean())
print("\nAverage daily progress:", df['daily_productivity'].mean())

# EDA Visualizations

#  Task completion frequency 
df[task_cols].sum().plot(kind="bar", figsize=(6,4))
plt.title("Total Completions per Task")
plt.ylabel("Count")
plt.show()

#  Tasks done per day 
plt.figure(figsize=(8,4))
sns.histplot(df['num_tasks_done'], bins=[0,1,2,3,4], discrete=True)
plt.title("Distribution of Tasks Done Per Day")
plt.xlabel("Tasks Done")
plt.ylabel("Frequency")
plt.show()

#  Sleep vs daily productivity 
plt.figure(figsize=(6,4))
sns.scatterplot(x='sleep_hours', y='daily_productivity', data=df, hue='is_active_day')
plt.title("Sleep Hours vs Daily Productivity")
plt.show()

#  Correlation heatmap 
plt.figure(figsize=(10,4))
sns.heatmap(df[['sleep_hours','sleep_quality','mood','stress','energy','daily_productivity']].corr(),
            annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap")
plt.show()

# 6. Streak Logic (for plotting later)

df['streak'] = (df['is_active_day'] *
                (df['is_active_day'].groupby((df['is_active_day'] != df['is_active_day'].shift()).cumsum())
                 .cumcount()+1))
print("\nStreak preview:\n", df[['date','is_active_day','streak']].head(15))


# 7. Weekly & Monthly Summary

def generate_report(df, start_date=None, end_date=None, freq='W'):
    temp = df.copy()
    if start_date: temp = temp[temp['date'] >= pd.to_datetime(start_date)]
    if end_date: temp = temp[temp['date'] <= pd.to_datetime(end_date)]
    
    grouped = temp.groupby(temp['date'].dt.to_period(freq))
    reports = []
    
    for period, g in grouped:
        total_days = g.shape[0]
        active_days = g['is_active_day'].sum()
        avg_progress = g['daily_productivity'].mean()
        streaks = (g['is_active_day']*(g['is_active_day'].groupby(
                    (g['is_active_day'] != g['is_active_day'].shift()).cumsum()).cumcount()+1)).max()
        perfect_days = (g['num_tasks_done'] == 4).sum()

        habit_means = {
            'LeetCode': g['leetcode'].mean(),
            'Capstone': g['capstone'].mean(),
            'Projects': g['projects'].mean(),
            'Misc': g['misc'].mean()
        }
        top_habit = max(habit_means, key=habit_means.get)

        # Correlation with productivity
        corr_stress = g['daily_productivity'].corr(g['stress'])
        corr_sleep = g['daily_productivity'].corr(g['sleep_hours'])
        corr_energy = g['daily_productivity'].corr(g['energy'])

        reports.append({
            'Period': str(period),
            'Active Days %': round(active_days/total_days*100,1),
            'Avg Progress': round(avg_progress,1),
            'Longest Streak': int(streaks) if not np.isnan(streaks) else 0,
            'Perfect Days (4/4)': int(perfect_days),
            'Top Habit': top_habit,
            'Stress→Progress Corr': round(corr_stress,2),
            'Sleep→Progress Corr': round(corr_sleep,2),
            'Energy→Progress Corr': round(corr_energy,2)
        })
    return pd.DataFrame(reports)

# Generate weekly and monthly reports
weekly_report = generate_report(df, freq='W')
monthly_report = generate_report(df, freq='M')

print("\nWEEKLY REPORT")
print(weekly_report.to_string(index=False))

print("\nMONTHLY REPORT")
print(monthly_report.to_string(index=False))
