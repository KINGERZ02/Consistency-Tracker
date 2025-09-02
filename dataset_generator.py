import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Parameters
num_days = 90
start_date = datetime.today() - timedelta(days=num_days)

# Helper functions
def generate_sleep_hours():
    return round(np.random.normal(7, 1.5), 1)  # avg ~7 hrs, some variance

def generate_sleep_quality(hours):
    if hours < 5:
        return random.choice([1, 2, 3])  # poor sleep
    elif hours < 7:
        return random.choice([4, 5, 6])
    else:
        return random.choice([6, 7, 8, 9])

def generate_mood(sleep_quality):
    if sleep_quality <= 3:
        return random.randint(1, 4)
    elif sleep_quality <= 6:
        return random.randint(3, 7)
    else:
        return random.randint(6, 10)

def generate_stress(mood):
    return max(1, min(10, 10 - mood + random.randint(-1, 1)))

def generate_energy(sleep_hours, mood):
    base = int((sleep_hours/8) * 5 + mood/2)
    return max(1, min(10, base + random.randint(-2, 2)))

def generate_habit_completion(energy, stress):
    prob = (energy/10) * (1 - (stress/15))
    return 1 if random.random() < prob else 0

def generate_productivity(habits, mood, stress, energy):
    score = 3 + sum(habits) + (mood/3) - (stress/5) + (energy/3)
    return int(max(1, min(10, round(score))))

# Generate data
records = []
for i in range(num_days):
    date = (start_date + timedelta(days=i)).date()
    sleep_hours = generate_sleep_hours()
    sleep_quality = generate_sleep_quality(sleep_hours)
    mood = generate_mood(sleep_quality)
    stress = generate_stress(mood)
    energy = generate_energy(sleep_hours, mood)

    # habits
    exercise = generate_habit_completion(energy, stress)
    reading = generate_habit_completion(energy, stress)
    meditation = generate_habit_completion(energy, stress)
    journaling = generate_habit_completion(energy, stress)

    habits = [exercise, reading, meditation, journaling]

    productivity = generate_productivity(habits, mood, stress, energy)

    records.append([
        date, round(sleep_hours, 1), sleep_quality, mood, stress, energy,
        exercise, reading, meditation, journaling, productivity
    ])

# Create DataFrame
columns = [
    "date", "sleep_hours", "sleep_quality", "mood", "stress", "energy",
    "exercise", "reading", "meditation", "journaling", "daily_productivity"
]
df = pd.DataFrame(records, columns=columns)

# Save as CSV
df.to_csv("synthetic_habit_dataset.csv", index=False)

print("✅ Dataset generated and saved as synthetic_habit_dataset.csv")
print(df.head())
