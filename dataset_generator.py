import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Parameters
num_days = 90
start_date = datetime.today() - timedelta(days=num_days)

# Function to simulate mood, stress, energy based on sleep
def generate_context(sleep_hours):
    if sleep_hours < 5:
        sleep_quality = random.randint(1, 3)
        mood = random.randint(1, 3)
        stress = random.randint(6, 10)
        energy = random.randint(1, 4)
    elif 5 <= sleep_hours <= 7:
        sleep_quality = random.randint(4, 7)
        mood = random.randint(4, 6)
        stress = random.randint(3, 6)
        energy = random.randint(4, 6)
    else:  # good sleep
        sleep_quality = random.randint(7, 10)
        mood = random.randint(6, 9)
        stress = random.randint(1, 4)
        energy = random.randint(6, 9)
    return sleep_quality, mood, stress, energy

# Generate dataset
data = []
for i in range(num_days):
    date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
    
    # Sleep hours
    sleep_hours = round(np.random.normal(7, 1.5), 1)
    sleep_hours = max(3, min(sleep_hours, 10))  # clamp between 3–10 hrs
    sleep_quality, mood, stress, energy = generate_context(sleep_hours)

    # Habits (core + misc)
    leetcode = np.random.binomial(1, 0.7)   # 70% chance
    capstone = np.random.binomial(1, 0.6)   # 60% chance
    projects = np.random.binomial(1, 0.5)   # 50% chance
    misc = np.random.binomial(1, 0.4)       # 40% chance

    # Productivity logic
    productivity = 5  # baseline
    productivity += leetcode + capstone + projects  # core habits add equally

    # Misc logic
    if leetcode and capstone and projects and misc:
        productivity += 1  # misc as bonus only if all 3 cores are done

    # Adjust based on context
    productivity += (mood - 5) * 0.2
    productivity += (energy - 5) * 0.3
    productivity -= (stress - 5) * 0.2
    productivity = round(max(1, min(productivity, 10)))  # clamp 1–10 scale

    data.append([
        date, leetcode, capstone, projects, misc,
        sleep_hours, sleep_quality, mood, stress, energy, productivity
    ])

# Create DataFrame
columns = ["date", "leetcode", "capstone", "projects", "misc",
           "sleep_hours", "sleep_quality", "mood", "stress", "energy",
           "daily_productivity"]

df = pd.DataFrame(data, columns=columns)


df.to_csv("habit_tracking_synthetic.csv", index=False)

print("✅ Synthetic dataset generated: habit_tracking_synthetic.csv")
print(df.head(10))
