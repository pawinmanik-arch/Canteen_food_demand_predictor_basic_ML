import pandas as pd
import random

# Number of records
num_records = 10000

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weathers = ['Sunny', 'Rainy', 'Cloudy']
exam_days = ['Yes', 'No']
food_types = ['Snack', 'Meal', 'Beverage']

data = []

for _ in range(num_records):
    day = random.choice(days)
    weather = random.choice(weathers)
    exam_day = random.choice(exam_days)
    food = random.choice(food_types)
    price = random.randint(20, 60)

    # Demand logic (realistic rules)
    if exam_day == 'Yes' and price <= 35:
        demand = 'High'
    elif weather == 'Rainy' and food == 'Snack':
        demand = 'Low'
    elif price > 50:
        demand = 'Low'
    else:
        demand = random.choice(['High', 'Low'])

    data.append([day, weather, exam_day, price, food, demand])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    'Day', 'Weather', 'Exam_Day', 'Price', 'Food_Type', 'Demand'
])

# Save CSV
df.to_csv("canteen_data.csv", index=False)

print(f"✅ canteen_data.csv with {num_records} records created successfully!")