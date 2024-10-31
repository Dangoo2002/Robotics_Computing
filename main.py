import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sns.set(style="whitegrid")


df = pd.read_csv('data/robotics_data.csv')  


df['Year'] = df['Year'].astype(int)
df['Industry'] = df['Industry'].astype(str) 
df['Robots_Adopted'] = pd.to_numeric(df['Robots_Adopted'], errors='coerce')
df['Productivity_Gain'] = pd.to_numeric(df['Productivity_Gain'], errors='coerce')
df['Cost_Savings'] = pd.to_numeric(df['Cost_Savings'], errors='coerce')
df['Jobs_Displaced'] = pd.to_numeric(df['Jobs_Displaced'], errors='coerce')
df['Training_Hours'] = pd.to_numeric(df['Training_Hours'], errors='coerce')

print("Summary of the Dataset:")
print(df.describe())


df = df.dropna(subset=['Robots_Adopted', 'Productivity_Gain', 'Cost_Savings', 'Jobs_Displaced', 'Training_Hours'])

robots_by_year = df.groupby('Year')['Robots_Adopted'].sum()
print("\nTotal Robots Adopted by Year:")
print(robots_by_year)


plt.figure(figsize=(10, 6))
sns.lineplot(x=robots_by_year.index, y=robots_by_year.values, marker="o")
plt.title("Total Robots Adopted by Year")
plt.xlabel("Year")
plt.ylabel("Total Robots Adopted")
plt.savefig(os.path.join(output_dir, "total_robots_adopted_by_year.png"))
plt.show()


plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="Year", y="Productivity_Gain", hue="Industry", marker="o")
plt.title("Productivity Gain Over Years by Industry")
plt.xlabel("Year")
plt.ylabel("Productivity Gain (%)")
plt.legend(title="Industry")
plt.savefig(os.path.join(output_dir, "productivity_gain_over_years_by_industry.png"))
plt.show()


plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="Year", y="Cost_Savings", hue="Industry", marker="o")
plt.title("Cost Savings Over Years by Industry (in Million USD)")
plt.xlabel("Year")
plt.ylabel("Cost Savings (M USD)")
plt.legend(title="Industry")
plt.savefig(os.path.join(output_dir, "cost_savings_over_years_by_industry.png"))
plt.show()


plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="Year", y="Jobs_Displaced", hue="Industry", marker="o")
plt.title("Job Displacement Trends by Industry")
plt.xlabel("Year")
plt.ylabel("Jobs Displaced")
plt.legend(title="Industry")
plt.savefig(os.path.join(output_dir, "job_displacement_trends_by_industry.png"))
plt.show()


plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="Year", y="Training_Hours", hue="Industry", marker="o")
plt.title("Training Hours for Skill Development by Industry")
plt.xlabel("Year")
plt.ylabel("Training Hours")
plt.legend(title="Industry")
plt.savefig(os.path.join(output_dir, "training_hours_by_industry_over_years.png"))
plt.show()


robots_by_industry_year = df.pivot_table(values='Robots_Adopted', index='Year', columns='Industry', aggfunc='sum')
robots_by_industry_year.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
plt.title("Robots Adopted by Industry Over Years")
plt.xlabel("Year")
plt.ylabel("Robots Adopted")
plt.savefig(os.path.join(output_dir, "robots_adopted_by_industry_over_years.png"))
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Robotics Metrics")
plt.savefig(os.path.join(output_dir, "correlation_matrix_of_robotics_metrics.png"))
plt.show()

industries = df['Industry'].unique()
for industry in industries:
    industry_data = df[df['Industry'] == industry]
    print(f"\nSummary Statistics for {industry}:")
    print(industry_data.describe())

    plt.figure(figsize=(10, 5))
    sns.lineplot(x="Year", y="Productivity_Gain", data=industry_data, marker="o")
    plt.title(f"Productivity Gain Over Years in {industry}")
    plt.xlabel("Year")
    plt.ylabel("Productivity Gain (%)")
    plt.savefig(os.path.join(output_dir, f"productivity_gain_over_years_{industry}.png"))
    plt.show()
