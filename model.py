import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid threading issues
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def train_and_plot():
    data = {
        'Age': [19,21,20,23,31,22,35,23,64,30,67,35,58,24,37,22,35,20,52,35,
                35,25,46,31,54,29,45,35,40,23,60,21,53,18,49,21,42,30,36,20],
        'Annual_Income': [15,15,16,16,17,17,18,18,19,19,20,20,20,21,21,23,23,24,
                          24,25,25,28,28,29,29,30,30,33,33,34,34,37,37,38,38,39,39,39,40,40],
        'Spending_Score': [39,81,6,77,40,76,6,94,3,72,14,99,15,77,13,79,35,66,29,
                           98,35,73,5,73,10,78,13,78,30,87,13,75,20,89,14,81,17,78,23,90]
    }
    df = pd.DataFrame(data)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    model = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = model.fit_predict(scaled)

    # Custom styling for the plot to match the dark dashboard
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=df['Annual_Income'],
        y=df['Spending_Score'],
        hue=df['Cluster'],
        palette='viridis',
        s=150,
        edgecolor='white',
        alpha=0.8
    )
    plt.title('Customer Segmentation (Income vs Spending)', fontsize=16, pad=20)
    plt.xlabel('Annual Income (k$)', fontsize=12)
    plt.ylabel('Spending Score (1-100)', fontsize=12)
    plt.legend(title='Cluster Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    os.makedirs('static', exist_ok=True)
    plt.savefig('static/plot.png', transparent=True)
    plt.close()

    summary = df.groupby('Cluster')[['Age','Annual_Income','Spending_Score']].mean().round(1)
    return summary.to_dict('index')
