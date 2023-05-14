import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def get_data():
    df = pd.read_csv(r'tech_layoffs.csv')

    df = df.drop(["additional_notes"], axis=1)

    if df['total_layoffs'] is not float:
        df.drop(df.loc[df['total_layoffs'] == 'Unclear'].index, inplace=True)

    if df['impacted_workforce_percentage'] is not float:
        df.drop(df.loc[df['impacted_workforce_percentage'] == 'Unclear'].index , inplace=True)

    df['reported_date'] = pd.to_datetime(df['reported_date'])

    #extracting date
    df['reported_year'] = df['reported_date'].dt.year
    df['reported_month'] = df['reported_date'].dt.month

    return df


def elbow_method():
    df = get_data()
    x = df.iloc[:, [1, 2]]
    x  = np.array(x)
    # Collecting the distortions into list
    distortions = []
    K = range(1,10)

    # Elbow method
    for k in K:
         kmeanModel = KMeans(n_clusters=k)
         kmeanModel.fit(x)
         distortions.append(kmeanModel.inertia_)
    # Plotting the distortions
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, "bx-")
    plt.xlabel("k")
    plt.ylabel("Distortion")
    plt.title("The Elbow Method showing the optimal clusters")
    #plt.show()

def k_means():

