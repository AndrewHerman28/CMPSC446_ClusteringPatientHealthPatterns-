import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def preprocess(df, printInfo=False):
    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Remove out of range heart rates
    df = df[df['HeartRate'] > 40]

    # Seperate features from target variable
    X = df.drop(columns=['RiskLevel'])
    y = df['RiskLevel']

    # Scale: Fit-transform features
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled_array, columns=X.columns)

    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled_df)
    pca_base = pd.DataFrame(pca_coords, columns=['PC1', 'PC2'])
    pca_base.to_csv("data/processed/pca_base_coords.csv", index=False)

    if printInfo:
        print(df.describe().T[['mean', 'std', 'min', '50%', 'max']])

    return X_scaled_df, y

