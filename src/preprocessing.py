import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.paths import PCA_BASE_CSV, PCA_VARIANCE_CSV

# Heart rates at or below this value are treated as data-entry errors.
MIN_HEART_RATE = 40


def clean_data(df):
    """Single source of truth for row cleaning.

    Used by preprocess() and by the accuracy evaluation so that labels and
    ground truth always refer to exactly the same rows.
    """
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[df['HeartRate'] > MIN_HEART_RATE].reset_index(drop=True)
    return df


def preprocess(df, printInfo=False):
    """Clean the data, split features/target, and standardize the features.

    Pure function: it does not write any files.
    """
    df = clean_data(df)

    # Separate features from target variable
    X = df.drop(columns=['RiskLevel'])
    y = df['RiskLevel']

    # Scale: fit-transform features
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled_array, columns=X.columns)

    if printInfo:
        print(df.describe().T[['mean', 'std', 'min', '50%', 'max']])

    return X_scaled_df, y


def save_pca_base_coords(X_scaled_df):
    """Project the scaled features onto 3 PCs and save them for plotting.

    PC1/PC2 are used by the 2D plots, PC1/PC2/PC3 by the 3D plots. The
    explained-variance ratio of each component is saved alongside for axis labels.
    """
    pca = PCA(n_components=3, random_state=42)
    pca_coords = pca.fit_transform(X_scaled_df)
    pca_base = pd.DataFrame(pca_coords, columns=['PC1', 'PC2', 'PC3'])
    pca_base.to_csv(PCA_BASE_CSV, index=False)

    pd.DataFrame({
        'Component': ['PC1', 'PC2', 'PC3'],
        'ExplainedVariance': pca.explained_variance_ratio_,
    }).to_csv(PCA_VARIANCE_CSV, index=False)
    return pca_base
