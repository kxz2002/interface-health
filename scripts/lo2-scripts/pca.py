import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def top_features(component_df, components, top):
    selected_features = set()
    for i in range(components):
        print(f"Top {top} features contributing to PC{i}:")
        values = component_df.iloc[i].abs().sort_values(ascending=False).head(top)
        print(values)
        selected_features.update(values.index)
    selected_features = sorted(list(selected_features))
    print(
        f"Selected {len(selected_features)} features from TOP-{TOP_features} of {N_components} best components:"
    )
    for i in selected_features:
        print(i)
    return selected_features


def calc_pca(features):
    pca = PCA()

    result = pca.fit_transform(features)
    components = pca.components_
    variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = variance_ratio.cumsum()
    component_df = pd.DataFrame(components, columns=df.columns)
    plt.plot(cumulative_variance[:15])
    return result, component_df, variance_ratio, cumulative_variance


# Raw data
df = pd.read_csv("./metrics/light-oauth2-data-1719592986.csv", index_col=None)
df = df.drop(columns=["timestamp", "run_end", "test_name"])

# Split the data into correct test and errors tests
df_correct = df.iloc[0:13]  # Correct test
df_errors = df.iloc[13:]  # All the errors

features = df.values
features_correct = df_correct.values
features_errors = df_errors.values

# Scaling (seems to have not effect)
# from sklearn.preprocessing import StandardScaler
# features_scaled = StandardScaler().fit_transform(features)

_, component_df, _, _ = calc_pca(features)
_, component_df_correct, _, _ = calc_pca(features_correct)
_, component_df_errors, _, _ = calc_pca(features_errors)

TOP_features = 5
N_components = 3  # From the plot 3 components are enough
print("\nFull time-series")
selected_features = top_features(
    component_df, components=N_components, top=TOP_features
)
print("\nCorrect test only")
selected_features_correct = top_features(
    component_df_correct, components=N_components, top=TOP_features
)
print("\nError tests only")
selected_features_errors = top_features(
    component_df_errors, components=N_components, top=TOP_features
)

# Scaled
# _, component_df_scaled, _, _ = calc_pca(features_scaled)
# selected_features_scaled = top_features(component_df_scaled, components=N_components, top=TOP_features)

plt.show()
