from data_preprocessing import load_data, clean_data
from visualization import plot_feature_distributions, plot_correlation_matrix, plot_genre_counts
from clustering_model import perform_clustering

def main():
    # Step 1: Load data
    data_path = "data/spotify_dataset.csv"
    df = load_data(data_path)
    if df is None:
        return

    # Step 2: Clean data
    df = clean_data(df)

    # Step 3: Visualizations
    plot_feature_distributions(df)
    plot_correlation_matrix(df)
    plot_genre_counts(df, genre_column="playlist_genre")

    # Step 4: Clustering
    features_to_use = ["danceability", "energy"]  # Change features based on dataset
    df, model = perform_clustering(df, features=features_to_use, n_clusters=5)

    # Step 5: Save clustered data
    df.to_csv("data/spotify_clustered.csv", index=False)
    print("✅ Clustered data saved to data/spotify_clustered.csv")

if __name__ == "__main__":
    main()

