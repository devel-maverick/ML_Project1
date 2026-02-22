from preprocessing.preprocess import load_data, preprocess_entire_dataframe
from preprocessing.preprocess import tfidf_vectorizer

df = load_data("data/WELFake_Dataset.csv")
df = preprocess_entire_dataframe(df)

print(df[["content","processed_content"]].head())


matrix,vectorizer = tfidf_vectorizer(df)

print(matrix.shape)
print(vectorizer.get_feature_names_out()[:10])
print("\nSample for row 4:")
print(matrix[4])



