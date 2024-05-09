from preprocess import preprocess_text
from feature_extraction import tfidf_features
from similarity_calculation import calculate_similarity


def detect_plagiarism(query_text, reference_texts, threshold=0.8):
    preprocessed_query = preprocess_text(query_text)
    preprocessed_references = [preprocess_text(text) for text in reference_texts]

    features_query, vectorizer = tfidf_features([preprocessed_query] + preprocessed_references)
    features_references = features_query[1:]
    features_query = features_query[:1]  # Extract query feature separately

    similarity_scores = calculate_similarity(features_query, features_references)

    print("Query features shape:", features_query.shape)
    print("Reference features shape:", features_references.shape)
    print("Similarity scores:", similarity_scores)

    plagiarism_results = []
    for i, score in enumerate(similarity_scores[0]):
        if score >= threshold:
            plagiarism_results.append({
                'reference_text': reference_texts[i],
                'similarity_score': score
            })

    return plagiarism_results


if __name__ == "__main__":

    with open("example_document.txt", "r") as file:
        example_document = file.read()

    with open("reference1.txt", "r") as file:
        reference_text1 = file.read()

    with open("reference2.txt", "r") as file:
        reference_text2 = file.read()

    reference_texts = [reference_text1, reference_text2]

    try:

        results = detect_plagiarism(example_document, reference_texts)

        if results:
            print("Plagiarized content detected:")
            for result in results:
                print("Reference:", result['reference_text'])
                print("Similarity Score:", result['similarity_score'])
                print()
        else:
            print("No plagiarism detected.")
    except ValueError as e:
        print("Error:", e)
