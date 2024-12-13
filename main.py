import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import comb
from collections import defaultdict
import itertools
from joblib import Parallel, delayed
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
# DATA PROCESSING

def clean_text(text):
    text = re.sub(r"(?i)\b(inch|inches|”|''|-inch|\"|”)\b", "inch", text)
    text = re.sub(r"(?i)\b(hertz|hz|-hz)\b", "hz", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"(\d+)\s*(inch|hz)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_brands(df):
    brand_set = set()
    for features in df['featuresMap']:
        if features:
            for key, value in features.items():
                if key.lower() in ['brand', 'brand name']:
                    brand_set.add(value.strip().lower())
    for title in df['title']:
        words = title.split()
        for word in words:
            if word.isalpha() and len(word) > 2:
                brand_set.add(word.strip().lower())
    return brand_set

def get_brand(product, known_brands):
    features = product.get('featuresMap', {})
    for key, value in features.items():
        if key.lower() in ['brand', 'brand name']:
            return clean_text(value)
    title = product.get('title', '')
    words = title.split()
    for word in words:
        word_clean = clean_text(word)
        if word_clean.lower() in known_brands:
            return word_clean
    return ''

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    products = []
    for model_id, product_list in data.items():
        for product in product_list:
            product['modelID'] = model_id
            if 'title' in product:
                product['cleaned_title'] = clean_text(product['title'])
            if 'featuresMap' in product:
                cleaned_values = []
                for key in sorted(product['featuresMap']):
                    value = product['featuresMap'][key]
                    if value == "Yes":
                        cleaned_value = clean_text(key)
                        cleaned_values.append(cleaned_value)
                    elif value not in ["No", "0"]:
                        cleaned_value = clean_text(str(value))
                        cleaned_values.append(cleaned_value)
                product['cleaned_featuresMap'] = cleaned_values
                product['shop'] = product.get('shop', '')
                product['title'] = product.get('title', '')
                product['featuresMap'] = product.get('featuresMap', '')
            products.append(product)

    flattened_products = products
    cleaned_data = [(p['modelID'], p['shop'], p['featuresMap'], p['title'], p['cleaned_title'], p['cleaned_featuresMap']) for p in flattened_products]
    df_cleaned = pd.DataFrame(cleaned_data, columns=['modelID','shop','featuresMap','title', 'cleaned_title', 'cleaned_featuresMap'])

    known_brands = extract_brands(df_cleaned)
    df_cleaned['brand'] = df_cleaned.apply(lambda row: get_brand(row, known_brands), axis=1)
    df_cleaned['brand'] = df_cleaned['brand'].fillna('').astype(str)

    df_cleaned['cleaned_featuresMap'] = df_cleaned['cleaned_featuresMap'].apply(lambda x: ', '.join(x))
    df_cleaned['cleaned_featuresMap'] = df_cleaned['cleaned_featuresMap'].apply(
        lambda x: ' '.join(word for word in x.split() if re.search(r'\d', word))
    )

    df_cleaned['combined_text'] = (
        df_cleaned['cleaned_title'] + ' ' +
        df_cleaned['cleaned_featuresMap'] + ' ' +
        (df_cleaned['brand'] + ' ') * 15
    )

    return df_cleaned

#CANDIDATE PAIR GENERATION

def compute_tfidf(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50, min_df=2, max_df=0.8)
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    return tfidf_matrix

def create_buckets(tfidf_matrix, n_components=100):
    # Random projection LSH
    plane_norms = np.random.rand(tfidf_matrix.shape[1], n_components) - 0.5
    projected_matrix = tfidf_matrix @ plane_norms
    buckets = {}
    for i in range(projected_matrix.shape[0]):
        binary_signature = (projected_matrix[i] > 0).astype(int)
        hash_str = ''.join(binary_signature.astype(str))
        if hash_str not in buckets:
            buckets[hash_str] = []
        buckets[hash_str].append(i)
    return buckets

def generate_candidate_pairs(buckets, df):
    candidate_pairs = set()
    for _, indices in buckets.items():
        if len(indices) > 1:
            for i in range(len(indices)):
                idx_i = indices[i]
                shop_i = df.loc[idx_i, 'shop']
                brand_i = df.loc[idx_i, 'brand']
                for j in range(i + 1, len(indices)):
                    idx_j = indices[j]
                    shop_j = df.loc[idx_j, 'shop']
                    brand_j = df.loc[idx_j, 'brand']
                    # Consider pairs from different shops with the same non-empty brand
                    if shop_i != shop_j and brand_i == brand_j and brand_i != '':
                        candidate_pairs.add(tuple(sorted((idx_i, idx_j))))
    return candidate_pairs


#CLUSTERING AND EVALUATION

def evaluate_clustering(unique_products, labels, duplicate_pairs, candidate_pairs):
    cluster_map = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_map[label].append(idx)
    predicted_duplicates = set()
    for cluster_indices in cluster_map.values():
        if len(cluster_indices) > 1:
            for (i1, i2) in itertools.combinations(sorted(cluster_indices), 2):
                predicted_duplicates.add((i1, i2))


    TP = len(predicted_duplicates & duplicate_pairs)
    FP = len(predicted_duplicates - duplicate_pairs)
    FN = len(duplicate_pairs - predicted_duplicates)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
    return TP, FP, FN, precision, recall, f1, predicted_duplicates

def pairwise_cosine_similarity(tfidf_matrix, i, j):
    sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0,0]
    return sim

def bootstrap_evaluation(df, n_bootstrap=3, n_components=50, threshold_candidates=[0.3,0.4,0.5,0.6,0.7]):
    metrics = {'fraction_comparison': [], 'precision': [], 'recall': [], 'f1': [], 'PQ': [], 'PC': [], 'f1_star': []}

    for bootstrap_index in range(n_bootstrap):
        print(f"BOOTSTRAP {bootstrap_index + 1}:")
        sample = df.sample(frac=1, replace=True, random_state=69 + bootstrap_index).reset_index(drop=True)
        sample['product_id'] = sample.apply(lambda row: (row['shop'], row['title']), axis=1)

        unique_product_indices = []
        seen_products = set()
        for idx, product_id in enumerate(sample['product_id']):
            if product_id not in seen_products:
                seen_products.add(product_id)
                unique_product_indices.append(idx)
        unique_products = sample.iloc[unique_product_indices].reset_index(drop=True)
        tfidf_matrix = compute_tfidf(unique_products)
        buckets = create_buckets(tfidf_matrix, n_components=n_components)
        candidate_pairs = generate_candidate_pairs(buckets, unique_products)

        # Actual duplicates
        # model_counts = unique_products.groupby('modelID').agg(
        #     Count=('modelID', 'size')
        # ).reset_index()
        # model_counts['Count'] = model_counts['Count'].astype(int)
        # total_duplicates = sum(comb(count, 2) for count in model_counts['Count'] if count > 1)

        model_to_indices = defaultdict(list)
        for idx, row in unique_products.iterrows():
            model_id = row['modelID']
            shop = row['shop']
            model_to_indices[model_id].append((idx, shop))

        duplicate_pairs = set()
        for indices_shops in model_to_indices.values():
            if len(indices_shops) > 1:
                for (i1, shop1), (i2, shop2) in itertools.combinations(indices_shops, 2):
                    modelID_i = unique_products.iloc[i1]['modelID']
                    modelID_j = unique_products.iloc[i2]['modelID']
                    if shop1 != shop2 and modelID_i == modelID_j:
                        duplicate_pairs.add(tuple(sorted((i1, i2))))
        total_duplicates = len(duplicate_pairs)
        total_possible_pairs = comb(len(unique_products), 2)

        n = len(unique_products)
        large_value = 10
        dist_matrix = np.full((n, n), large_value)
        np.fill_diagonal(dist_matrix, 0.0)

        for (i, j) in candidate_pairs:
            shop_i = unique_products.iloc[i]['shop']
            shop_j = unique_products.iloc[j]['shop']
            brand_i = unique_products.iloc[i]['brand']
            brand_j = unique_products.iloc[j]['brand']

            if shop_i != shop_j and brand_i == brand_j and brand_i != '':
                sim = pairwise_cosine_similarity(tfidf_matrix, i, j)
                dist = 1 - sim
                dist_matrix[i,j] = dist
                dist_matrix[j,i] = dist
            else:
                continue

        best_f1_star = -1
        best_results = None

        for dist_threshold in threshold_candidates:
            clustering = AgglomerativeClustering(
                metric='precomputed',
                linkage='single',
                n_clusters=None,
                distance_threshold=dist_threshold
            ).fit(dist_matrix)

            labels = clustering.labels_
            TP, FP, FN, precision, recall, f1, predicted_duplicates = evaluate_clustering(unique_products, labels, duplicate_pairs, candidate_pairs)
            predicted_duplicates = predicted_duplicates.intersection(candidate_pairs)

            PQ = TP / len(candidate_pairs) if len(candidate_pairs) > 0 else 0
            PC = TP / total_duplicates if total_duplicates > 0 else 0
            f1_star = 2*(PQ*PC)/(PQ+PC) if (PQ+PC)>0 else 0
            fraction_comparison = len(candidate_pairs)/total_possible_pairs if total_possible_pairs>0 else 0

            if f1_star > best_f1_star:
                best_f1_star = f1_star
                best_results = (precision, recall, f1, PQ, PC, f1_star, TP, fraction_comparison, candidate_pairs, total_possible_pairs, total_duplicates, unique_products)

        precision, recall, f1, PQ, PC, f1_star, TP, fraction_comparison, candidate_pairs, total_possible_pairs, total_duplicates, unique_products = best_results

        percentage_reduction = (1 - fraction_comparison)*100 if total_possible_pairs>0 else 0
        print(f"Maximum number of comparisons: {total_possible_pairs}")
        print(f"Actual number of comparisons (Number of candidate pairs after LSH): {len(candidate_pairs)}")
        print(f"Percentage reduction in comparisons: {percentage_reduction:.2f}%")
        print(f"Number of duplicates in bootstrap sample: {total_duplicates}")

        actual_duplicates = 0
        for (i, j) in candidate_pairs:
            modelID_i = unique_products.iloc[i]['modelID']
            modelID_j = unique_products.iloc[j]['modelID']
            shop_i = unique_products.iloc[i]['shop']
            shop_j = unique_products.iloc[j]['shop']
            if modelID_i == modelID_j and shop_i != shop_j:
                actual_duplicates += 1
        print(f"True duplicates in candidate pairs: {actual_duplicates}")
        print(f"Number of duplicates found by our algorithm: {TP}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Measure: {f1:.4f}")
        print(f"Pair Quality (PQ): {PQ:.4f}")
        print(f"Pair Completeness (PC): {PC:.4f}")
        print(f"F1*-Measure: {f1_star:.4f}")
        print("")

        metrics['fraction_comparison'].append(fraction_comparison)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['PQ'].append(PQ)
        metrics['PC'].append(PC)
        metrics['f1_star'].append(f1_star)

    return {metric: np.mean(values) for metric, values in metrics.items()}


# Main Execution


if __name__ == "__main__":
    start_time = time.time()
    file_path = r"C:\Users\s00336\Desktop\Master\block2\cs\indi\TVs-all-merged\TVs-all-merged.json"
    df = load_data(file_path)

    first_range = np.arange(1, 81, 2)
    n_range = first_range
    results_list = Parallel(n_jobs=-1)(
        delayed(bootstrap_evaluation)(df, n_bootstrap=20, n_components=n) for n in n_range
    )

    for res, n in zip(results_list, n_range):
        res['n_components'] = n

    aggregated_results = pd.DataFrame(results_list)

    # PQ
    plt.figure(figsize=(14, 8))
    plt.plot(aggregated_results['fraction_comparison'], aggregated_results['PQ'], marker='o',
             linestyle='-', color='b', label='Pair Quality')
    plt.xlabel('Fraction of Comparisons')
    plt.ylabel('Pair Quality')
    plt.title('Pair Quality vs Fraction of Comparisons')
    plt.grid(True)
    plt.legend()
    plt.show()

    # PC
    plt.figure(figsize=(14, 8))
    plt.plot(aggregated_results['fraction_comparison'], aggregated_results['PC'], marker='o',
             linestyle='-', color='g', label='Pair Completeness')
    plt.xlabel('Fraction of Comparisons')
    plt.ylabel('Pair Completeness')
    plt.title('Pair Completeness vs Fraction of Comparisons')
    plt.grid(True)
    plt.legend()
    plt.show()

    # F1*
    plt.figure(figsize=(14, 8))
    plt.plot(aggregated_results['fraction_comparison'], aggregated_results['f1_star'], marker='o',
             linestyle='-', color='r', label='F1*-Measure')
    plt.xlabel('Fraction of Comparisons')
    plt.ylabel('F1*-Measure')
    plt.title('F1*-Measure vs Fraction of Comparisons')
    plt.grid(True)
    plt.legend()
    plt.show()

    # F1
    plt.figure(figsize=(14, 8))
    plt.plot(aggregated_results['fraction_comparison'], aggregated_results['f1'], marker='o', linestyle='-',
             color='m', label='F1-Measure')
    plt.xlabel('Fraction of Comparisons')
    plt.ylabel('F1-Measure')
    plt.title('F1-Measure vs Fraction of Comparisons')
    plt.grid(True)
    plt.legend()
    plt.show()

    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
