# Random Projection LSH for Product Duplicate Detection

Code for product duplicate detection using Random Projection Locality-Sensitive Hashing (LSH) combined with hierarchical clustering (single linkage).

# Overview

1. **Data Loading and Cleaning:**  
   We mainly standardize units, remove special characters, and filter out noise. Additionally, we repeatedly add the brand name to the product text to give it higher weight.

2. **Vectorization (TF-IDF):**  
   After cleaning, transform each product's title and extracted numerical features to a TF-IDF vector. 

3. **Locality-Sensitive Hashing (LSH) via Random Projection:**  
   Random projection LSH to map TF-IDF vectors into binary signatures. Then take pairs for buckets with more than one product to generate candidate pairs.

4. **Clustering for Duplicate Detection:**  
   Using the candidate pairs, build a distance matrix and run single linkage hierarchical clustering. Products that cluster tightly below a threshold are considered duplicates..

5. **Evaluation:**  
   Randomly sample 63% of the data with 20 bootstraps.  F1, F1*, Pair Quality (PQ), and Pair Completeness (PC) used for evaluation. 
## Files

Main code is main.py. Dataset is TVs-all-merged.json.
