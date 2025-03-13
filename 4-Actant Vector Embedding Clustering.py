# =======================
# 1) INSTALL PACKAGES
# =======================
!pip install sentence-transformers openai umap-learn matplotlib openpyxl scikit-learn hdbscan


# =======================
# 2) IMPORT LIBRARIES
# =======================
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
import openai
import os
import json
from sklearn.metrics import silhouette_samples, silhouette_score
import hdbscan  # For HDBSCAN
from google.colab import files

# =======================
# 3) OPENAI CONFIG
# =======================
openai.api_key = "sk-proj-_A"  # Replace with your actual OpenAI API key

# Configuration
MIN_K = 1
MAX_K = 20

# =======================
# 4) FILE UPLOAD & READING
# =======================
uploaded = files.upload()
file_name = next(iter(uploaded))
file_extension = os.path.splitext(file_name)[1].lower()

if file_extension == '.csv':
    df = pd.read_csv(file_name)
elif file_extension in ['.xls', '.xlsx']:
    df = pd.read_excel(file_name)
else:
    raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

# =======================
# 5) LOAD SENTENCE TRANSFORMER
# =======================
model = SentenceTransformer('all-mpnet-base-v2')

# =======================
# 6) FIND BEST HDBSCAN CLUSTERING
# =======================
def find_optimal_clusters_hdbscan(embedding_data, min_k, max_k):
    """
    Iterate over min_cluster_size in [min_k, max_k], run HDBSCAN each time,
    compute silhouette on non-outlier points, pick the best silhouette solution.

    Returns:
      best_param (int): which min_cluster_size gave best silhouette
      best_score (float): that best silhouette
      best_labels (np.array): cluster labels from best solution, outliers = -1
    """
    best_param = None
    best_score = -1
    best_labels = None

    print("Testing different min_cluster_size values for silhouette scores:")
    for min_size in range(min_k, max_k + 1):
        # Basic HDBSCAN config:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_size,
            min_samples=1,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(embedding_data)

        # We measure silhouette only on non-outlier points
        non_outlier_mask = (labels != -1)
        distinct_clusters = np.unique(labels[non_outlier_mask])

        if len(distinct_clusters) > 1:
            score = silhouette_score(embedding_data[non_outlier_mask],
                                     labels[non_outlier_mask],
                                     metric='euclidean')
        else:
            score = -1

        print(f"  min_cluster_size={min_size}, #clusters(excl. outliers)={len(distinct_clusters)}, silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_param = min_size
            best_labels = labels

    return best_param, best_score, best_labels

# =======================
# 7) UMAP + HDBSCAN + GPT NAMES
# =======================
def cluster_and_name_actant(actant_column):
    """
    For a given actant column (like 'Opponent_Actant'),
    1) gather text data,
    2) embed with sentence-transformers,
    3) run UMAP with multiple dimension candidates,
    4) run HDBSCAN for each dimension, searching min_cluster_size in [MIN_K..MAX_K],
    5) pick the best silhouette solution,
    6) label outliers as -1,
    7) name each cluster with GPT,
    8) store results in df columns.
    """
    # Make sure column exists
    if actant_column not in df.columns:
        raise ValueError(f"No column '{actant_column}' in the DataFrame.")

    # Text data
    actant_data = df[actant_column].fillna('').tolist()

    # Embeddings
    print(f"\nGenerating embeddings for {actant_column}...")
    embeddings = model.encode(actant_data, show_progress_bar=True)

    # We'll try multiple UMAP dims
    dim_candidates = [50, 100, 200, 300, 400, 500]
    best_global_score = -1
    best_global_dim = None
    best_global_labels = None
    best_global_umap = None

    print(f"Testing UMAP dimensions {dim_candidates} for {actant_column}...")
    for dim in dim_candidates:
        print(f"\n  -- UMAP dimension={dim}...")
        reducer = umap.UMAP(n_components=dim, random_state=42)
        umap_embedding = reducer.fit_transform(embeddings)

        # Attempt HDBSCAN with min_cluster_size in [MIN_K..MAX_K]
        best_param, best_score, best_labels = find_optimal_clusters_hdbscan(umap_embedding, MIN_K, MAX_K)

        print(f"    => Best min_cluster_size={best_param}, silhouette={best_score:.4f} for dimension={dim}")
        if best_score > best_global_score:
            best_global_score = best_score
            best_global_dim = dim
            best_global_labels = best_labels
            best_global_umap = umap_embedding

    print(f"\n*** Best overall dimension={best_global_dim}, best silhouette={best_global_score:.4f}")

    if best_global_labels is None:
        raise ValueError(f"No valid clustering solution found for {actant_column}.")

    # Number of actual (non-outlier) clusters:
    non_outlier_mask = (best_global_labels != -1)
    distinct_clusters = np.unique(best_global_labels[non_outlier_mask])
    n_clusters = len(distinct_clusters)
    n_outliers = np.sum(best_global_labels == -1)

    print(f"For {actant_column}, HDBSCAN found {n_clusters} clusters (excluding {n_outliers} outliers).")

    # Save cluster labels
    df[f'{actant_column}_Cluster'] = best_global_labels
    # Save numeric UMAP embeddings as strings
    df[f'{actant_column}_UMAP_Embedding'] = [str(list(vec)) for vec in best_global_umap]

    # Compute silhouette samples only if 2+ non-outlier clusters
    if n_clusters > 1:
        sample_silhouette_values = np.zeros(len(best_global_labels))
        silhouette_vals = silhouette_samples(
            best_global_umap[non_outlier_mask],
            best_global_labels[non_outlier_mask]
        )
        sample_silhouette_values[non_outlier_mask] = silhouette_vals
    else:
        sample_silhouette_values = [0]*len(best_global_labels)

    df[f'{actant_column}_Silhouette_Score'] = sample_silhouette_values

    # Name each non-outlier cluster with GPT
    cluster_names = {}
    for cluster_id in distinct_clusters:
        cluster_data = df[df[f'{actant_column}_Cluster'] == cluster_id]
        text_samples = cluster_data[actant_column].str.cat(sep='\n')

        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates concise and meaningful cluster names from individual actants ie. abstract concepts."},
            {
                "role": "user",
                "content": f"""
Analyze the following cluster of actants:
{text_samples}

Provide a 1-3 word name summarizing the cluster's universal theme or function that can be applied to all individual actants. If all the individual actants have the same name, dont change it.
"""
            }
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )

        name = response['choices'][0]['message']['content'].strip()
        cluster_names[cluster_id] = name

    # Outliers labeled as "Outlier" for clarity
    named_labels = []
    for lab in best_global_labels:
        if lab == -1:
            named_labels.append("Outlier")
        else:
            named_labels.append(cluster_names[lab])

    df[f'{actant_column}_Narrative_Name'] = named_labels

    # Print distribution
    dist_summary = df.groupby(f'{actant_column}_Cluster')[actant_column].count()
    print(f"\nFinal cluster distribution for {actant_column} (including outliers as -1):\n{dist_summary}")

    # Return a subset DataFrame of relevant columns
    return df[[actant_column,
               f'{actant_column}_Cluster',
               f'{actant_column}_Narrative_Name',
               f'{actant_column}_Silhouette_Score',
               f'{actant_column}_UMAP_Embedding']]

# =======================
# 8) CLUSTER ALL ACTANT ROLES
# =======================
actant_roles = [
    'Subject_Actant',
    'Object_Actant',
    'Sender_Actant',
    'Receiver_Actant',
    'Helper_Actant',
    'Opponent_Actant'
]

all_results = {}
for actant in actant_roles:
    try:
        print(f"\nProcessing {actant}...")
        result = cluster_and_name_actant(actant)
        all_results[actant] = result
    except Exception as e:
        print(f"Error processing {actant}: {e}")
        # If there's an error, store an empty DataFrame so we don't break the entire script.
        all_results[actant] = pd.DataFrame()

# =======================
# 9) SAVE INITIAL RESULTS
# =======================
intermediate_file = 'all_actant_clusters_with_names_hdbscan.xlsx'
with pd.ExcelWriter(intermediate_file) as writer:
    # If *all* are empty, we must write a dummy sheet or Excel will raise an error.
    if not any(df_ is not None and not df_.empty for df_ in all_results.values()):
        dummy_df = pd.DataFrame({"NoData": ["NoActantsProcessed"]})
        dummy_df.to_excel(writer, sheet_name="NoData", index=False)
    else:
        for actant, result_df in all_results.items():
            if result_df is not None and not result_df.empty:
                result_df.to_excel(writer, sheet_name=actant, index=False)

files.download(intermediate_file)

# =======================
# 10) MERGE CLUSTERS WITH GPT
# =======================
xls = pd.ExcelFile(intermediate_file)
sheet_names = xls.sheet_names

actant_clusters_data = {}
for sheet_name in sheet_names:
    sheet_data = xls.parse(sheet_name)
    # If this is the dummy sheet, skip
    if "NoData" in sheet_name:
        continue

    # Build a unique list of cluster names
    col_name = f"{sheet_name}_Narrative_Name"
    if col_name not in sheet_data.columns:
        # If the sheet lacks the needed columns, skip
        continue

    clusters_data = sheet_data[col_name].unique().tolist()
    formatted_data = f"Actant Role: {sheet_name}\nCluster Names: {clusters_data}\n"
    actant_clusters_data[sheet_name] = formatted_data

merge_results = {}
for role, data in actant_clusters_data.items():
    prompt = f"""
You are a helpful assistant tasked with analyzing clusters of actants to determine which clusters are semantically similar and should be merged.

### Purpose:

The goal is to group clusters that represent the exact same concept or have unambiguous semantic overlap. Avoid merging clusters that represent distinct but related ideas (e.g., "National Resistance" and "Commemoration of Resistance"). Always prioritize preserving distinctions unless there is clear evidence that two clusters are intended to refer to the same concept.

Instructions:

1. Below, you will find a list of identified clusters.

2. Your job is to:
   - Analyze the clusters and identify groups of clusters that represent the exact same concept.
   - For each group of similar clusters, suggest a single merged cluster name that represents the shared theme of the group in 1-3 words max.

3. To ensure precision:
   - Avoid assuming overlap between clusters unless their meanings clearly indicate they are interchangeable or similar.
   - For example, clusters dealing with different levels of specificity or different functions should not be merged.

4. It is acceptable to leave clusters unchanged if no clear semantic equivalence exists. Do not force merges if no clear overlap exists.

5. Do not rename or modify the names of clusters that are not part of a merged group.

6. Format your output as follows, in valid JSON:
   {{
     "Merge_Groups": [
       {{
         "Clusters_To_Merge": ["Cluster 1", "Cluster 3"],
         "Merged_Cluster_Name": "Shared Theme of Cluster 1 and 3"
       }},
       {{
         "Clusters_To_Merge": ["Cluster 2", "Cluster 4"],
         "Merged_Cluster_Name": "Shared Theme of Cluster 2 and 4"
       }}
     ]
   }}

### Data:

{data}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for analyzing and merging clusters."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.7
    )

    try:
        raw_content = response['choices'][0]['message']['content']
        cleaned_content = raw_content.strip('```json').strip('```').strip()
        merge_results[role] = json.loads(cleaned_content)

        print(f"\n=== Merge Results for {role} ===")
        print(json.dumps(merge_results[role], indent=2))

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for role {role}: {e}")
        merge_results[role] = {"Merge_Groups": []}

# =======================
# 11) APPLY MERGES & SAVE
# =======================
merged_dfs = []
for sheet_name in sheet_names:
    sheet_data = xls.parse(sheet_name)
    if "NoData" in sheet_name:
        merged_dfs.append(sheet_data)
        continue

    if f"{sheet_name}_Narrative_Name" not in sheet_data.columns:
        merged_dfs.append(sheet_data)
        continue

    merge_map = {}
    for group in merge_results.get(sheet_name, {}).get("Merge_Groups", []):
        for cluster in group["Clusters_To_Merge"]:
            merge_map[cluster] = group["Merged_Cluster_Name"]

    final_col = f"{sheet_name}_Final_Cluster"
    name_col = f"{sheet_name}_Narrative_Name"

    if name_col in sheet_data.columns:
        sheet_data[final_col] = sheet_data[name_col].map(merge_map).fillna(sheet_data[name_col])
    merged_dfs.append(sheet_data)

output_file = "merged_actant_clusters_hdbscan.xlsx"
with pd.ExcelWriter(output_file) as writer:
    # Similarly, ensure at least one sheet is written
    if len(merged_dfs) == 0:
        dummy_df = pd.DataFrame({"NoData": ["NoActantsMerged"]})
        dummy_df.to_excel(writer, sheet_name="NoData", index=False)
    else:
        all_empty = True
        for idx, sheet_name in enumerate(sheet_names):
            if "NoData" in sheet_name:
                merged_dfs[idx].to_excel(writer, sheet_name=sheet_name, index=False)
                all_empty = False
            else:
                if not merged_dfs[idx].empty:
                    merged_dfs[idx].to_excel(writer, sheet_name=sheet_name, index=False)
                    all_empty = False
        if all_empty:
            dummy_df = pd.DataFrame({"NoData": ["NoActantsMerged"]})
            dummy_df.to_excel(writer, sheet_name="NoData", index=False)

files.download(output_file)

# ==================================================================================================
#       MERGED SCRIPT CONTINUES: We now feed the final .xlsx output into the second code
# ==================================================================================================

# =================================================================================
# **** THIS IS THE COMMAND THAT PASSES THE 1ST SCRIPT'S OUTPUT TO THE 2ND SCRIPT ***
# =================================================================================
second_script_input_file = output_file
print(f"\n[INFO] The first script has created '{second_script_input_file}'. Passing it to the second script...\n")

# =======================
# ******* SECOND CODE *******
# (Slightly modified to skip manual upload & unify variable naming)
# =======================

# Install necessary packages
!pip install sentence-transformers openai umap-learn matplotlib openpyxl scikit-learn hdbscan

"""
This script processes outliers (noise labeled as `-1` in the HDBSCAN clustering output).
It uses GPT to assign outliers collectively to the most appropriate existing clusters or create new clusters if possible.
The script is structured for Google Colab, allowing file upload for input and automatic download of the processed output.
"""

import pandas as pd
import json
import openai
from google.colab import files

openai.api_key = "sk-proj-_"  # Replace with your actual OpenAI API key

print("[INFO] Skipping manual upload. We'll use the file automatically from the first script.")
file_name = second_script_input_file
print(f"[INFO] Processing file: {file_name}")

xls = pd.ExcelFile(file_name)
sheet_names = xls.sheet_names
processed_sheets = {}

def clean_gpt_response(raw_response):
    try:
        cleaned_response = raw_response.strip('```').strip('json').strip()
        json.loads(cleaned_response)  # Validate JSON
        return cleaned_response
    except json.JSONDecodeError as e:
        print(f"Error validating GPT response: {e}")
        print(f"Raw response was: {raw_response}")
        return None

for sheet_name in sheet_names:
    print(f"\nProcessing sheet: {sheet_name}...")
    data = xls.parse(sheet_name)

    # If we have a dummy or NoData sheet, skip
    if "NoData" in sheet_name:
        print(f"Skipping dummy sheet: {sheet_name}")
        processed_sheets[sheet_name] = data.copy()
        continue

    # We need to check if it has the columns we expect
    cluster_col = f"{sheet_name}_Cluster"
    final_col = f"{sheet_name}_Final_Cluster"
    if cluster_col not in data.columns:
        print(f"Column '{cluster_col}' not found in {sheet_name}. Skipping outlier processing.")
        processed_sheets[sheet_name] = data.copy()
        continue
    if final_col not in data.columns:
        # Possibly hasn't merged, so let's create it by copying the narrative name if it exists
        name_col = f"{sheet_name}_Narrative_Name"
        if name_col in data.columns:
            data[final_col] = data[name_col]
        else:
            data[final_col] = ""

    # Identify outliers
    outliers = data[data[cluster_col] == -1]

    # Summarize existing clusters
    existing_clusters = data[data[cluster_col] != -1][final_col].unique()
    cluster_summaries = {}
    for cluster in existing_clusters:
        # Ensure all items are strings to avoid TypeError
        cluster_texts = data[data[final_col] == cluster][sheet_name].fillna('').astype(str).tolist()
        cluster_summaries[cluster] = "\n".join(cluster_texts[:10])  # Up to 10 samples per cluster

    # Collect all outlier texts
    outlier_texts = outliers[sheet_name].tolist()

    if len(outlier_texts) > 0:
        prompt = f"""
        You are a helpful assistant tasked with analyzing a group of outlier actants and determining their best placement among existing clusters or creating new clusters if needed.

        ### Existing Clusters:
        {json.dumps(cluster_summaries, indent=2)}

        ### Outlier Texts:
        {json.dumps(outlier_texts, indent=2)}

        ### Task:
        1. Analyze the outlier texts collectively. For context, all of the outlier concepts are tied to the Slovene national holiday Day of resistance to occupation.
        2. Assign each outlier to one of the existing clusters by returning the cluster name.
        3. If a group of outliers forms a coherent theme but doesn't match existing clusters, create a new cluster name for them. There must be at least 10 outliers to create a new cluster name. If there is less than ten of them do not create a new cluster under no circumstance.
        4. Try fitting each outlier to an already existing cluster. In the end every single outliers should be assigned to a cluster.

        Output your response as valid JSON with the following structure:
        {{
            "Assignments": {{
                "Outlier Text 1": "Cluster Name or Outlier",
                "Outlier Text 2": "Cluster Name or Outlier"
            }},
            "New Clusters": [
                {{
                    "Cluster Name": "New Cluster Name",
                    "Members": ["Outlier Text 1", "Outlier Text 3"]
                }}
            ]
        }}
        """

        retries = 3
        while retries > 0:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for assigning outliers to clusters."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=12000,
                    temperature=0.5
                )

                raw_response = response['choices'][0]['message']['content'].strip()
                print(f"Raw GPT response for {sheet_name}:\n{raw_response}\n")

                cleaned_response = clean_gpt_response(raw_response)
                if cleaned_response is None:
                    raise ValueError("Invalid GPT response.")

                gpt_response = json.loads(cleaned_response)
                assignments = gpt_response.get("Assignments", {})
                new_clusters = gpt_response.get("New Clusters", [])

                # Update the dataframe with reassigned outliers
                for outlier_text, assigned_cluster in assignments.items():
                    idx = data[(data[sheet_name] == outlier_text) & (data[cluster_col] == -1)].index
                    if not idx.empty:
                        data.loc[idx, final_col] = assigned_cluster
                        print(f"Updated: Outlier '{outlier_text}' assigned to '{assigned_cluster}'.")

                # Add new clusters to the dataframe
                for new_cluster in new_clusters:
                    cluster_name = new_cluster["Cluster Name"]
                    for member in new_cluster["Members"]:
                        idx = data[(data[sheet_name] == member) & (data[cluster_col] == -1)].index
                        if not idx.empty:
                            data.loc[idx, final_col] = cluster_name
                            print(f"Updated: New cluster '{cluster_name}' created for member '{member}'.")

                break  # Success, break out of retry loop

            except Exception as e:
                retries -= 1
                print(f"Error processing outliers in sheet {sheet_name}, retries left: {retries}. Error: {e}")
                if retries == 0:
                    print(f"Failed to process outliers for sheet {sheet_name}. Skipping...")

    else:
        print(f"No outliers found in sheet {sheet_name}.")

    processed_sheets[sheet_name] = data.copy()

# Save the updated data to a new Excel file
output_file_name = "all_actant_clusters_outliers_processed_batch.xlsx"
with pd.ExcelWriter(output_file_name) as writer:
    if not processed_sheets:
        dummy_df = pd.DataFrame({"NoData": ["NoActantsProcessed"]})
        dummy_df.to_excel(writer, sheet_name="NoData", index=False)
    else:
        all_empty = True
        for sheet_name, processed_data in processed_sheets.items():
            if not processed_data.empty:
                processed_data.to_excel(writer, sheet_name=sheet_name, index=False)
                all_empty = False
        if all_empty:
            dummy_df = pd.DataFrame({"NoData": ["NoActantsProcessed"]})
            dummy_df.to_excel(writer, sheet_name="NoData", index=False)

files.download(output_file_name)
print("Processing complete. File is ready for download.")
