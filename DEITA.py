import pickle
import json
import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm

def cosine_distance(vec1, vec2):
    """Calculate the cosine distance between two vectors."""
    return cosine(vec1, vec2)

def select_data(alpaca_data, alpaca_embeddings, max_size, diversity_threshold):
    """
    Select data based on quality and diversity.
    
    :param alpaca_data: List of dictionaries with 'score' key for quality score
    :param alpaca_embeddings: List of embeddings corresponding to the alpaca_data
    :param max_size: Maximum size of the selected dataset
    :param diversity_threshold: Cosine distance threshold for diversity
    :return: Selected dataset
    """
    selected = []
    remaining_pool = []

    # Step 1: Initial Selection based on Quality Score
    first_score_5_found = False
    for data, embedding in zip(alpaca_data, alpaca_embeddings):
        if not first_score_5_found and data['score'] == 5:
            selected.append((data, embedding))
            first_score_5_found = True
        else:
            remaining_pool.append((data, embedding))

    # Step 2: Diversity-based Selection
    while len(selected) < max_size and remaining_pool:
        if len(remaining_pool) % 1000 == 0:
            print("remaining pool:", len(remaining_pool), "selected", len(selected), "diversity_threshold", diversity_threshold)
        
        current_data, current_embedding = remaining_pool.pop(0)

        # Find nearest neighbor in the selected set
        nearest_distance = float('inf')
        for _, selected_embedding in selected:
            distance = cosine_distance(current_embedding, selected_embedding)
            nearest_distance = min(nearest_distance, distance)

        #print(nearest_distance)
        # If the nearest distance is above the threshold, add to selected
        if nearest_distance > diversity_threshold:
            selected.append((current_data, current_embedding))

        if len(selected) == max_size:
            break
            
    json_friendly_selected = [{'data': data, 'embedding': embedding.tolist()} for data, embedding in selected]
    return json_friendly_selected

if __name__ == '__main__':

    for diversity_threshold in [0.15]:
    
        alpaca_embedding_path = 'wpq_alpaca.pkl'
        # Open the file in binary read mode
        with open(alpaca_embedding_path, 'rb') as file:
            # Load the object from the file
            data = pickle.load(file)
        alpaca_embedding = data['text_embedding']
    
        # Paths to the files
        merged_alpaca_data_path = 'merged_alpaca_data.json'
        # Read the data from both files
        with open(merged_alpaca_data_path, 'r') as file:
            merged_alpaca_data = json.load(file)
    
        # Sort the merged data based on score
        # Assuming that each entry in the data has a 'score' field
        sorted_alpaca_data = sorted(merged_alpaca_data, key=lambda x: x['score'], reverse=True)
        
        # Get the sorted indices based on the score
        sorted_indices = [i[0] for i in sorted(enumerate(merged_alpaca_data), key=lambda x: x[1]['score'])]
        
        # Apply the same sorting order to the other array
        sorted_alpaca_embedding = [alpaca_embedding[i] for i in sorted_indices]
    
        selected_alpaca_data = select_data(sorted_alpaca_data, sorted_alpaca_embedding, max_size=6000, diversity_threshold=diversity_threshold)
        
        # Convert the selected data to JSON and save it
        json_data = json.dumps(selected_alpaca_data, indent=4)
        with open(f'selected_alpaca_data_threshold_{diversity_threshold}.json', 'w') as file:
            file.write(json_data)