import json

# Path to your JSON file
file_path = 'selected_alpaca_data_threshold_0.1.json'

# Function to remove 'embedding' attribute and elevate 'data' to top level
def transform_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Create a new list with only the 'data' value of each item
    modified_data = [item['data'] for item in data if 'data' in item]

    # Write the modified data back to the file
    with open(file_path, 'w') as file:
        json.dump(modified_data, file, indent=4)

# Function to count items in the JSON file
def count_items_in_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

        # If the data is a list, return its length
        if isinstance(data, list):
            return len(data)
        # If the data is a dictionary, return the count of values
        elif isinstance(data, dict):
            return sum(len(v) if isinstance(v, list) else 1 for v in data.values())
        # If the data is neither a list nor a dictionary, return 0 or an appropriate message
        else:
            return "The JSON structure is not a list or dictionary, and cannot be counted in this manner."

# Call the function
transform_json(file_path)

# Counting items in the JSON file
item_count = count_items_in_json(file_path)
print(f"Number of items in the JSON file: {item_count}")