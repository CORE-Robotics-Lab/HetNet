import csv
import json
import os
# Function to convert a CSV to JSON
# Takes the file paths as arguments
def make_json(csvFilePath, jsonFilePath):
     
    # create a dictionary
    data = {}
    steps = []
    # win_rate = []
    ep_length_hetgat = []
    ep_length_mappo = []
    # Open a csv reader called DictReader
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
        
        # Convert each row into a dictionary
        # and add it to data
        for rows in csvReader:
             
            # Assuming a column named 'No' to
            # be the primary key
            # key = rows['Step']
            # data['Step'] = rows['Step']
            # data['Step'].append(rows['Step'])
            steps.append(float(rows["Step"]))
            ep_length_hetgat.append(float(rows["algorithm_name: hetgat_mappo - eval_average_episode_lengths"]))
            ep_length_mappo.append(float(rows["algorithm_name: rmappo - eval_average_episode_lengths"]))
    data['step'] = steps
    data['ep_length_hetgat'] = ep_length_hetgat
    data['ep_length_mappo'] = ep_length_mappo
    # Open a json writer, and use the json.dumps()
    # function to dump data
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))
        
if __name__ == "__main__":
    # Decide the two file paths according to your
    # computer system
    csvFilePath = os.path.join(os.getcwd(), 'onpolicy', 'results', 'ep_length_new.csv')
    jsonFilePath = os.path.join(os.getcwd(), 'onpolicy', 'results', 'ep_lengths_new.json')
    
    # Call the make_json function
    make_json(csvFilePath, jsonFilePath)