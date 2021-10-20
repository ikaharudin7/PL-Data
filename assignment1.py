# This is the file you will need to edit in order to complete assignment 1
# You may create additional functions, but all code must be contained within this file


# Some starting imports are provided, these will be accessible by all functions.
# You may need to import additional items
import numpy as np
import pandas as pd
import json
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('punkt')
import math
from numpy import dot
from numpy.linalg import norm
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from itertools import combinations
from matplotlib import pyplot as plt

# You should use these two variable to refer the location of the JSON data file and the folder containing the news articles.
# Under no circumstances should you hardcode a path to the folder on your computer (e.g. C:\Chris\Assignment\data\data.json) as this path will not exist on any machine but yours.
datafilepath = 'data/data.json'
articlespath = 'data/football'

def task1():
    #Complete task 1 here
    with open(datafilepath, 'r') as data:
        json_obj = data.readlines()
    json_obj = [(obj.rstrip()).lstrip()[:-1] for obj in json_obj[1:-1]]
    
    # Iterate through the list to find 'team_codes'
    for item in json_obj:
        if "teams_code" in item:
            teams_code = item
            break
    
    list_start = teams_code.index("[")
    list_end = teams_code.index("]")
    teams_code = teams_code[list_start+1:list_end]
    teams_code = teams_code.replace("\"", "").split(", ")
    teams_code.sort()
    data.close()
    return teams_code


    
def task2():
    #Complete task 2 here
    df = pd.read_json(datafilepath, orient='index')
    # Retrieve list of dictionaries (not in dataframe format)
    data_list = df.loc["clubs"][0]
    
    # Sort alphabetically
    data_list = sorted(data_list, key = lambda i: i['club_code'])
    
    # Lists for stored data
    team_codes = []
    goals_scored = []
    goals_conceded = []
    
    # Append the values to the lists
    for dictionary in data_list:
        team_codes.append(dictionary['club_code'])
        goals_scored.append(dictionary['goals_scored'])
        goals_conceded.append(dictionary['goals_conceded'])
    
    # Create into DataFrame and make into csv
    data_format = {'team_code': team_codes, 
                   'goals_scored_by_team': goals_scored, 
                   'goals_scored_against_team': goals_conceded}
    
    df = pd.DataFrame(data_format, columns= ['team_code', 'goals_scored_by_team', 'goals_scored_against_team'])
    
    df.to_csv('task2.csv', index = False)
            
    return 
      
def task3():
    #Complete task 3 here
    # Get list of files and initialize lists
    all_files = os.listdir(articlespath)
    all_files.sort()
    
    # Get rid of unwanted file
    if '.ipynb_checkpoints' in all_files:
        all_files.remove('.ipynb_checkpoints')
        
    total_goals = []
    goals = []
       
    # Iterate through files and use regex to find scores
    for file in all_files:
        highest = 0
        with open(os.path.join(articlespath, file), 'r') as f:
            text = f.read()
            scores = re.findall("[0-9]+-[0-9]+", text)
            
            # Check if the scores are valid football scores and then take the highest from each text
            for score in scores: 
                goals = score.split("-")
                if (0<=int(goals[0]) and int(goals[0])<100) and (0<=int(goals[1]) and int(goals[1])<100):
                    if (int(goals[0])+int(goals[1])>highest):
                        highest = int(goals[0])+int(goals[1])
                        
        total_goals.append(highest)
    
    
    # Create into DataFrame and make into csv
    data_format = {'filename': all_files, 
                   'total_goals': total_goals}
    
    df = pd.DataFrame(data_format, columns= ['filename', 'total_goals'])
    
    df.to_csv('task3.csv', index = False)
    
    return

def task4():
    #Complete task 4 here
    # Obtain dataframe from task3
    task3()
    df = pd.read_csv('task3.csv')
    axes = df.boxplot()
    
    # Label
    axes.set_title("Box Plot of Total Goals in Articles")
    axes.set_xlabel('Box Plot')
    axes.set_ylabel('Number of Goals in a Match')
    plt.xticks([1], [''])
    task = plt.savefig('task4.png')
        
    return task
    
def task5():

    df = pd.read_json(datafilepath, orient='index')
    # Retrieve list of participating clubs
    data_list = df.loc["participating_clubs"][0]
    sorted(data_list)
    
    # Get list of files and initialize lists
    all_files = os.listdir(articlespath)
    all_files.sort()
    
    # Get rid of unwanted file
    if '.ipynb_checkpoints' in all_files:
        all_files.remove('.ipynb_checkpoints')
    
    total_mentions = []
    
    # Iterate through files and use regex to find mentions
    for team in data_list:  
        mentions = 0
        for file in all_files:
            with open(os.path.join(articlespath, file), 'r') as f:
                text = f.read()
                if len(re.findall(team, text))>0:
                    mentions+=1
        # Append to the list
        total_mentions.append(mentions)
        
    # Create into DataFrame and make into csv
    data_format = {'club_name': data_list, 
                   'number_of_mentions': total_mentions}
    df = pd.DataFrame(data_format, columns= ['club_name', 'number_of_mentions'])
    df.to_csv('task5.csv', index = False)
    
    # Create bar chart from data_list and total_mentions
    bar_data = df
    graph = bar_data.plot.bar(x="club_name", y="number_of_mentions")
    graph.set_xlabel('Club Name')
    graph.set_ylabel('Number of Mentions')
    graph.set_title("Number of Club's Mentions in Articles")
    task = plt.savefig('task5.png', bbox_inches='tight')
    plt.close('all')
    
    return
    
def task6():
    #Complete task 6 here
    # Obtain the csv and articlespath
    df = pd.read_csv('task5.csv')
    all_files = os.listdir(articlespath)
    all_files.sort()
    # Get rid of unwanted file
    if '.ipynb_checkpoints' in all_files:
        all_files.remove('.ipynb_checkpoints')
    
    team_names = list(df['club_name'])
    similarities = []
    # Iterate through the permutations and create arrays for dataframe 
    for team1 in team_names:
        sim_list = []
        for team2 in team_names:
            mentions = 0
            # Get the total mentions of each club
            i = df[df['club_name']==team1]
            team1_mentions = int(i['number_of_mentions'])
            j = df[df['club_name']==team2]
            team2_mentions = int(j['number_of_mentions'])
            
            # Get the number of articles mentioning both club1 and 2
            for file in all_files:
                with open(os.path.join(articlespath, file), 'r') as f:
                    text = f.read()
                    if len(re.findall(team1, text))>0 and len(re.findall(team2, text))>0:
                        mentions+=1
            if team1_mentions + team2_mentions != 0:
                sim = 2*(mentions)/(team1_mentions + team2_mentions)
            elif team1==team2:
                sim = 1
            else: 
                sim = 0
            sim_list.append(sim)
        # Add to similarities array
        similarities.append(sim_list)
    # Render the image
    dataF = pd.DataFrame(similarities, index=team_names, columns=team_names)
    heat = sns.heatmap(dataF)
    heat.set_xlabel('Club Name')
    heat.set_ylabel('Club Name')
    heat.set_title("Heat Map of Club Similarity")
    hmap = heat.get_figure()
    hmap.savefig('task6.png', bbox_inches='tight')   
    plt.close('all')
    return
    
def task7():
    #Complete task 7 here
    mentions_data = pd.read_csv('task5.csv')
    goals_data = pd.read_csv('task2.csv')
    
    # Get data in array form
    team_names = list(mentions_data['club_name'])
    num_goals = list(goals_data['goals_scored_by_team'])
    num_mentions = list(mentions_data['number_of_mentions'])

    # Make dataframe
    task7_df = {'number_of_goals': num_goals, 
                'number_of_mentions':num_mentions}
    task7_dataframe = pd.DataFrame(task7_df, columns= ['number_of_goals', 'number_of_mentions'])
    
    task7_dataframe.plot.scatter(x='number_of_goals', y='number_of_mentions',
                                title = "Number of Mentions vs Number of Goals for Each Club")
    plt.savefig('task7.png', bbox_inches='tight')
    plt.close('all')
    
    return
    
def task8(filename):
    #Complete task 8 here
    
    final = []
    # Get list of files and initialize lists
    with open(filename, 'r') as f:
        text = f.read()
        text = re.sub('[-\']', ' ', text)
        text = nltk.word_tokenize(text)
         
        # Change words to lowercase
        for word in text:
            word = word.lower()
            if word in stopwords.words('english'):
                continue
            word = re.sub('[^a-zA-Z]', '', word)
            if len(word)>1:
                final.append(word)
        
    
    return final
    
def task9():
    #Complete task 9 here
    # Get list of files and initialize lists
    all_files = os.listdir(articlespath)
    all_files.sort()
    
    # Get rid of unwanted file
    if '.ipynb_checkpoints' in all_files:
        all_files.remove('.ipynb_checkpoints')
    
    # Process every text, and get the set of words
    all_words = []
    total_words = []
    for file in all_files:
        tokens = sorted(task8(os.path.join(articlespath, file)))
        all_words.append(tokens)
        total_words = total_words + tokens
    
    total_set = set(total_words)
    all_vectors = []
    count_vector = []
    
    # Get tbe count vectors
    for file in all_files:
        # Get the total tokens from each 
        index = all_files.index(file)
        tokens = all_words[index]

        # Count the words in 1 text in the total_tokens
        vector = {}
        for word in total_set:
            vector[word] = tokens.count(word)
        
        vector = list(vector.values())
        count_vector.append(vector)
    
    # Make the count vectors into tf-idf vectors
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(count_vector)
    doc_tfidf = tfidf.toarray()
                           
    # Get cosine similarity
    cosine_sim = []
    article1 = []
    article2 = []
    combos = list(combinations(all_files, 2))
    for pair in combos:
        index1 = all_files.index(pair[0])
        index2 = all_files.index(pair[1])
        
        v1 = doc_tfidf[index1]
        v2 = doc_tfidf[index2]
        cos_sim = dot(v1, v2)/(norm(v1)*norm(v2))
        
        article1.append(pair[0])
        article2.append(pair[1])
        cosine_sim.append(cos_sim)
        
    task9_df = {'article1': article1, 
                'article2': article2,
                'similarity': cosine_sim}
    task9_dataframe = pd.DataFrame(task9_df, columns= ['article1', 'article2', 'similarity'])
    task9_dataframe.sort_values('similarity')
    answer = task9_dataframe.sort_values('similarity', ascending = False).head(10)
    answer.to_csv('task9.csv', index = False)
    
            
    return