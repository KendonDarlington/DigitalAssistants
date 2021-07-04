#Import some libraries
import pandas as pd
import nltk
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

#You may need to run this bit of code to get the vader lexicon if it is your first time using it.
#nltk.download('vader_lexicon')

#Bring in data. Set the working directory to the folder you created for this project.
os.chdir('C:\\PathToYourFolder')
dfResponses = pd.read_csv('DigitalAssistantResponses.csv')

#Create a dataframe to store our results
dfCompoundScores = pd.DataFrame(columns = ['Assistant', 'Sentence', 'Value'])

#Loop through our responses, break them into sentences and get the compound scores. Slam this into dfCompoundScores
sid = SentimentIntensityAnalyzer()
for index, row in dfResponses.iterrows():
    sid = SentimentIntensityAnalyzer()
    sentences = [] 
    assistant = row[0]
    responseText = dfResponses.loc[dfResponses['Assistant'] == row[0]]['Response'][index]
    sentences = []    
    tokenizer = tokenize.sent_tokenize(responseText) 
    sentences.extend(tokenizer)
    for sentence in sentences:
        ss = sid.polarity_scores(sentence)
        compoundScore = ss['compound']
        dfCompoundScores = dfCompoundScores.append({'Assistant' : assistant, 'Sentence' : sentence, 'Value': compoundScore}, ignore_index=True)
        
#Group by assistant and average the compound sentiment scores 
dfAvgSentiment = dfCompoundScores[['Assistant','Value']]
dfAvgSentiment  = dfAvgSentiment.groupby(['Assistant'])
dfAvgSentiment = dfAvgSentiment.mean()
dfAvgSentiment = dfAvgSentiment.reset_index()

#Write this to a csv so it can be picked up by D3
dfAvgSentiment.to_csv('ResponseAggregatedSentiment.csv', index = False)

#Run these from the console
#cd C:\PathToYourFolder
#python -m http.server
