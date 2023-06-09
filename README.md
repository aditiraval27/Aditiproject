# Aditiproject
## coding question Classification 2
#### main.py 4
#Import external dependencies.
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import spacy
app = Flask(__name__, template_folder='./')
import nltk
from sklearn.metrics import fbeta_score, accuracy_score
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

#Import classifier.
classifier = pickle.load(open('classifier.pickle', 'rb'))


#Normalizes questions.
def normalizeQuestions(text):
  tokens = word_tokenize(text.lower())
  filtered_tokens = [
    token for token in tokens if token not in stopwords.words('english')
  ]
  lemmatizer = WordNetLemmatizer()
  lemmatized_tokens = [
    lemmatizer.lemmatize(token) for token in filtered_tokens
  ]
  return ' '.join(lemmatized_tokens)


#Listen on /classify.
@app.route('/classify', methods=['POST'])
@cross_origin()
def classify():
  #Get the question from the request.
  question = request.json['question']

  #Normalize the question.
  print("Normalizing question...")
  normalized = normalizeQuestions(question)

  #Convert to dict for classification.
  q = {}
  for w in normalized.split(' '):
    q[w] = True

  #Classify the question.
  answer = classifier.classify(q)

  #Return the classification.
  return jsonify({'answer': answer})


#Serve the frontend.
@app.route('/')
def home():
  print("Home")
  return render_template('./frontend.html')


#Start the Flask API.
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)
  
  
###### training.py 6

#Trains the model for usage by the API.


#Import external dependencies.
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import fbeta_score, accuracy_score
import pickle


#Read the training data CSV.
print("Reading training csv...")
df = pd.read_csv('training.csv')
df.info()

#Normalize questions.
print("Normalizing questions...")
def normalizeQuestions(text):
  tokens = word_tokenize(text.lower())
  filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
  lemmatizer = WordNetLemmatizer()
  lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
  return ' '.join(lemmatized_tokens)
df['QUESTONS'] = df['QUESTONS'].apply(normalizeQuestions)

#Convert dataframe to dictionary.
data = df.to_dict(orient='records')

#Now, convert questions into tuples for training.
print("Converting questions into tuples...")
for(i, row) in enumerate(data):
  q = {}
  for w in row['QUESTONS'].split(' '):
    q[w] = True
  data[i] = (q, row['CATEGORIES'])

#Train the model.
print("Training model...")
classifier = nltk.NaiveBayesClassifier.train(data)
print(accuracy_score(classifier))

#Save the model.
print("Saving model...")
save_classifier = open('classifier.pickle', 'wb')
pickle.dump(classifier, save_classifier)
save_classifier.close()
print("Done.")
