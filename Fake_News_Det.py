from flask import Flask, render_template, request
from text_summerization import summerizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))
dataframe = pd.read_csv('news.csv')
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_det(news):
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        text = request.form['text']
        original_text, summary, orig_word_count, summary_word_count = summerizer(text)
        return render_template('index.html',
                               original_text=original_text,
                               summary=summary,
                               orig_word_count=orig_word_count,
                               summary_word_count=summary_word_count)

if __name__ == '__main__':
    app.run(debug=True)
