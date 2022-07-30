import pickle
from flask import Flask, request, jsonify
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

app = Flask(__name__)
vect = pickle.load(open('vect.pkl', 'rb'))
model = pickle.load(open('nlpmodel3.pkl', 'rb'))
snowball = SnowballStemmer(language="english")


@app.route("/predict/", methods=['POST'])
def predict():
    news = request.form.get('news')
    news = word_tokenize(news)
    stem_words = []
    for word in news:
        wrd = snowball.stem(word)
        stem_words.append(wrd)

    news = []
    for words in stem_words:
        if len(words) > 2:
            news.append(words)

    news = (' '.join(news))
    news = vect.fit_transform([news]).toarray()
    res = model.predict(news)
    return jsonify({'truth': int(res)})


if __name__ == '__main__':
    app.run(debug=True)
