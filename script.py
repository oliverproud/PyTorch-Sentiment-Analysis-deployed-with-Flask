#Link between model.py and index.html to take user inputs and display results

import flask
from flask import request
from model import predict_sentiment

# Initialize the app
app = flask.Flask(__name__)

@app.route('/')
def index():

    #displays the shown string above the user entered text
    header_review = "Review:"
    #displays the show string above the model determined sentiment
    header_sentiment = "Sentiment:"

    print(request.args)
    #contains a dictionary containing the parsed contents of the query string
    if(request.args):
        #passes contents of query string to the prediction function contained in model.py
        x_input, prediction = predict_sentiment(request.args['text_in'])
        print(prediction[0]['prob'])
        #indexes the returned dictionary for the sentiment probability
        if((prediction[0]['prob']) > 0.5):
            prediction = "Positive"
            return flask.render_template('index.html', text_in=x_input, prediction=prediction, header_review=header_review, header_sentiment=header_sentiment)
        else:
            prediction = "Negative"
            return flask.render_template('index.html', text_in=x_input, prediction=prediction, header_review=header_review, header_sentiment=header_sentiment)
    #if the parsed query string does not contain anything then return index page
    else:
        return flask.render_template('index.html')
