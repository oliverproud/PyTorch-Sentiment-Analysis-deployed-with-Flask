# Deploying a PyTorch model with Flask

3 layer Convolutional model trained on IMDb dataset (will use a different dataset soon), using GloVe pretrained word embedddings.

As this was my first "full stack" project I wanted to keep it relatively simple and quick by using a Convolutional model and deploying it on Heroku using Flask (I have now re-deployed the model on Google Cloud Run). I will later explore using a BERT or most likely XLNET architecture from the Hugging Face team.

The model weights and word dictionary are all available to download from S3:
- [model weights](https://sent-model.s3.eu-west-2.amazonaws.com/conv-sentiment_model1.pt) 
- [word dict](https://sent-model.s3.eu-west-2.amazonaws.com/word_dict.pkl)

If you want to check out my deployed version you can head to [sentiment.oliverproud.com](https://sentiment.oliverproud.com).

If you have any questions, feedback or problems of any kind, get in touch by messaging me on [Twitter - @oliverwproud](https://twitter.com/oliverwproud) or submitting an issue.

### References

Resources I used and found helpful: 

- <https://github.com/bentrevett/pytorch-sentiment-analysis>
- <https://github.com/choonghee-lee/Deploying-a-Sentiment-Analysis-Model>
- <http://josh-tobin.com/troubleshooting-deep-neural-networks.html>
- <https://spacy.io/models>
