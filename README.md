# Deploying a PyTorch model with Flask

![Build and Deploy to Cloud Run](https://github.com/oliverproud/PyTorch-Sentiment-Analysis-deployed-with-Flask/workflows/Build%20and%20Deploy%20to%20Cloud%20Run/badge.svg?branch=master)
[![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/oliverproud/pytorch-sent-analysis-flask?logo=Docker&logoColor=FFF)](https://hub.docker.com/repository/docker/oliverproud/pytorch-sent-analysis-flask)
[![Website](https://img.shields.io/website?down_message=offline&up_message=online&url=https%3A%2F%2Fsentiment.oliverproud.com)](https://sentiment.oliverproud.com)

Try the demo at [sentiment.oliverproud.com](https://sentiment.oliverproud.com)



3 layer Convolutional model trained on IMDb dataset, using GloVe pretrained word embedddings.

As this was my first "full stack" project I wanted to keep it relatively simple and quick by using a Convolutional model and deploying it on Heroku using Flask (I have now re-deployed the model on Google Cloud Run). ~~I will later explore using a BERT or most likely XLNET architecture from the Hugging Face team~~ ✔ [See here](https://github.com/oliverproud/bert-sequence-classification).

The model weights and word dictionary are available to download from S3:
- [model weights](https://sent-model.s3.eu-west-2.amazonaws.com/conv-sentiment_model1.pt) 
- [word dict](https://sent-model.s3.eu-west-2.amazonaws.com/word_dict.pkl)

# Installation

If you are testing this on your own machine I would recommend you run it in a virtual environment or use Docker, as not to affect the rest of your files.

### Docker

To use my docker repository:

```bash
docker pull oliverproud/pytorch-sent-analysis-flask
```

Run the container:

```bash
docker run -dp 8080:8080 oliverproud/pytorch-sent-analysis-flask
```

Or Git Pull this repository and

Build the container:

```bash
docker build -t pytorch-sent-analysis-flask .
```

Run the container:

```bash
docker run -dp 8080:8080 pytorch-sent-analysis-flask
```

### Python venv

In Python3 you can set up a virtual environment with

```bash
python3 -m venv /path/to/new/virtual/environment
```

Or by installing virtualenv with pip by doing
```bash
pip3 install virtualenv
```
Then creating the environment with
```bash
virtualenv venv
```
and finally activating it with
```bash
source venv/bin/activate
```

You **must** have Python3

Install the requirements with:
```bash
pip3 install -r requirements.txt
```

### Contact

If you have any questions, feedback or problems of any kind, get in touch by messaging me on [Twitter - @oliverwproud](https://twitter.com/oliverwproud) or by submitting an issue.

If you want to check out my deployed version you can head to [sentiment.oliverproud.com](https://sentiment.oliverproud.com).

### References

Resources I used and found helpful: 

- <https://github.com/bentrevett/pytorch-sentiment-analysis>
- <https://github.com/choonghee-lee/Deploying-a-Sentiment-Analysis-Model>
- <http://josh-tobin.com/troubleshooting-deep-neural-networks.html>
- <https://spacy.io/models>
