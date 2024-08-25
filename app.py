# app.py
from flask import Flask, request, render_template
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SummarizationOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import os
from config import API_KEY, SERVICE_URL

app = Flask(__name__)

# Updated generate_summary function
def generate_summary(text):
    authenticator = IAMAuthenticator(API_KEY)
    nlu_service = NaturalLanguageUnderstandingV1(
        version='2023-08-25',
        authenticator=authenticator
    )
    nlu_service.set_service_url(SERVICE_URL)
    
    response = nlu_service.analyze(
        text=text,
        features=Features(summarization=SummarizationOptions(limit=50))
    ).get_result()
    
    summary = response['summarization'][0]['text']
    return summary

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/summary', methods=['POST'])
def summarize():
    if request.method == 'POST':
        file = request.files['file']
        text = file.read().decode('utf-8')
        summary = generate_summary(text)
        return render_template('summary.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
