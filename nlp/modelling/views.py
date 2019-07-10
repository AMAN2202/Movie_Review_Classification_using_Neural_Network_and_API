from django.shortcuts import render
import os

from django.http import JsonResponse
from sklearn.externals import joblib

CURRENT_DIR = os.path.dirname(__file__)
_nlp=os.path.join(CURRENT_DIR,'nlp')
vectorize_file=os.path.join(_nlp, 'vectorizer')
model_file = os.path.join(_nlp, 'logistic.model')

model = joblib.load(model_file)
vectorize=joblib.load(vectorize_file)


# Create your views here.
def home(request):
	return render(request,'base.html')
def api_sentiment_pred(request):
	fin_res="result_after_prediction_to_be_returned"
	try:
	    review = [request.GET['review']]
	    review=vectorize.transform(review)
	    res=model.predict(review)
	    if(res[0]<abs(1-res[0])):
	    	fin_res="negative"
	    else:
	    	fin_res="positive"
	except:
		fin_res="some error occured"
	return (JsonResponse(fin_res, safe=False))
