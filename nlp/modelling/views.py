from django.shortcuts import render
import os
import pickle
from django.http import JsonResponse
from sklearn.externals import joblib
import tensorflow as tf
global graph,model
graph = tf.get_default_graph()

CURRENT_DIR = os.path.dirname(__file__)
_nlp=os.path.join(CURRENT_DIR,'nlp')
vectorize_file=os.path.join(_nlp, 'vectorizer')
model_file = os.path.join(_nlp, 'model')

model = pickle.load(open(model_file, 'rb'))
vectorize=pickle.load(open(vectorize_file, 'rb'))


# Create your views here.
def home(request):
	return render(request,'base.html')
def api_sentiment_pred(request):
	fin_res="result_after_prediction_to_be_returned"
	try:
	    review = [request.GET['review']]
	    print(review)
	    review=vectorize.transform(review)
	    res=[]
	    with graph.as_default():
	    	res= model.predict(review)
	    print(res)
	    if(res[0]<abs(1-res[0])):
	    	fin_res="negative"
	    else:
	    	fin_res="positive"
	except Exception as e:
    		print(e)
    		fin_res="some error ocuured"
	return (JsonResponse(fin_res, safe=False))
