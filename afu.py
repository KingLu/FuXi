#encoding utf-8
from flask import Flask, render_template,request
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import base64
import predict_2 as pr2


import sys 
import os
sys.path.append(os.path.abspath("./model"))
from load import * 

app = Flask(__name__)


	

def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	#print(imgstr)
	with open('output.png','wb') as output:
		output.write(base64.b64decode(imgstr))
	

@app.route('/')
def index():
	#initModel()
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
  imgData = request.get_data()
  #print(imgData)
  convertImage(imgData)
  print("图像转化完成")
  imvalue = pr2.imageprepare('output.png')
  predint = pr2.predictint(imvalue)
  
  print("数字识别完成")
  out = predint[0]
  print(out)
  response = np.array_str(out)
  return response	
	

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)
	#app.run(debug=True)
