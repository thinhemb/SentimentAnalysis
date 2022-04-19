from flask import Flask, render_template,request
import os
import sys
sys.path.append('./main/')
from local_test import predict as pre



app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('home.html')


@app.route('/predict/', methods=['GET','POST'])
def pred():
    
    if request.method == "POST":       
        #get form data
        text = request.form.get('text')
        predict = pre(text)  
        
        return render_template('home.html',predict= predict)
    pass
    
    

if __name__=="__main__":
    app.run()