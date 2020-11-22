from wsgiref.simple_server import make_server
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
import torch
from torchvision import transforms
import numpy as np
from util import base64_to_pil




device='cuda' if torch.cuda.is_available() else 'cpu'

# Declare a flask app
app = Flask(__name__)

model=torch.load('F:/PYTORCH/utils_Ren/模型部署/model.pth').to(device)
class_map={0:'hot_dog',1:'non_hot_dog'}

print('Model loaded. Check http://127.0.0.1:5000/')


def process_data(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")
 
    # Resize the input image nad preprocess it.
    image = transforms.Resize(224)(image)
    image = transforms.ToTensor()(image)
 
    # Convert to Torch.Tensor and normalize.
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
 
    # Add batch_size axis.
    image = image[None]
    image = image.to(device)
    return image





def model_predict(img,model):
    model.eval()
    img = process_data(img)
    pre = model(img).argmax(dim=1).float().cpu()
    return pre



@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')
    #return '<h1>Home</h1>'

@app.route('/signin', methods=['GET'])
def signin_form():
    return '''<form action="/signin" method="post">
              <p><input name="username"></p>
              <p><input name="password" type="password"></p>
              <p><button type="submit">Sign In</button></p>
              </form>'''
@app.route('/signin',methods=['POST'])
def hello_world():
    # 需要从request对象读取表单内容：
    if request.form['username']=='admin' and request.form['password']=='password':
        return render_template('index.html')
    return '<h3>Bad username or password.</h3>'


@app.route('/predict',methods=['GET','POST'])
def predict():
    # 需要从request对象读取表单内容：
    if request.method == 'POST':
        render_template('index.html')
        img = base64_to_pil(request.json)
        prediction=model_predict(img,model).numpy().item()
        prediction_class=class_map[prediction]
        return jsonify(result=prediction_class)

    return '<h3>Bad username or password.</h3>'

if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    server = make_server('', 5000,app)
    server.serve_forever()





#在浏览器中输入 http://localhost:5000/test