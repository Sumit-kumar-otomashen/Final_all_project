from flask import Flask,render_template,request,redirect,url_for,flash

app  = Flask(__name__)

@app.route('/')
def home():
    return "<h2>Hello, World!</h2>"

@app.route('/welcome')
def welcome():
    return "Welcome to ABC Corporation"

@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)