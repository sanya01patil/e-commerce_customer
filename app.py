from flask import Flask, render_template
from model import train_and_plot

app = Flask(__name__)

@app.route('/')
def home():
    summary = train_and_plot()
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
