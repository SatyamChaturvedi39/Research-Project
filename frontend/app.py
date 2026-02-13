from flask import Flask, render_template, request
import os

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__,
            template_folder=os.path.join(basedir, 'templates'),
            static_folder=os.path.join(basedir, 'static'))

@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        left_player = request.form.get("left_player")
        right_player = request.form.get("right_player")
        result = "Simulate " + left_player + " - " + right_player
        return render_template("index.html", result=result)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
