from flask import Flask, render_template, request

app = Flask(__name__)

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
