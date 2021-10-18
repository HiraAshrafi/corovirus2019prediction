import _pickle
from flask import Flask, render_template, request
app = Flask(__name__)

file = open('model.pkl', 'rb')
clf = _pickle.load(file)
file.close()
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == "POST":
        mydict = request.form
        fever = int(mydict['fever'])
        pain = int(mydict['pain'])
        age = int(mydict['age'])
        runnynoice = int(mydict['runnynoice'])
        Breathing = int(mydict['Breathing'])

        inputs = [fever, pain, age, runnynoice, Breathing]
        pred = clf.predict_proba([inputs])[0][1]
        print(pred)
        return render_template("show.html", inf=round(pred*100))
    return render_template("index.html")
    # return 'this is probality'+str(pred)


if __name__ == "__main__":
    app.run(debug=True)
