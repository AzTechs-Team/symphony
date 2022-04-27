from flask import Flask, render_template, request, jsonify
import pickle
# not able to import
# from Werkzeug import secure_filename

app = Flask(__name__)
# model = pickle.load(open('model/model.pkl','rb'))

@app.route("/")
def hello_world():
    return "Server to deploy our model <3"

@app.route('/api/predict',methods=['POST'])
def predict():
#     data = request.get_json(force=True)
#     prediction = model.predict([[np.array(data['exp'])]])
#     output = prediction[0]
#     return jsonify(output)    
      f = request.files['music']
      f.save(f.filename)
      return 'file uploaded successfully'

@app.route('/api/upload')
def upload_file():
   return render_template('index.html')
	
@app.route('/api/uploader', methods = ['GET', 'POST'])
def upload_music():
   if request.method == 'POST':
      f = request.files['music']
      f.save(f.filename)
      return 'file uploaded successfully'

if __name__ == '__main__':
    app.run(port=5000, debug=True)