import pickle
from helper import extract_features, scaling_data 
model = pickle.load(open('model.pkl','rb'))
def predict():
    # pass
#     data = request.get_json(force=True)

    audio_features = extract_features("Nirvana-Smells_Like_Teen_Spirit.mp3")
    scaled_audio_features = scaling_data(audio_features)
    # print(scaled_audio_features)
    prediction = model.predict(scaled_audio_features)
    print(prediction)
#     output = prediction[0]
#     return jsonify(output)    
predict()