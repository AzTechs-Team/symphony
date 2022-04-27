from helper import extract_features, scaling_data 
import tensorflow as tf
# from keras.models import load_model
import numpy as np

load_options = tf.saved_model.LoadOptions(
    experimental_io_device='/job:localhost')
saved_model_path = '/tmp/tf_save'

loaded_model = tf.keras.models.load_model(saved_model_path, options=load_options)
print("faklsdjflkadsjfklas", loaded_model)

def predict():
    # pass
#     data = request.get_json(force=True)

    # audio_features = extract_features(r"D:\\college_stuff\\ai\\symphony\\src\\Nirvana-Smells_Like_Teen_Spirit.mp3")
    # scaled_audio_features = scaling_data(audio_features)
    scaled_audio_features = np.array([[0.00000000e+00, -1.33598104e+00,  5.36500230e-02,
                                       -1.77334208e+00, -7.39886776e-01, -7.33024614e-01,
                                       1.14873943e+00, -4.31429970e-01,  5.35220089e+00,
                                       -7.93439597e-01,  2.09461450e+00, -5.81638573e-01,
                                       2.90479938e-01,  2.14759923e-01, -9.90301055e-01,
                                       -2.25873462e-01, -2.75228665e+00,  1.67699220e+00,
                                       1.01789770e+00,  3.51469665e+00,  4.34793434e-01,
                                       4.34742811e-01, -1.94797087e+00,  1.71178794e-03,
                                       1.62016732e-01,  4.78006604e-02, -1.83531863e+00,
                                       2.23550981e-01,  3.33685276e-01,  3.57795501e-02,
                                       -1.59426617e+00,  1.05157538e+00,  1.18464984e+00,
                                       4.07761392e-01, -1.77856210e+00,  2.35331962e-01,
                                       -2.27522539e-02,  3.12989412e-01, -1.32957899e+00,
                                       9.53584612e-01,  8.85032776e-02,  4.40464784e-01,
                                       -7.77956714e-01, -2.40423945e-01, -1.62113873e+00,
                                       6.17138921e-02, -1.59465564e+00,  7.20020374e-02,
                                       -4.64691558e-01, -1.60304775e-01, -1.99163586e+00,
                                       3.18419629e-01, -2.36777030e+00,  1.94663412e-01,
                                       -1.42133801e+00,  9.67286833e-02]])
    # print(scaled_audio_features)
    prediction = loaded_model.predict(scaled_audio_features)
    print(prediction)
#     output = prediction[0]
#     return jsonify(output)    
predict()