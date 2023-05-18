from tensorflow import keras
import numpy as np

benim_modelim = keras.models.load_model('celcius_modeli.h5')

celcius_degrees = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_degrees = np.array([-40, 14, 32, 46.4, 59, 71.6, 100.4], dtype=float)



def ValidateModel(model, celcius_degrees, fahrenheit_degrees):
  predictions = model.predict(celcius_degrees)

  for idx in range(len(predictions)):
    print(f"[{celcius_degrees[idx]}] C degree -> Predicted Fahrenheit: {round(predictions[idx][0], 2)} F - Real Fahrenheit: {fahrenheit_degrees[idx]} F")


ValidateModel(benim_modelim, celcius_degrees, fahrenheit_degrees)


user_celcius = float(input("Please enter the Celcius value you want to predict of it's Fahrenheit: "))
print(benim_modelim.predict(np.array([user_celcius])))