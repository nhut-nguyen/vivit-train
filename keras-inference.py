from keras import models

model_path = "vivit_model_20250520.keras"
model = models.load_model(model_path)
model.summary()