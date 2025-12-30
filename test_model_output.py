from keras.models import load_model
import numpy as np

model = load_model("sign_language_model_full.h5")

# Print model summary
model.summary()

# Create dummy input (64x64 RGB)
dummy_input = np.random.rand(1, 64, 64, 3)

# Predict on dummy input
prediction = model.predict(dummy_input)[0]
predicted_class_index = np.argmax(prediction)

print(f"\n🔎 Total classes predicted by model: {len(prediction)}")
print(f"Predicted class index: {predicted_class_index}")
print(f"Confidence score: {prediction[predicted_class_index]*100:.2f}%")
