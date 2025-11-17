import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tf2onnx
import numpy as np

# --- Rebuild your exact model architecture ---
# (Make sure these values are correct!)
STATE_SIZE = 1210 # <--- IMPORTANT: Update this
ACTION_SIZE = 7   # <--- IMPORTANT: Update this

def _build_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(2048, input_dim=state_size, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# 1. Create a model instance and load your weights
agent_model = _build_model(STATE_SIZE, ACTION_SIZE)
agent_model.load_weights("weights_done/Best_Upto_now.h5")
print("Model weights loaded.")

# 2. Define the input signature
# This tells the converter the exact input shape
# The 'None' allows for a flexible batch size
input_signature = [
    tf.TensorSpec(shape=(None, STATE_SIZE), dtype=tf.float32, name="input")
]

# 3. Convert the model
# Note: 'opset=13' is a good, stable default
onnx_model, _ = tf2onnx.convert.from_keras(agent_model, 
                                          input_signature, 
                                          opset=13)
print("Model converted to ONNX.")

# 4. Save the .onnx file
with open("weights_done/best_upto_now.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Successfully saved robot_agent.onnx")