import pickle
import pandas as pd

# Correct file path
# file_path = r"C:\Users\a\Desktop\Fitness-AI\thesis_bidirectionallstm_label_encoder.pkl"

# Open and read the .pkl file
# with open(file_path, "rb") as file:  # Open file in binary read mode
   # data = pickle.load(file)  # Pass the file object to pickle.load

# Print the loaded data
# print(data)

demo=pd.read_pickle(r'C:\Users\a\Desktop\Fitness-AI\thesis_bidirectionallstm_label_encoder.pkl')
print(demo)
