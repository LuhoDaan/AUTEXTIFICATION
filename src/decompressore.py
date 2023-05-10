import bz2
import pickle

# Name of the compressed pickle file
filename = '/home/matteo/PycharmProjects/AUTEXTIFICATION/src/cache/NLPdataset_train_multi.pbz2'

# Open the compressed pickle file in read-binary mode
with bz2.BZ2File(filename, 'rb') as f:
    # Load the decompressed src from the pickle file
    data = pickle.load(f)

# Save the src to a text file
with open('/home/matteo/PycharmProjects/AUTEXTIFICATION/src/cache/decompressed.txt', 'w') as f:
    f.write(str(data))