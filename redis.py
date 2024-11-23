import redis

# Connect to the local Redis server
r = redis.StrictRedis(host="localhost", port=6379, db=0)
# Check if the connection is successful
print(r.ping())  # Should return True if connected successfully


import torch
import pickle

features = torch.randn(10, 10)  # Example features (e.g., a batch of feature vectors)
# Serialize the tensor using pickle (or use torch.save for tensors)
serialized_features = pickle.dumps(features)
# serialized_features = torch.save(features)

# Store the serialized features in Redis using a key
r.set('model_feature_1', serialized_features)

serialized_features = r.get('model_feature_1')
# Deserialize the features back into a tensor (or NumPy array, depending on the original format)
features = pickle.loads(serialized_features)
# Use the features (e.g., pass to your model)
print(features)

r.close()
