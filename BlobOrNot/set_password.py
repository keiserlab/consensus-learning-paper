import os
import numpy as np
password_file = 'password.npy'
if os.path.exists(password_file):
    password = np.load(password_file).item()
else:
    password = {}
# todo: this is not a good way to store user credentials
# key is username value is password
password['user1'] = 'test1'
password['user2'] = 'test2'
np.save(password_file, password)
