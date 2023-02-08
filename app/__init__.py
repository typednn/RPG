import os

if 'DATA_PATH' in os.environ:
    DATA_PATH = os.environ['DATA_PATH']
else:
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')