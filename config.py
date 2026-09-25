import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPLIES_PATH = os.path.join(DATA_DIR, "replies.json")
CONVERSATIONS_PATH = os.path.join(DATA_DIR, "conversations.json")

HOST = "0.0.0.0"
PORT = 5000
DEBUG = True
