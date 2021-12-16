"""
Placehold handle for FastAPI functionality of loading the
model/transformers only when the API is started, instead
of whenever the POST method is called.
"""
clf = None
