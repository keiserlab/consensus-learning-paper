from flask import Flask
from flask_sqlalchemy import SQLAlchemy

application = Flask('blobornot')
application.config.from_object('config')

try:
	db = SQLAlchemy(application)
except:
	print('dbinit')

