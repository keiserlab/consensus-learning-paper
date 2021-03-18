import os

# Flask secret key for sessions - https://flask.palletsprojects.com/en/1.1.x/quickstart/#sessions
SECRET_KEY = b'\xef\xfeT\x12bN~\xe7LF\xe2jRuR\xcb'

# export BLOBORNOT_RUN_LOCAL environment variable to test locally
RUN_LOCAL = 'BLOBORNOT_RUN_LOCAL' in os.environ
RUN_CLOUD = not RUN_LOCAL

# Database
# format: mysql+pymysql://(user):(password)@(db_endpoint):3306/(db_name)
DB_NAME = os.environ['RDS_DB_NAME']
DB_USER = os.environ['RDS_USERNAME']
DB_PASSWORD = os.environ['RDS_PASSWORD']
DB_HOST = os.environ['RDS_HOSTNAME']
DB_PORT = os.environ['RDS_PORT']
SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_POOL_RECYCLE = 3600
#https://flask-wtf.readthedocs.io/en/stable/form.html
WTF_CSRF_ENABLED = True
# to use local sqlite database, uncomment this line
# SQLALCHEMY_DATABASE_URI = 'sqlite:///test.db'

# S3 - images are pulled from S3 even when running locally
# region environment variable defines the region the app is in
REGION = os.getenv('AWS_REGION', 'us-west-1')
# S3
S3_ACCESS_KEY_ID = os.getenv('S3_ACCESS_KEY_ID')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')
S3_REGION_NAME = REGION
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')