from database import db
from database.models import TileAnnotation

# delete all records
try:
	import backup_database
	num_rows_deleted = db.session.query(TileAnnotation).delete()
	db.session.commit()
	print(num_rows_deleted, ' records deleted')
except:
	print('error')
	db.session.rollback()

# db initialization	
db.drop_all()

db.create_all()

print("DB created.")
