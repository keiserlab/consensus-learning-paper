import os
import time
import csv
from database import application, db
from database.models import TileAnnotation as TileLabel
import pandas as pd

# https://stackoverflow.com/questions/2952366/dump-csv-from-sqlalchemy

BACKUP_DIR = ''
backup = pd.DataFrame()
all_record = TileLabel.query.all()
localtime = time.strftime("%Y%m%d%H%M%S", time.localtime())
filename = f"backup_{localtime}_{application.config['REGION']}.csv"
csv_path = os.path.join(BACKUP_DIR, filename)

with open(csv_path, 'w') as csv_file:
    out_csv = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    header = TileLabel.__table__.columns.keys()
    out_csv.writerow(header)

    for record in all_record:
        out_csv.writerow([getattr(record, c) for c in header])


print(f"Backup at {csv_path}")
print(f"{len(all_record)} records")

