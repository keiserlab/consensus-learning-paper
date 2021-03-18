from database import db
from datetime import datetime


class TileAnnotation(db.Model):
	# __tablename__ = 'tile_annotations'
	id = db.Column(db.Integer, primary_key=True)
	tilename = db.Column(db.String(80), unique=False, nullable=False)
	username = db.Column(db.String(80), unique=False, nullable=False)
	rot = db.Column(db.Integer, unique=False, nullable=False, default=0)
	cored = db.Column(db.Boolean, default=False, nullable=False)
	diffuse = db.Column(db.Boolean, default=False, nullable=False)
	CAA = db.Column(db.Boolean, default=False, nullable=False)
	negative = db.Column(db.Boolean, default=False, nullable=False)
	notsure = db.Column(db.Boolean, default=False, nullable=False)
	flag = db.Column(db.Boolean, default=False, nullable=False)
	timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)

	def __repr__(self):
		return (f'<TileAnnotation('
				f'tilename: {self.tilename}, '		
				f'username {self.username}, '
				f'rot: {self.rot}, '		
				f'cored: {self.cored}, '
				f'diffuse: {self.diffuse}, '
				f'CAA: {self.CAA}, '
				f'negative: {self.negative}, '
				f'notsure: {self.notsure}, '
				f'flag: {self.flag}>')
