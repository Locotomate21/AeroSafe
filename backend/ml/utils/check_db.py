from database.connection import SessionLocal
from models.models import Weather

db = SessionLocal()
records = db.query(Weather).all()
for r in records:
    print(r.city, r.temperature, r.humidity, r.description, r.timestamp)
db.close()
