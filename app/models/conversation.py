from datetime import datetime, timedelta, timezone
from . import db

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    messages = db.relationship('Message', backref='conversation', lazy=True, cascade='all, delete-orphan')

    def get_indian_time(self):
        ist_offset = timedelta(hours=5, minutes=30)
        ist_time = self.created_at + ist_offset
        return ist_time.strftime('%H:%M')

    def get_full_indian_time(self):
        ist_offset = timedelta(hours=5, minutes=30)
        ist_time = self.created_at + ist_offset
        return ist_time.strftime('%d %b %Y, %H:%M:%S')

    def get_friendly_date(self):
        ist_offset = timedelta(hours=5, minutes=30)
        ist_time = self.created_at + ist_offset
        now = datetime.now(timezone.utc) + ist_offset
        today = now.date()
        message_date = ist_time.date()

        if message_date == today:
            return "Today" 
        elif message_date == today - timedelta(days=1):
            return "Yesterday"
        else:
            return ist_time.strftime('%d %b %Y')