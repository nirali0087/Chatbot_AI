import json
import numpy as np
from datetime import datetime, timedelta, timezone
from . import db


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    is_user = db.Column(db.Boolean, nullable=False)
    content = db.Column(db.Text, nullable=False)

    embedding = db.Column(db.LargeBinary, nullable=True)

    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


    def set_embedding(self, embedding_vector):
        if embedding_vector is None:
            self.embedding = None
            return
        
        if isinstance(embedding_vector, np.ndarray):
            arr = embedding_vector.tolist()
        else:
            arr = list(embedding_vector)
        self.embedding = json.dumps(arr).encode('utf-8')

    def get_embedding(self):
        if self.embedding is None:
            return None
       
        arr = json.loads(self.embedding.decode('utf-8'))
        return np.array(arr, dtype=np.float32)


    def get_indian_time(self):
        ist_offset = timedelta(hours=5, minutes=30)
        ist_time = self.timestamp + ist_offset
        return ist_time.strftime('%H:%M')

    def get_full_indian_time(self):
        ist_offset = timedelta(hours=5, minutes=30)
        ist_time = self.timestamp + ist_offset
        return ist_time.strftime('%d %b %Y, %H:%M:%S')

    def get_friendly_date(self):
        ist_offset = timedelta(hours=5, minutes=30)
        ist_time = self.timestamp + ist_offset
        now = datetime.utcnow() + ist_offset
        today = now.date()
        message_date = ist_time.date()
    
        if message_date == today:
            return "Today"
        elif message_date == today - timedelta(days=1):
            return "Yesterday"
        else:
            return ist_time.strftime('%d %b %Y')
