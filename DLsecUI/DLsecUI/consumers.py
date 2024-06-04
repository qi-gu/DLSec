# consumers.py
from channels.generic.websocket import AsyncWebsocketConsumer
import json

class ScoreConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        pass

    async def send_score(self, event):
        score = event['score']
        await self.send(text_data=json.dumps({'score': score}))