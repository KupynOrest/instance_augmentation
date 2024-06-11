import random
from typing import Dict, Any, Optional
from tinydb import TinyDB


class DBWrapper:
    def __init__(self, db_path: str, keyname: str = "original_filename"):
        db = TinyDB(db_path)
        self.database = self.db_to_dict(db)
        self.keys = list(self.database.keys())
        self.keyname = keyname

    @staticmethod
    def db_to_dict(db) -> Dict[str, Any]:
        data = {}
        for x in db.all():
            if x["original_filename"] in data.keys():
                data[x["original_filename"]].append(x)
            else:
                data[x["original_filename"]] = [x]
        return data

    def search(self, key: str) -> Optional[Dict[str, Any]]:
        items = self.database.get(key, None)
        if items is not None:
            return random.choice(items)
        return None
