import sqlite3
from typing import List, Dict
from dataclasses import dataclass

def distance_xyz(p1, p2):
    x = abs(p1[0]-p2[0])
    y = abs(p1[1]-p2[1])
    z = abs(p1[2]-p2[2])
    return (x**2+y**2+z**2)**0.5

@dataclass
class Advertisement:
    id: int
    name: str
    url: str
    image: bytes
    position: tuple
    floor: int

class AdManager:
    advertisements: Dict[int, Advertisement]

    def __init__(self, database_path):
        self.advertisements = {}
        self.connection = sqlite3.connect(database_path)
        self.cursor = self.connection.cursor()
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS advertises (
        advertise_id INTEGER PRIMARY KEY AUTOINCREMENT,
        advertise_name TEXT UNIQUE,
        advertise_url TEXT,
        advertise_image BLOB,
        pose_x REAL,
        pose_y REAL,
        pose_z REAL,
        floor INTEGER
        )
        ''')
        # get the number of ads
        self.cursor.execute('''
        SELECT COUNT(*) FROM advertises
        ''')
        self.ad_count = self.cursor.fetchone()[0]
        if self.ad_count > 0:
            self.cursor.execute('''
            SELECT * FROM advertises
            ''')
            ads_items = self.cursor.fetchall()
            
            for ad_item in ads_items:
                self.advertisements[ad_item[0]] = Advertisement(ad_item[0], ad_item[1], ad_item[2], ad_item[3], (ad_item[4], ad_item[5], ad_item[6]), ad_item[7])
        print('init ad manager', self.ad_count)
        self.connection.commit()
    
    
    def AddAd(self, name: str, url: str, image_path: str, position: List[float], floor):
        # read image and convert to blob
        image = open(image_path, 'rb').read()
        image_blob = sqlite3.Binary(image)
        self.cursor.execute('''
        INSERT INTO advertises (advertise_name, advertise_url, advertise_image, pose_x, pose_y, pose_z, floor)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, url, image, position[0], position[1], position[2], floor))
        self.connection.commit()
        self.ad_count += 1
        self.advertisements[self.ad_count] = Advertisement(self.ad_count, name, url, image_blob, (position[0], position[1], position[2]), floor)
    
    def GetAdsByPositionAndFloor(self, position, floor):
        # get ads 15m near the position and in the same floor
        ads = []
        for ad in self.advertisements.values():
            if ad.floor == floor:
                dis = distance_xyz(position, ad.position)
                if dis < 15:
                    ads.append(ad)
                else:
                    print('ad too far for', dis)
        return ads
    

        
def Test():
    ad_manager = AdManager('Ads.db')
    ad_manager.AddAd('test', 'test', 'user58.jpg', [0, 0, 0])
    

# Test()