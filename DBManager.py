import sqlite3
from typing import List
from dataclasses import dataclass

@dataclass
class UserPose:
    id: int
    user_id: int
    position: tuple
    rotation: tuple
    floor: int
    timestamp: str

@dataclass
class User:
    id: int
    name: str
    intrinsic: tuple

class SQLite3Manager:
    def __init__(self, database_name: str):
        self.connection = sqlite3.connect(database_name)
        self.cursor = self.connection.cursor()

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_name TEXT UNIQUE,
        focal_length REAL,
        cy REAL,
        cx REAL,
        dist_coeff REAL
        )
        ''')
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_poses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER REFERENCES users(user_id),
        pose_x REAL,
        pose_y REAL,
        pose_z REAL,
        rot_w REAL,
        rot_x REAL,
        rot_y REAL,
        rot_z REAL,
        floor INTEGER,
        timestamp TEXT
        )
        ''')

        self.connection.commit()

    

    def add_user_pose(self, user_pose: UserPose):
        sql = '''
        INSERT INTO user_poses (user_id, pose_x, pose_y, pose_z, rot_w, rot_x, rot_y, rot_z, floor, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        values = (user_pose.user_id, user_pose.position[0], user_pose.position[1], user_pose.position[2], user_pose.rotation[0], user_pose.rotation[1], user_pose.rotation[2], user_pose.rotation[3], user_pose.floor, user_pose.timestamp)
        self.cursor.execute(sql, values)
        self.connection.commit()

    # def add_user(self, user: User):
    #     sql = '''
    #     INSERT INTO users (user_name, focal_length, cy, cx, dist_coeff)
    #     VALUES (?, ?, ?, ?, ?)
    #     '''
    #     values = (user.name, user.intrinsic[0], user.intrinsic[1], user.intrinsic[2], user.intrinsic[3])
    #     self.cursor.execute(sql, values)
    #     self.connection.commit()

    # add user if not exists, or update user if same name
    def add_or_update_user(self, user: User):
        sql = '''
        INSERT OR REPLACE INTO users (user_name, focal_length, cy, cx, dist_coeff)
        VALUES (?, ?, ?, ?, ?)
        '''
        values = (user.name, user.intrinsic[0], user.intrinsic[1], user.intrinsic[2], user.intrinsic[3])
        self.cursor.execute(sql, values)
        self.connection.commit()
        print('add_or_update_user')
    

    def get_all_user_poses(self) -> List[UserPose]:
        sql = 'SELECT * FROM user_poses'
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()
        user_poses = []
        for row in rows:
            user_pose = UserPose(
                id=row[0],
                user_id=row[1],
                position=(row[2], row[3], row[4]),
                rotation=(row[5], row[6], row[7], row[8]),
                floor=row[9],
                timestamp=row[10]
                )
            user_poses.append(user_pose)
        return user_poses
    
    def clear_user_pose(self):
        self.cursor.execute('DELETE FROM user_poses')
        self.connection.commit()
    
    def clear_user(self):
        self.cursor.execute('DELETE FROM users')
        self.connection.commit()
    
    def clear(self):
        self.clear_user_pose()
        self.clear_user()

    def close(self):
        self.connection.close()

    def find_user_by_name(self, name: str) -> User:
        sql = 'SELECT * FROM users WHERE user_name=?'
        self.cursor.execute(sql, (name,))
        row = self.cursor.fetchone()
        if row is None:
            return None
        user = User(
            id=row[0],
            name=row[1],
            intrinsic=(row[2], row[3], row[4], row[5])
        )
        return user


# manager = SQLite3Manager('test.db')
# manager.add_user_pose(UserPose(None, 1, (1, 2, 3), (4, 5, 6, 7), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
# manager.add_user_pose(UserPose(None, 1, (1, 2, 3), (4, 5, 6, 7), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
# res = manager.get_all_user_poses()
# print(res)
# manager.clear()