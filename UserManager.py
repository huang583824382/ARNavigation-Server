from enum import Enum
import json
import websockets
from typing import List, Dict, Tuple, Set
import time
import numpy as np
from Database import SQLite3Manager, UserPose, User
from unittest.mock import MagicMock
from AdManager import AdManager

class UserStatus(Enum):
    Offline = 0
    Online = 1
    Navigating = 2

class UserLocStatus(Enum):
    Uninitialized = 0
    Ready = 1
    NeedRelocation = 2
    LocationExpired = 3

class UserInstance:
    name: str
    status: UserStatus
    locStatus: UserLocStatus
    pose: Tuple[List[float], List[float]] # tvec, qvec
    navigationPath: Dict # path = [nodeIDs]
    lastUpdate: float
    intrinsic: List[float]
    
    def __init__(self, name, intrinsic) -> None:
        self.status = UserStatus.Offline
        self.locStatus = UserLocStatus.Uninitialized
        self.pose = ([0, 0, 0], [0, 0, 0, 0])
        self.navigationPath = None
        self.name = name
        self.status = UserStatus.Online
        self.intrinsic = intrinsic
        self.ads = []
        print('create user:', name, intrinsic)

    def UpdatePose(self, tvec: list, qvec: list, floor: int = None):

        # print('update pose:', tvec, qvec)
        self.pose = (tvec, qvec)
        self.lastUpdate = time.time()
        self.locStatus = UserLocStatus.Ready
        if floor is not None:
            self.floor = floor
        else:
            self.floor = 0
    
    def StartNavigation(self, navigationPath):
        self.navigationPath = navigationPath
        self.status = UserStatus.Navigating

    def StopNavigation(self):
        self.navigationPath = None
        self.status = UserStatus.Online
    
    def GetPosition(self):
        return self.pose[0]
    
    def Status_Online(self):
        self.status = UserStatus.Online
    
    def Status_Offline(self):
        self.status = UserStatus.Offline
    
    
class UserManager:
    users: Dict[websockets.WebSocketServerProtocol, UserInstance]
    userName2Websocket: Dict[str, websockets.WebSocketServerProtocol]
    sharingUsers: Set[websockets.WebSocketServerProtocol]
    DBManager: SQLite3Manager
    adManager: AdManager

    def __init__(self) -> None:
        self.users = dict()
        self.sharingUsers = set()
        self.DBManager = SQLite3Manager('ARNavigationData.db')
        self.adManager = AdManager("Ads.db")

    def AddUser(self, websocket, name = 'default_name', intrinsic = None, navigating = False):
        if self.FindUserByName(name) is not None:
            print('user already exists')
            ws, user = self.FindUserByName(name)
            self.RemoveUser(ws)
            self.users[websocket] = user
            user.status = UserStatus.Online if navigating is False else UserStatus.Navigating
            if ws in self.sharingUsers:
                self.sharingUsers.remove(ws)
                # 留到main中去加入
            if intrinsic is None:
                intrinsic = [0, 0, 0, 0]
            user.ads.clear()
            self.DBManager.add_or_update_user(User(None, name, tuple(intrinsic)))
        else:
            print('user not exists')
            user = UserInstance(name, intrinsic)
            self.users[websocket] = user
            if intrinsic is None:
                intrinsic = [0, 0, 0, 0]
            self.DBManager.add_or_update_user(User(None, name, tuple(intrinsic)))

    def RemoveUser(self, websocket):
        if self.users.get(websocket) is None:
            print('user not found')
            return
        self.users.pop(websocket)

    def BroadcastSharingUsers(self, package: bytes):
        websockets.broadcast(self.sharingUsers, package)

    def StartNavigation(self, websocket, navigationPath):
        if self.users.get(websocket) is None:
            return
        self.users[websocket].StartNavigation(navigationPath)

    def StopNavigation(self, websocket):
        if self.users.get(websocket) is None:
            return
        self.users[websocket].StopNavigation()

    def UpdatePose(self, websocket, tvec: list, qvec: list, floor: int = None):
        if self.users.get(websocket) is None:
            return
        self.users[websocket].UpdatePose(tvec, qvec, floor)
        name = self.users[websocket].name
        # TODO: floor havent been saved in DB
        if floor is None:
            floor = 0
        self.DBManager.add_user_pose(UserPose(None, self.DBManager.find_user_by_name(name).id, tuple(tvec), tuple(qvec), floor, str(time.time())))
        

    def QueryForAds(self, websocket):
        ads = self.adManager.GetAdsByPositionAndFloor(self.users[websocket].GetPosition(), self.users[websocket].floor)
        # if self.users[websocket].ads not equal with ads
        if self.users[websocket].ads != ads:
            self.users[websocket].ads = ads
            # get ads from adManager
            # return ads and send
            print('query for ads', len(ads))
            return ads
        else:
            print('query for ads but no new ad found')
            return None

    def UpdateLocStatus(self, websocket, locStatus: int):
        if self.users.get(websocket) is None:
            print('user not found')
            return
        print('update loc status', self.users[websocket].name, locStatus)
        if locStatus == 0:
            self.users[websocket].locStatus = UserLocStatus.Uninitialized
        elif locStatus == 1:
            self.users[websocket].locStatus = UserLocStatus.Ready
        elif locStatus == 2:
            self.users[websocket].locStatus = UserLocStatus.NeedRelocation
        elif locStatus == 3:
            self.users[websocket].locStatus = UserLocStatus.LocationExpired
    
    def FindUserByConnection(self, websocket):
        return self.users.get(websocket)
    
    def FindUserByName(self, name):
        for ws, user in self.users.items():
            if user.name == name:
                return ws, user
        return None
    
    def AddSharing(self, websocket):
        print('add sharing', self.FindUserByConnection(websocket).name)
        self.sharingUsers.add(websocket)

    def RemoveSharing(self, websocket):
        if websocket in self.sharingUsers:
            print('remove sharing', self.FindUserByConnection(websocket).name)
            self.sharingUsers.remove(websocket)

    def GetBroadcastInfo(self):
        print('GetBroadcastInfo')
        userList = []
        for sharingUserWS in self.sharingUsers:
            user = self.users[sharingUserWS]
            print('user:', user.name)
            userInfo  = {}
            pose = {}
            state = {}
            userInfo['name'] = user.name
            pose['tvec'] = user.pose[0]
            pose['qvec'] = user.pose[1]
            userInfo['pose'] = pose
            state['status'] = user.status.value
            state['locStatus'] = user.locStatus.value
            userInfo['state'] = state
            userInfo['floor'] = user.floor
            if user.status == UserStatus.Navigating and user.navigationPath is not None:
                userInfo['path'] = user.navigationPath
            userList.append( userInfo)
        message={}
        message['Sharing'] = userList
        print('Broadcast message:',message)
        return json.dumps(message)

def test():
    websocket_mock1 = MagicMock(spec = websockets.WebSocketServerProtocol)
    websocket_mock2 = MagicMock(spec = websockets.WebSocketServerProtocol)
    userManager = UserManager()
    userManager.AddUser(websocket_mock1, 'test1', [1, 2, 3, 4])
    userManager.AddUser(websocket_mock2, 'test2', [1, 2, 3, 4])

    # print(userManager.FindUserByConnection(websocket_mock).name)
    userManager.UpdatePose(websocket_mock1, [1, 2, 3], [1, 2, 3, 4])
    userManager.UpdatePose(websocket_mock2, [1, 2, 3], [1, 2, 3, 4])
    userManager.DBManager.clear()

# test()