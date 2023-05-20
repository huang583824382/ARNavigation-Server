import cv2
import struct
import numpy as np
import threading
import websockets
import asyncio
import queue
import os, time, base64, shutil
from hloc.my_localize_lib import MainWork, localize_image_multifloors
import json
from typing import Union
from pathlib import Path
from mapManager import MapManager
from UserManager import UserManager

CONNECTIONS = list()

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def img_rotate(src, angel):
    h,w = src.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angel, 1.0)
    # 调整旋转后的图像长宽
    rotated_h = int((w * np.abs(M[0,1]) + (h * np.abs(M[0,0]))))
    rotated_w = int((h * np.abs(M[0,1]) + (w * np.abs(M[0,0]))))
    M[0,2] += (rotated_w - w) // 2
    M[1,2] += (rotated_h - h) // 2
    # 旋转图像
    rotated_img = cv2.warpAffine(src, M, (rotated_w,rotated_h))

    return rotated_img

class Singleton(object):
    def __init__(self, PackageWorker):
        self._PackageWorker = PackageWorker
        self._instance = {}

    def __call__(self):
        if self._PackageWorker not in self._instance:
            self._instance[self._PackageWorker] = self._PackageWorker()
        return self._instance[self._PackageWorker]


@Singleton
class Package:
    types = {'locRequest':1, 
             'locRet':2,
             'errorInfo':3,
             'initUser':4,
             'navRequest':5,
             'pathInfo':6,
             'sharePos':7,
             'updatePose':8,
             'broadcast':9,
             "adInfo":10}
    
    def generate_package(self, type: int, data: Union[bytes, None])->bytes:
        length = 0
        if data is not None:
            length = len(data)
        frame = struct.pack('<h{}s'.format(length), type, data)
        # print(frame)
        return frame
    
    def parse_package(self, websocket: websockets.WebSocketServerProtocol, frame: bytes):
        # 以数字打印frame前10个字节
        for i in range(10):
            print(frame[i], end=' ')

        # print()
        head = struct.unpack('<h', frame[:2])
        # frameHead = head[0]
        # if frameHead != self.frameHead: 
        #     raise ValueError('Get Error Package')
        type = head[0]
        print('Get Package, type is', type, 'length =', len(frame)-2)
        # length = head[2]
        # print('length:', length, 'actual data length =', len(frame)-8)
        res = self.parse_payload(type, frame[2:], websocket)
        return type, res

    def parse_payload(self, type: int, payload: bytes, websocket: websockets.WebSocketServerProtocol):
        global dataset_id, saveDatasetPath, ds_img_num
        # if type == self.types['text']:
        #     print('Get text message:', payload.decode())
        #     return payload.decode()
        if type == self.types['locRequest']:
            print('Get image message')
            d = np.frombuffer(payload, dtype=np.uint8)
            img = cv2.imdecode(d, cv2.IMREAD_COLOR)
            img = img_rotate(img, 90)
            name = userManager.FindUserByConnection(websocket).name
            cv2.imwrite(name+'.jpg', img) 
            locateQueue.put(websocket)
            return None
        
        elif type == self.types['navRequest']:
            json_msg = json.loads(payload.decode())
            print(payload.decode())
            des = json_msg['Destination']
            # 先更新用户的位置
            if(des == ''):
                userManager.StopNavigation(websocket)
                return None
            else:
                print('Get navigation to', des)
                # 解析当前位置和目标地点
                tvec = json_msg['Pose']['tvec']
                qvec = json_msg['Pose']['qvec']
                userManager.UpdatePose(websocket, tvec, qvec)
                desFloor = json_msg['DesFloor']
                srcFloor = json_msg['SrcFloor']
                mode = json_msg['Mode']
                path = mapManager.FindPath(userManager.FindUserByConnection(websocket), srcFloor, des, desFloor, mode)
                if len(path) == 0:
                    print('User not ready')
                    return False
                print('find path:',path)
                pathInfo = mapManager.WritePath2JSON(path, 'path.json')
                userManager.StartNavigation(websocket, pathInfo)
                return True
            
        elif type == self.types['initUser']:
            print('Get init')
            msg = payload.decode()
            print(msg)
            initJson = json.loads(msg)
            userManager.AddUser(websocket, initJson['Name'], initJson['Intrinsic'], initJson['Navigating'])
            if initJson.get('Sharing') is not None and initJson['Sharing'] is True:
                userManager.AddSharing(websocket)
            return msg

        elif type == self.types['sharePos']:
            print('Get share')
            msg = payload.decode()
            shareJSON = json.loads(msg)
            if shareJSON['Share'] == True:
                userManager.AddSharing(websocket)
                BroadcastSharingUsers(userManager)
            elif shareJSON['Share'] == False:
                userManager.RemoveSharing(websocket)
                BroadcastSharingUsers(userManager)
            return msg
        
        elif type == self.types['updatePose']:
            print('Get updatePose')
            msg = payload.decode()
            # print(msg)
            poseJSON = json.loads(msg)
            tvec = poseJSON['Pose']['tvec']
            qvec = poseJSON['Pose']['qvec']
            floor = poseJSON['Floor']
            userManager.UpdatePose(websocket, tvec, qvec, floor)
            userManager.UpdateLocStatus(websocket, poseJSON['LocStatus'])
            if websocket in userManager.sharingUsers:
                BroadcastSharingUsers(userManager)
            # query for ads
            ads = userManager.QueryForAds(websocket)
            if ads is not None:
                return ads
            else:
                return None

def BroadcastSharingUsers(userManager):
    global packageWorker
    print('BroadcastSharingUsers')
    msg = userManager.GetBroadcastInfo()
    package = packageWorker.generate_package(packageWorker.types['broadcast'], msg.encode())
    userManager.BroadcastSharingUsers(package)

def clear_dataset(dataset: Union[Path, str]):
    outputs = Path('outputs/'+dataset.name)
    if outputs.exists():
        shutil.rmtree(outputs)
    if dataset.exists():
        shutil.rmtree(dataset)
        os.mkdir(dataset)

async def recvMain(websocket):
    print("recvMain in")
    global mainWork, mapManager, userManager
    try:
        async for message in websocket:
            type, res = packageWorker.parse_package(websocket, message)
            await ReplyClient(websocket, type, res)
    except Exception as e:
        user = userManager.FindUserByConnection(websocket)
        if user is not None:
            user.Status_Offline()
            userManager.RemoveSharing(websocket)
        print("Exception raised:", e.with_traceback())
        

async def ReplyClient(websocket, type, res):
    if type == packageWorker.types['locRequest']: 
        pass
        
    elif type == packageWorker.types['navRequest']: 
        if res is False:
            print('navigation failed, send error')
            replyFrame = packageWorker.generate_package(packageWorker.types['errorInfo'], "navigation failed, not initialized".encode())
            await websocket.send(replyFrame)
            return
        elif res is True:
            print('send navigation result')
            pathJSON = open('path.json', 'r', encoding='utf8').read()
            replyFrame = packageWorker.generate_package(packageWorker.types['pathInfo'], pathJSON.encode())
            await websocket.send(replyFrame)
        elif res is None:
            print("navigation finished")

    elif type == packageWorker.types['initUser']:
        print('send init')
        places_json = open(mapManager.StorePlaces(), 'r').read()
        # 组合起来
        replyFrame = packageWorker.generate_package(packageWorker.types['initUser'], places_json.encode())
        await websocket.send(replyFrame)

    elif type == packageWorker.types['updatePose']:
        if res is not None:
            print('send ads')
            msg = {}
            for ad in res:
                item = {}
                item['name'] = ad.name
                item['url'] = ad.url
                b64_str = base64.b64encode(ad.image).decode('utf-8')
                item['image'] = b64_str

                item['position'] = ad.position
                item['floor'] = ad.floor
                msg[ad.id] = item
            replyFrame = packageWorker.generate_package(packageWorker.types['adInfo'], json.dumps(msg).encode())
            await websocket.send(replyFrame)



async def serverStart():
    print("websocket server start at", '127.0.0.1')
    async with websockets.serve(recvMain, '127.0.0.1', 6006):
        await asyncio.Future()  # run forever

def locateWorker():
    print('locateWorker start')
    global datasets
    while True:
        if locateQueue.empty() is False:
            websocket = locateQueue.get()
            name = userManager.FindUserByConnection(websocket).name
            ret, log, dataset = localize_image_multifloors(datasets, '', name+'.jpg', overwrite=True, ransac_thresh=12, intrinsic=userManager.FindUserByConnection(websocket).intrinsic)
            print('final dataset:',dataset)
            
            inliers = 0
            if log['PnP_ret'].get('num_inliers') is not None:
                inliers = int(log['PnP_ret']['num_inliers'])
            if ret['success'] is True and inliers > 50:
                print("query success")
                ret['qvec'] = ret['qvec'].tolist()
                ret['tvec'] = ret['tvec'].tolist()
                ret['camera']['params'] = ret['camera']['params'].tolist()
                log['keypoints_query'] = log['keypoints_query'].tolist()
                
                log['floor'] = floors[datasets.index(str(dataset))]
                # update the pose in mapManager
                with open('retlog.json', 'w') as fp:
                    json.dump(log, fp, cls=NpEncoder)
                    fp.close()
                # userManager.UpdatePose(websocket, ret['tvec'], ret['qvec'])
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(locateReply(websocket, True, log))
                except Exception as e:
                    print("Exception raised:", e)

            else:
                print("query failed", inliers)
                with open('retlog.json', 'w') as fp:
                    json.dump(log, fp, cls=NpEncoder)
                    fp.close()
                # 在线程中运行send_message协程
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(locateReply(websocket, False, log))
                except Exception as e:
                    print("Exception raised:", e)
                # await locateReply(websocket, False, log)

        else:
            time.sleep(0.1)


async def locateReply(websocket, res, reslog):
    if res is True:
        print('send localization result')
        replyFrame = packageWorker.generate_package(packageWorker.types['locRet'], json.dumps(reslog, cls=NpEncoder).encode())
        await websocket.send(replyFrame)
    elif res is False:
        print('localization failed, send error')
        numInliers = 0
        if reslog['PnP_ret'].get('num_inliers') is not None:
            numInliers = reslog['PnP_ret']['num_inliers']
        replyFrame = packageWorker.generate_package(packageWorker.types['errorInfo'], "location failed {}".format(numInliers).encode())
        await websocket.send(replyFrame)
                
transform_mB1 = np.array([[-0.327894, 0.008077, 0.000998, -3.541684], \
                        [0.008077, 0.327895, -0.000025, -4.810959], \
                        [-0.000998, 0.000000, -0.327993, -17.294159], \
                        [0, 0, 0, 1]])
zB1 = -17.4

transform_m1 = np.array([[-0.324391, 0.000000, 0.004362, -4.282498], \
                        [0.000000, 0.324420, 0.000000, -4.545443], \
                        [-0.004362, 0.000000, -0.324391, -12.906135], \
                        [0, 0, 0, 1]])
z1 = -12.85

transform_m2 = np.array([[-0.324265, 0.009865, 0.001872, -4.034072], \
                        [0.009865, 0.324270, -0.000057, -4.227960], \
                        [-0.001873, 0.000000, -0.324415, -8.932178], \
                        [0, 0, 0, 1]])
z2 = -8.98

transform_m3 = np.array([[-0.324419, 0.000000, 0.000710, -4.112169], \
                        [0.000000, 0.324420, 0.000000, -4.402324], \
                        [-0.000710, 0.000000, -0.324419, -5.046721], \
                        [0, 0, 0, 1]])
z3 = -5.06

transform_m4 = np.array([[-0.324410, 0.000000, 0.002592, -4.141242], \
                        [0.000000, 0.324420, 0.000000, -4.168504], \
                        [-0.002592, 0.000000, -0.324410, -1.295929], \
                        [0, 0, 0, 1]])
z4 = -1.48

#TODO: MapManager需要和多人结合起来 
mapManager = MapManager()
mapManager.AddFloorMap('library_map/B1L.geojson', transform_mB1.T, zB1, -1)
mapManager.AddFloorMap('library_map/1L.geojson', transform_m1.T, z1, 1)
mapManager.AddFloorMap('library_map/2L.geojson', transform_m2.T, z2, 2)
mapManager.AddFloorMap('library_map/3L.geojson', transform_m3.T, z3, 3)
mapManager.AddFloorMap('library_map/4L.geojson', transform_m4.T, z4, 4)

locateQueue = queue.Queue()
mainWork = MainWork()
packageWorker = Package()
userManager = UserManager()
datasets = ['0429-B1L', '0501-1L', '0501-2L', '0429-3L', '0429-4L']
floors = [-1, 1, 2, 3, 4]
t = threading.Thread(target=locateWorker)
t.start()

asyncio.run(serverStart())

