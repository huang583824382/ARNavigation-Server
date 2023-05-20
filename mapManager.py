import json
import math
from typing import Dict, List, Union, Optional
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
from UserManager import UserInstance, UserLocStatus
import heapq

def latlng_to_xy(lat, lng):
    R = 6371000 # 地球半径，以米为单位
    x = R * math.radians(lng)
    y = R * math.log(math.tan(math.pi/4 + math.radians(lat)/2))
    return (x, y)

def distance_xy(p1, p2):
    x = abs(p1[0]-p2[0])
    y = abs(p1[1]-p2[1])
    return (x**2+y**2)**0.5

def cv2ImgAddText(img, text, pos, textColor=(0, 255, 0), textSize=20):
    left = pos[0]
    top = pos[1]
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(fm.findfont(fm.FontProperties(fname=r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")), textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


class FloorMapManager:
    def __init__(self, f_geojson: Union[Path, str], transform_m, z_coor, floor) -> None:
        self.min_x = 10e9
        self.min_y = 10e9
        self.max_x = -10e9
        self.max_y = -10e9
        self.coor2idx = {}
        self.idx2coor = {}
        self.adjacency_m = {}
        self.place2idx = {}
        self.idx2place = {}
        self.staircase = []
        self.lift = []

        # print('init floor map manager', len(self.place2idx))
        self.floor = floor
        self.z_coor = z_coor
        if isinstance(f_geojson, str):
            f_geojson = Path(f_geojson)
        if f_geojson.exists() is False:
            print('path not exist')
            return
        with open(str(f_geojson), 'r', encoding='utf8') as fp:
            data = fp.read()
            fp.close()

        obj = json.loads(data)
        # print(obj.keys())
        features = obj['features']
        
        for f in features:
            geometry = f['geometry']
            if geometry['type'] == 'Point':
                coordinate = geometry['coordinates']
                xycoor = latlng_to_xy(coordinate[0], coordinate[1])
                # print(xycoor)
                self.min_x = min(self.min_x, xycoor[0])
                self.max_x = max(self.max_x, xycoor[0])
                self.min_y = min(self.min_y, xycoor[1])
                self.max_y = max(self.max_y, xycoor[1])
                geometry['coordinates'] = xycoor
            if geometry['type'] == 'LineString':
                coordinates = geometry['coordinates']
                for i in range(len(coordinates)):
                    xycoor = latlng_to_xy(coordinates[i][0], coordinates[i][1])
                    self.min_x = min(self.min_x, xycoor[0])
                    self.max_x = max(self.max_x, xycoor[0])
                    self.min_y = min(self.min_y, xycoor[1])
                    self.max_y = max(self.max_y, xycoor[1])
                    coordinates[i] = xycoor
                # print(geometry['coordinates'])
        # print('self.min_x, self.max_x, self.min_y, self.max_y')
        # print(self.min_x, self.max_x, self.min_y, self.max_y)
        for f in features:
            geometry = f['geometry']
            if geometry['type'] == 'Point':
                # transform
                if transform_m.shape[0] == 3:
                    coor_q = np.array([geometry['coordinates'][0], geometry['coordinates'][1], 1])
                else:
                    coor_q = np.array([geometry['coordinates'][0], geometry['coordinates'][1], 0, 1])
                # coor_res = np.matmul(transform_m, coor_q.transpose())
                coor_res = coor_q.dot(transform_m)
                coordinate = (coor_res[0], coor_res[1])
                geometry['coordinates'] = coordinate
                if coordinate not in self.coor2idx.keys():
                    self.coor2idx[coordinate] = len(self.coor2idx)
            if geometry['type'] == 'LineString':
                coordinates = geometry['coordinates']
                for i in range(len(coordinates)):
                    # transform
                    if transform_m.shape[0] == 3:
                        coor_q = np.array([coordinates[i][0], coordinates[i][1], 1])
                    else:
                        coor_q = np.array([coordinates[i][0], coordinates[i][1], 0, 1])
                    # coor_res = np.matmul(transform_m, coor_q.transpose())
                    coor_res = coor_q.dot(transform_m)
                    coordinate = (coor_res[0], coor_res[1])
                    coordinates[i] = coordinate
                    if coordinate not in self.coor2idx.keys():
                        self.coor2idx[coordinate] = len(self.coor2idx)
        # print(coor2idx)
        for key, val in self.coor2idx.items():
            self.idx2coor[val] = key
        # print(idx2coor)

        for f in features:
            geometry = f['geometry']
            if geometry['type'] == 'Point':
                geometry['coordinates'] = self.coor2idx[geometry['coordinates']]
            if geometry['type'] == 'LineString':
                coordinates = geometry['coordinates']
                for i in range(len(coordinates)):
                    coordinates[i] = self.coor2idx[coordinates[i]]

        self.adjacency_m = [[-1 for i in range(len(self.coor2idx))] for i in range(len(self.coor2idx))]
        
        
        for f in features:
            geometry = f['geometry']
            if geometry['type'] == 'Point':
                property_name = list(f['properties'])[0]
                self.idx2place[geometry['coordinates']] = list(f['properties'])[0]
                self.place2idx[list(f['properties'])[0]] = geometry['coordinates']

                if f['properties'][property_name] == '1':
                    self.lift.append(property_name)
                elif f['properties'][property_name] == '2':
                    self.staircase.append(property_name)

            if geometry['type'] == 'LineString':
                coordinates = geometry['coordinates']
                for i in range(len(coordinates)-1):
                    s = coordinates[i]
                    e = coordinates[i+1]
                    # print(s, e)
                    distance = distance_xy(self.idx2coor[s], self.idx2coor[e])
                    self.adjacency_m[s][e] = distance
                    self.adjacency_m[e][s] = distance
        # print(self.adjacency_m)
        print('Floor Map load success')
        print(self.place2idx.keys())

    def GetNearestNodeIdx(self, userPosition)->int:
        res = 0
        mindis = 10e9
        # print(len(self.coor2idx))
        for coor, idx in self.coor2idx.items():
            tmp = distance_xy(userPosition, coor)
            # print('dis = ',tmp, idx)
            if(tmp<mindis):
                mindis = tmp
                res = idx
        print('nearest node idx = ', res, self.idx2place.get(res, 'None'))
        return res

    def store_places(self)->str:
        places = list()
        for idx, place in self.idx2place.items():
            places.append(place)
        print('Store places to json')
        res = dict()
        res['places'] = places
        with open('places.json', 'w', encoding="utf8") as fp:
            json.dump(res, fp, ensure_ascii=False)
        return 'places.json'

    def FindPath(self, user: UserInstance, e: int)->list:
        if user.locStatus == UserLocStatus.Uninitialized:
            return []
        s = int(self.GetNearestNodeIdx(user.GetPosition()))
        path = self.dijkstra(s, e)
        return path
    
    def draw_path(self, path: list, img: cv2.Mat = None):
        color_white = (255, 255, 255)
        color_black = (0, 0, 0)
        color_green = (0, 255, 0)
        self.max_x = -10e9
        self.max_y = -10e9
        self.min_x = 10e9
        self.min_y = 10e9
        for idx, coor in self.idx2coor.items():
            self.min_x = min(self.min_x, coor[0])
            self.max_x = max(self.max_x, coor[0])
            self.min_y = min(self.min_y, coor[1])
            self.max_y = max(self.max_y, coor[1])

        print('self.min_x, self.max_x, self.min_y, self.max_y')
        print(self.min_x, self.max_x, self.min_y, self.max_y)
        width = self.max_x-self.min_x
        height = self.max_y-self.min_y

        sperm = 100
        mat_w = int(width*2*sperm)
        mat_h = int(height*2*sperm)
        if img is None:
            img = np.zeros((mat_w, mat_h, 3), np.uint8)
            img.fill(255)

        idx2imgcoor = dict()
        for (key, val) in self.idx2coor.items():
            center_y = int((val[0]-self.min_x)*sperm+mat_w/4)
            center_x = int((val[1]-self.min_y)*sperm+mat_h/4)
            idx2imgcoor[key] = (center_x, center_y)

        for i in range(len(path)-1):
            s = path[i]
            e = path[i+1]
            cv2.line(img, idx2imgcoor[s], idx2imgcoor[e], color_green, 2)
        return img
    
    
    def DrawMap(self):
        color_white = (255, 255, 255)
        color_red = (255, 0, 0)
        color_green = (0, 255, 0)
        color_black = (0, 0, 0)
        self.max_x = -10e9
        self.max_y = -10e9
        self.min_x = 10e9
        self.min_y = 10e9
        for idx, coor in self.idx2coor.items():
            
            self.min_x = min(self.min_x, coor[0])
            self.max_x = max(self.max_x, coor[0])
            self.min_y = min(self.min_y, coor[1])
            self.max_y = max(self.max_y, coor[1])

        width = self.max_x-self.min_x
        height = self.max_y-self.min_y

        sperm = 5
        mat_w = int(width*2*sperm)
        mat_h = int(height*2*sperm)

        img = np.zeros((mat_w, mat_h, 3), np.uint8)
        img.fill(255)
        idx2imgcoor = dict()
        for (key, val) in self.idx2coor.items():
            print(key, val)
            center_y = int((val[0]-self.min_x)*sperm+mat_w/4)
            center_x = int((val[1]-self.min_y)*sperm+mat_h/4)
            idx2imgcoor[key] = (center_x, center_y)
            cv2.circle(img, idx2imgcoor[key], 2, color_black)
            img = cv2ImgAddText(img, str(MapManager.idx2idx_all[(key, self.floor)]), (idx2imgcoor[key][0]-5, idx2imgcoor[key][1]+5), color_black, 10)
        for i in range(len(self.idx2coor)):
            for j in range(i):
                if self.adjacency_m[i][j]>0:
                    cv2.line(img, idx2imgcoor[i], idx2imgcoor[j], color_black, 1)
        
        for (key, val) in self.idx2place.items():
            img = cv2ImgAddText(img, str(val), (idx2imgcoor[key][0]+5, idx2imgcoor[key][1]-20), color_black, 20)
        
        
        center_y = int((0-self.min_x)*sperm+mat_w/4)
        center_x = int((0-self.min_y)*sperm+mat_h/4)
        cv2.circle(img, (center_x, center_y), 3, color_red, thickness=5)
        cv2.line(img, (center_x, center_y), (center_x, center_y+50), color_green, 3)
        cv2.line(img, (center_x, center_y), (center_x+50, center_y), color_red, 3)
        return img
    
    def test(self):
        print(self.idx2place)
        print('staircase', self.staircase)
        print('lift', self.lift)

class MapManager:
    floorMaps: Dict[int, FloorMapManager] = {} # floor -> FloorMapManager
    # floor2places = {}
    idx2idx_all = {} # (idx, floor) -> idx_all
    places2idx_all = {} # (place, floor) -> idx_all
    adj_matrix = None
    staircases_idx = set()
    lifts_idx = set()

    def Clear(self):
        MapManager.floorMaps = {}
        MapManager.idx2idx_all = {}
        MapManager.places2idx_all = {}
        MapManager.adj_matrix = None
        MapManager.staircases_idx = set()
        MapManager.lifts_idx = set()

    def AddFloorMap(self, f_geojson: Union[Path, str], transform_m, z_coor, floor):
        floorMap = FloorMapManager(f_geojson, transform_m, z_coor, floor)
        self.floorMaps[floor] = floorMap
        # places = []
        for idx in floorMap.idx2coor.keys():
            # places.append(place)
            self.idx2idx_all[(idx, floor)] = len(self.idx2idx_all)

        for place in floorMap.place2idx.keys():
            self.places2idx_all[(place, floor)] = self.idx2idx_all[(floorMap.place2idx[place], floor)]
        # self.floor2places[floor] = places
        for staircase in floorMap.staircase:
            self.staircases_idx.add(self.idx2idx_all[(floorMap.place2idx[staircase], floor)])

        for lift in floorMap.lift:
            self.lifts_idx.add(self.idx2idx_all[(floorMap.place2idx[lift], floor)])

    def GenerateAdjMatrix(self, mode = 0):
        # mode = 0: use lift and staircase
        # mode = 1: use only staircase
        # mode = 2: use only lift

        self.adj_matrix = [[-1 for i in range(len(self.idx2idx_all))] for i in range(len(self.idx2idx_all))]
        self.adj_graph = {}
        floorMaps_sorted = {k: self.floorMaps[k] for k in sorted(self.floorMaps)}
        floors = list(floorMaps_sorted.keys())
        # print(floors)

        lastFloor = floors[0]
        for floor, map in floorMaps_sorted.items():
            for i in range(len(map.idx2coor)):
                neighbors = {}
                for j in range(len(map.idx2coor)):
                    if map.adjacency_m[i][j]>0:
                        # print(i, j)
                        neighbors[self.idx2idx_all[(j, floor)]] = map.adjacency_m[i][j]
                        self.adj_matrix[self.idx2idx_all[(i, floor)]][self.idx2idx_all[(j, floor)]] = map.adjacency_m[i][j]
                        self.adj_matrix[self.idx2idx_all[(j, floor)]][self.idx2idx_all[(i, floor)]] = map.adjacency_m[i][j]
                self.adj_graph[self.idx2idx_all[(i, floor)]] = neighbors
            if floor == lastFloor:
                continue

            if mode == 0 or mode == 1:
                for stair in self.floorMaps[floor].staircase:
                    if stair in self.floorMaps[lastFloor].staircase:
                        # 楼梯对应
                        idx1 = self.idx2idx_all[(self.floorMaps[floor].place2idx[stair], floor)]
                        idx2 = self.idx2idx_all[(self.floorMaps[lastFloor].place2idx[stair], lastFloor)]
                        self.adj_matrix[idx1][idx2] = 10
                        self.adj_matrix[idx2][idx1] = 10
                        self.adj_graph[idx1][idx2] = 10
                        self.adj_graph[idx2][idx1] = 10
            
            if mode == 0 or mode == 2:
                for lift in self.floorMaps[floor].lift:
                    for tmpfloor in floors:
                        if(tmpfloor == floor):
                            break
                        if lift in self.floorMaps[lastFloor].lift:
                            # 电梯对应
                            idx1 = self.idx2idx_all[(self.floorMaps[floor].place2idx[lift], floor)]
                            idx2 = self.idx2idx_all[(self.floorMaps[tmpfloor].place2idx[lift], tmpfloor)]
                            self.adj_matrix[idx1][idx2] = 20
                            self.adj_matrix[idx2][idx1] = 20
                            self.adj_graph[idx1][idx2] = 20
                            self.adj_graph[idx2][idx1] = 20
            lastFloor = floor

    def dijkstra(self, start, end):
        graph = self.adj_graph
        # print(graph)
        # graph = adjacency_matrix_to_list(self.adj_matrix)
        print('dijkstra', start, end)
        # print(graph)
        if start == end:
            return [end]
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        heap = [(0, start)]
        previous_nodes = {node: None for node in graph}
        visited = set()
        while heap:
            current_distance, current_node = heapq.heappop(heap)
            if current_node == end:
                path = []
                while previous_nodes[current_node] is not None:
                    path.append(current_node)
                    current_node = previous_nodes[current_node]
                path.append(start)
                path.reverse()
                print('distance:', distances[end])
                return path
            visited.add(current_node)
            for neighbor, weight in graph[current_node].items():
                if neighbor in visited:
                    continue
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(heap, (distance, neighbor))
        return None
    
    def FindPath(self, userInstance: UserInstance, src_floor, des: str, des_floor, mode = 0):
        local_idx = self.floorMaps[src_floor].GetNearestNodeIdx(userInstance.GetPosition())
        global_start_idx = self.idx2idx_all[(local_idx, src_floor)]
        global_end_idx = self.idx2idx_all[(self.floorMaps[des_floor].place2idx[des], des_floor)]
        self.GenerateAdjMatrix(mode)
        path = self.dijkstra(global_start_idx, global_end_idx)
        return path

    def WritePath2JSON(self, path: list, jsonPath: str):
        d = dict()
        localIdxFloor_pairList = []
        for global_idx in path:
            localIdxFloor_pairList.append(self.Idx_all2localIdxFloor_pair(global_idx))
        d['Src'] = self.floorMaps[localIdxFloor_pairList[0][1]].idx2place.get(localIdxFloor_pairList[0][0], 'None')
        d['Des'] = self.floorMaps[localIdxFloor_pairList[-1][1]].idx2place.get(localIdxFloor_pairList[-1][0], 'None')
        d['Path'] = path
        # d['PathCoor'] = [[self.idx2coor[t][0], self.idx2coor[t][1], self.z_coor] for t in path]
        d['PathCoor'] = [[self.floorMaps[pair[1]].idx2coor[pair[0]][0], self.floorMaps[pair[1]].idx2coor[pair[0]][1], self.floorMaps[pair[1]].z_coor] for pair in localIdxFloor_pairList]
        d['Floor'] = [pair[1] for pair in localIdxFloor_pairList]
        types = []
        for i in path:
            if i in self.staircases_idx:
                types.append(2)
            elif i in self.lifts_idx:
                types.append(1)
            else:
                types.append(0)
        d['Type'] = types

        # startp = user.GetPosition()
        # startp[2] = self.z_coor
        # d['PathCoor'].insert(0, startp)
        fp = open(jsonPath, 'w', encoding="utf8")
        json.dump(d, fp, ensure_ascii=False)
        return d

    def Idx_all2localIdxFloor_pair(self, idx_all: int):
        localIdxFloor_pair = (0, 0)
        for pair, idx in self.idx2idx_all.items():
            if idx == idx_all:
                return pair # (localIdx, floor)
            
    def StorePlaces(self):
        places_all = dict()
        for (floor, floorMap) in self.floorMaps.items():
            res_floor = []
            for (place, idx) in floorMap.place2idx.items():
                res_floor.append(place)
            res_floor.sort()
            places_all[floor] = res_floor
        res = dict()
        res['places'] = places_all
        with open('places.json', 'w', encoding = 'utf8') as f:
            json.dump(res, f, ensure_ascii=False)
            f.close()
        return 'places.json'

    def DrawMap(self):
        # fig, axs = plt.subplots(1, len(self.floorMaps), figsize=(20, 10))
        index = 0
        for floor, floorMap in self.floorMaps.items():
            img = floorMap.DrawMap()
            # axs[index].imshow(img)
            index += 1
            print(img.shape)
            img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_cv = cv2.resize(img_cv, (2000, 1000))
            cv2.imshow(str(index), img_cv)
        # plt.show()
        cv2.waitKey(0)

    def test(self):
        for floor, floorMap in self.floorMaps.items():
            print('floor', floor)
            floorMap.test()
        # print('places2idx_all', self.places2idx_all)
        # print(self.places2idx_all[('男卫生间', 1)])
        self.GenerateAdjMatrix()
        # print(self.adj_matrix)
        # path = self.dijkstra(137, 72)
        # print('path res', path)
        # self.WritePath2JSON(path, 'testMultiFloor.json')
        # self.DrawMap()
