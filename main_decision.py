import time
import glob
import os.path
import sys

# tmp = ['/home/zz/carmaker_ros/ros/ros1_ws/devel/lib/python2.7/dist-packages',
#        '/opt/ros/ros1/lib/python2.7/dist-packages', '/opt/ros/kinetic/lib/python2.7/dist-packages',
#        '/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla-0.9.6-py3.5-linux-x86_64.egg',
#        '/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg_FILES']
# for index in range(0, len(tmp)):
#     if tmp[index] in sys.path:
#         sys.path.remove(tmp[index])

if not ('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg_FILES' in sys.path):
    sys.path.append('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg_FILES')

if not ('/home/zz/Downloads/CARLA_0.9.6/HDMaps' in sys.path):
    sys.path.append('/home/zz/Downloads/CARLA_0.9.6/HDMaps')
if not ('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla' in sys.path):
    sys.path.append('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla')


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import cv2
import carla
from sensor_class import *
from KeyboardShortCutSetting import *
import random
from HUD import HUD
# from model import *
# import MPCController
from Controller import *
from decision_trainer import *
import logging
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
# from agents.navigation.roaming_agent import RoamingAgent
# from agents.navigation.basic_agent import BasicAgent


try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

from pygame import gfxdraw

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue


class CarlaEnv():
    pygame.init()
    font = pygame.font.init()

    def __init__(self,world):
        #화면 크기
        n_iters = 100
        self.width = 800
        self.height = 600
        #센서 종류
        self.actor_list = []
        self.extra_list = []
        self.ROI_extra_list = []
        self.extra_controller_list = []
        self.extra_dl_list = []
        self.ROI_extra_dl_list = []
        self.player = None
        self.camera_rgb = None
        self.camera_semseg = None
        self.lane_invasion_sensor = None
        self.collision_sensor = None
        self.gnss_sensor = None
        self.waypoint = None
        self.path = []
        self.save_dir = None
        self.world = world
        self.map = world.get_map()
        self.spectator = self.world.get_spectator()
        self.hud = HUD(self.width, self.height)
        self.waypoints = self.map.generate_waypoints(3.0)
        self.lane_change_time = time.time()
        self.max_Lane_num = 3
        self.ego_Lane = 2
        self.agent = None
        self.controller = None
        self.is_first_time = True
        self.decision = None
        self.simul_time = time.time()
        self.accumulated_reward = 0
        self.end_point = 0
        self.ROI_length = 1000 #(meters)

        ## visualize all waypoints ##
        # for n, p in enumerate(self.waypoints):
        #     world.debug.draw_string(p.transform.location, 'o', draw_shadow=True,
        #                             color=carla.Color(r=255, g=255, b=255), life_time=999)

        settings = self.world.get_settings()
        # settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.02
        self.world.apply_settings(settings)
        self.extra_num = 0
        self.section = 0
        self.lane_distance_between_start =None
        self.lane_distance_between_end = None
        self.episode_start = None
        self.restart()
        self.main_test()

    def restart(self):
        self.decision = None
        self.simul_time = time.time()
        self.lane_change_time = time.time()
        self.max_Lane_num = 3
        self.ego_Lane = 2
        self.controller = None
        self.accumulated_reward = 0
        self.section = 0
        self.episode_start = time.time()
        print('destroying actors.')
        if len(self.actor_list) !=0:
            for actor in self.actor_list:
                # print("현재 actor 개수",len(self.actor_list))
                actor.destroy()
        if len(self.extra_list)!=0:
            for extra in self.extra_list:
                # print("현재 extra 개수",len(self.extra_list))
                extra.destroy()
        self.actor_list = []
        self.extra_list = []
        self.extra_controller_list = []
        self.extra_dl_list = []

        blueprint_library = self.world.get_blueprint_library()
        # start_pose = random.choice(self.map.get_spawn_points())
        start_pose = self.map.get_spawn_points()[102]

        ##spawn points 시뮬레이션 상 출력##
        # print(start_pose)
        # for n, x in enumerate(self.map.get_spawn_points()):
        #     world.debug.draw_string(x.location, 'o', draw_shadow=True,
        #                             color=carla.Color(r=0, g=255, b=255), life_time=30)

        # self.load_traj()

        self.waypoint = self.map.get_waypoint(start_pose.location,lane_type=carla.LaneType.Driving)
        self.end_point = self.waypoint.next(400)[0].transform.location
        # print(self.waypoint.transform)
        ## Ego vehicle의 global Route 출력##
        world.debug.draw_string(self.waypoint.next(400)[0].transform.location, 'o', draw_shadow=True,
                                color=carla.Color(r=0, g=255, b=255), life_time=100)


        # print(start_pose)
        # print(self.waypoint.transform)

        # self.controller = MPCController.Controller

        self.player = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.bmw.grandtourer')),
            start_pose)
        # print(self.player.bounding_box) # ego vehicle length

        self.actor_list.append(self.player)

        self.camera_rgb =RGBSensor(self.player, self.hud)
        self.actor_list.append(self.camera_rgb.sensor)

        # self.camera_depth =DepthCamera(self.player, self.hud)
        # self.actor_list.append(self.camera_depth.sensor)

        # self.camera_semseg = SegmentationCamera(self.player,self.hud)
        # self.actor_list.append(self.camera_semseg.sensor)

        self.collision_sensor = CollisionSensor(self.player, self.hud)  # 충돌 여부 판단하는 센서
        self.actor_list.append(self.collision_sensor.sensor)

        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)  # lane 침입 여부 확인하는 센서
        self.actor_list.append(self.lane_invasion_sensor.sensor)

        self.gnss_sensor = GnssSensor(self.player)
        self.actor_list.append(self.gnss_sensor.sensor)

        # --------------
        # Spawn Surrounding vehicles
        # --------------

        print("Generate Extra")
        spawn_points=[]
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        # print(*blueprints)
        for i in range(10, 10 * (self.extra_num) + 1, 10):
            dl=random.choice([-1,0,1])
            self.extra_dl_list.append(dl)
            spawn_point = None
            if dl==-1:
                spawn_point = self.waypoint.next(i)[0].get_left_lane().transform
            elif dl==0:
                spawn_point = self.waypoint.next(i)[0].transform
            elif dl==1:
                spawn_point = self.waypoint.next(i)[0].get_right_lane().transform
            else:
                print("Except ")
            spawn_point = carla.Transform((spawn_point.location + carla.Location(z=1)), spawn_point.rotation)
            spawn_points.append(spawn_point)
            # print(blueprint_library.filter('vehicle.bmw.grandtourer'))
            # blueprint = random.choice(blueprint_library.filter('vehicle.bmw.grandtourer'))


            blueprint = random.choice(blueprints)
            # print(blueprint.has_attribute('color'))
            if blueprint.has_attribute('color'):
                    # color = random.choice(blueprint.get_attribute('color').recommended_values)
                    # print(blueprint.get_attribute('color').recommended_values)
                    color = '255,255,255'
                    blueprint.set_attribute('color', color)
            extra = self.world.spawn_actor(blueprint,spawn_point)
            self.extra_list.append(extra)
        print('Extra Genration Finished')


        self.spectator.set_transform(carla.Transform(self.player.get_transform().location + carla.Location(z=100),
                            carla.Rotation(pitch=-90)))



        ROI=1000
        # extra_target_velocity = 10
        port = 8000
        traffic_manager = client.get_trafficmanager(port)
        traffic_manager.set_global_distance_to_leading_vehicle(100.0)
        tm_port = traffic_manager.get_port()
        for extra in self.extra_list:
            extra.set_autopilot(True,tm_port)
            # Pure_puresuit_controller(extra, self.waypoint, None, 50)  # km/h
            target_velocity = 30 #random.randrange(10, 40) # km/h
            extra.enable_constant_velocity(extra.get_transform().get_right_vector() * target_velocity/3.6)
            traffic_manager.auto_lane_change(extra,False)
        # self.player.set_autopilot(True,tm_port)
        # traffic_manager.auto_lane_change(self.player, False)

        self.controller = Pure_puresuit_controller(self.player, self.waypoint, self.extra_list, 80)  # km/h


        # target_velocity = 60 / 3.6
        # forward_vec = self.player.get_transform().get_forward_vector()
        # print(forward_vec)
        # velocity_vec =  target_velocity*forward_vec
        # self.player.set_target_velocity(velocity_vec)
        # print(velocity_vec)

        # print(velocity_vec)
        # client.get_trafficmanager.auto_lane_change(extra, False)
        ###Test####
        # clock = pygame.time.Clock()
        # Keyboardcontrol = KeyboardControl(self, False)
        # display = pygame.display.set_mode(
        #     (self.width, self.height),
        #     pygame.HWSURFACE | pygame.DOUBLEBUF)
        # while True:
        #     if Keyboardcontrol.parse_events(client, self, clock):
        #         return
        #     self.spectator.set_transform(
        #         carla.Transform(self.player.get_transform().location + carla.Location(z=50),
        #                         carla.Rotation(pitch=-90)))
        #     self.camera_rgb.render(display)
        #     self.hud.render(display)
        #     pygame.display.flip()
        #
        #     self.controller.apply_control()
        #     # self.world.wait_for_tick(10.0)
        #     clock.tick(30)
        #
        #     self.hud.tick(self, clock)

        #### Test Finished #####
        #### Test2 #####
        # cnt=0
        # clock = pygame.time.Clock()
        # while True:
        #     # print(self.waypoint.lane_id)
        #     self.spectator.set_transform(
        #         carla.Transform(self.player.get_transform().location + carla.Location(z=100),
        #                         carla.Rotation(pitch=-90)))
        #     cnt += 1
        #     if cnt == 100:
        #         print('수해유 ㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠ')
        #         decision = 1
        #         self.controller.apply_control(decision)
        #     else:
        #        self.controller.apply_control()
        #     clock.tick(30)
        #### Test2 Finished #####


        # self.input_size = (self.extra_num)*4 + 1
        self.input_size =  4 #dr dv da dl
        self.output_size = 3

        # for response in client.apply_batch_sync(batch):
        #     if response.error:
        #         logging.error(response.error)
        #     else:
        #         self.extra_list.append(response.actor_id)

    def draw_image(self, surface, image, blend=False):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if blend:
            image_surface.set_alpha(100)
        surface.blit(image_surface, (0, 0))

    def save_traj(self,current_location):
        f= open('route.txt','a')
        f.write(str(current_location.transform.location.x)+" ")
        f.write(str(current_location.transform.location.y)+" ")
        f.write(str(current_location.transform.location.z))
        f.write("\n")
        f.close()

    def load_traj(self):
        f= open("route.txt",'r')
        while True:
            line = f.readline()
            if not line: break
            _list = line.split(' ')
            _list = list(map(float,_list))
            point = carla.Transform()
            point.location.x = _list[0]
            point.location.y = _list[1]
            point.location.z = _list[2]
            self.path.append(point)
        #trajectory visualize
        # for n, x in enumerate(self.path):
        #     world.debug.draw_string(x.location, 'o', draw_shadow=True,
        #                             color=carla.Color(r=255, g=255, b=255), life_time=30)
        f.close()
    def get_gray_segimg(self,image_semseg):
        image_size = 96
        array = np.frombuffer(image_semseg.raw_data, dtype=np.dtype("uint8"))
        array = array.reshape((self.height, self.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)


        cv2.imshow("camera.semantic_segmentation", array)
        cv2.resizeWindow('Resized Window', 100, 100)
        cv2.waitKey(1)

        array = np.reshape([cv2.resize(array, (image_size, image_size))],
                           (1, image_size, image_size))  # (1, 96, 96)

        return array

    def step(self, action):
        SECONDS_PER_EPISODE = 1000
        plc = 0
        decision = None
        '''
        # Simple Action (action number: 3)
        action_test = action -1
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=action_test*self.STEER_AMT))
        '''

        # Complex Action
        if action == 0:  # LK
            decision = 0
        elif action == -1:  # left
            plc += 10
            decision = -1
        elif action == 1:  # right
            plc += 10
            decision = 1

        '''
        if len(self.collision_hist) != 0:
            done = True
            reward = -200 + (time.time()-self.episode_start)*15/SECONDS_PER_EPISODE
        else:
            done = False
            reward = 1
        '''
        end_length=math.sqrt((self.end_point.x - self.player.get_location().x)**2+(self.end_point.y - self.player.get_location().y)**2)

        done = False
        if len(self.collision_sensor.history) != 0:
            done = True
            reward = -100
        elif end_length < 15:
            done = True
            reward = 1
        else:
            reward = 1 - (self.controller.desired_vel-self.controller.velocity)/self.controller.desired_vel-plc

        self.accumulated_reward += reward

        if time.time() > self.episode_start + SECONDS_PER_EPISODE:
            done = True

        #state length = 4 * num of extra vehicles + 1

        state = self.get_next_state() #get now state

        # if len(state) == 29:
        #     print(" ")
        return state, decision ,reward, None , done
                                        # Next State 표현 필요
    def get_next_state(self,decision=None):
        """
        dl : relative lane num after ching the lane
        dr, dv, da : now state
        """


        state = []
        for x, actor in enumerate(self.ROI_extra_list):
            extra_pos = actor.get_transform().location
            extra_vel = actor.get_velocity()
            extra_acel = actor.get_acceleration()
            dr = ((extra_pos.x - self.player.get_transform().location.x) ** 2 + (
                        extra_pos.y - self.player.get_transform().location.y) ** 2 + (
                              (extra_pos.z - self.player.get_transform().location.z) ** 2) - (self.waypoint.lane_width*(self.ROI_extra_dl_list[x]))**2)**0.5
            dv = ((extra_vel.x - self.player.get_velocity().x) ** 2 + (
                    extra_vel.y - self.player.get_velocity().y) ** 2 + (
                          (extra_vel.z - self.player.get_transform().location.z) ** 2)) ** 0.5
            da = ((extra_acel.x - self.player.get_acceleration().x) ** 2 + (
                    extra_acel.y - self.player.get_acceleration().y) ** 2 +
                  (extra_acel.z - self.player.get_acceleration().z) ** 2) ** 0.5


            if decision == 1:
                self.extra_dl_list[x] =self.extra_dl_list[x]+1
                self.ROI_extra_dl_list[x] =self.ROI_extra_dl_list[x]+1

            elif decision == -1:
                self.extra_dl_list[x] = self.extra_dl_list[x]-1
                self.ROI_extra_dl_list[x] = self.ROI_extra_dl_list[x]-1

            else:
                pass


            state_dyn = [dr, dv, da, self.ROI_extra_dl_list[x]]
            state.append(state_dyn)
            # state.append(dr)

            # state.append(dv)
            # state.append(da)
            # state.append(self.ROI_extra_dl_list[x])


        x_static = []
        if decision == 1:
            self.ego_Lane +=1
        elif decision ==-1:
            self.ego_Lane +=-1
        else:
            pass

        x_static.append(self.ego_Lane, self.controller.velocity, self.search_distance_vaild()/self.ROI_length )




        # state.append(self.ego_Lane)
        return state

    def search_distance_vaild(self):
        index= self.max_Lane_num-self.ego_Lane
        last_lane_waypoint = self.self.controller.waypoint
        allowable_error = 3
        distance = 9999
        i=1
        if self.max_Lane_num ==4:
            while index>0: #get waypoint of forth's lane
                last_lane_waypoint = last_lane_waypoint.get_right_lane()
                index -= 1
            while allowable_error >= distance:
                distance = ((last_lane_waypoint.next(i)[0].x-self.lane_start_point.x)**2+(last_lane_waypoint.next(i)[0].y-self.lane_start_point.y)**2+(last_lane_waypoint.next(i)[0].z-self.lane_start_point.z)**2)**0.5
                i +=3
            return min(distance, self.ROI_length)
        elif self.max_Lane_num ==3:
            while index>1: #get waypoint of third's lane
                last_lane_waypoint = last_lane_waypoint.get_right_lane()
                index -= 1
            while allowable_error >= distance:
                distance = ((last_lane_waypoint.next(i)[0].x-self.lane_finished_point.x)**2+(last_lane_waypoint.next(i)[0].y-self.lane_finished_point.y)**2+(last_lane_waypoint.next(i)[0].z-self.lane_finished_point.z)**2)**0.5
            return min()


    def safety_check(self,decision, safe_lane_change_again_time=4):
        if decision !=0 and decision !=None:
            if (time.time()-self.lane_change_time) <= safe_lane_change_again_time:
                return 0 #즉 직진
            elif self.agent.selection_method == 'random' and decision == 1 and self.ego_Lane ==self.max_Lane_num:
                action = random.randrange(-1,1)
                return action
            elif self.agent.selection_method == 'random' and decision == -1 and self.ego_Lane ==1:
                action = random.randrange(0, 2)
                return action
            elif self.ego_Lane ==self.max_Lane_num and decision ==1:
                return int(self.agent.q_value[0:1].max().item())-1
            elif self.ego_Lane ==1 and decision ==-1:
                return int(self.agent.q_value[1:2].max().item())-1
            elif decision ==1 and self.ego_Lane != self.max_Lane_num:
                self.lane_change_time = time.time()
                return decision
            elif decision ==-1 and self.ego_Lane != 1:
                self.lane_change_time = time.time()
                return decision
            else: #3차선에서 우 차도 판단, 1차선에서 좌차선 판단
                if self.agent.selection_method == 'random' and decision == 1 and self.ego_Lane == self.max_Lane_num:
                    action = random.randrange(-1, 1)
                    return action
                elif self.agent.selection_method == 'random' and decision == -1 and self.ego_Lane == 1:
                    action = random.randrange(0, 2)
                    return action
                elif self.ego_Lane == self.max_Lane_num and decision == 1:
                    return int(self.agent.q_value[0:1].max().item()) - 1
                elif self.ego_Lane == 1 and decision == -1:
                    return int(self.agent.q_value[1:2].max().item()) - 1
        else:
            return decision
    # def safety_check_between_vehicles(self):
    #     if extra_lists

    def main_test(self):
        PATH = "/home/a/RL_decision/trained_info.tar"
        print(torch.cuda.get_device_name())
        clock = pygame.time.Clock()
        Keyboardcontrol = KeyboardControl(self, False)
        display = pygame.display.set_mode(
            (self.width, self.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.lane_start_point = [carla.Location(x=14.905815, y=-135.747452, z=0.000000),carla.Location(x=172.745468, y=-364.531799, z=0.000000) , carla.Location(x=376.542816, y=-10.352980, z=0.000000)]
        self.lane_finished_point = [carla.Location(x=14.631096, y=-205.746918, z=0.000000),carla.Location(x=232.962860, y=-364.149139, z=0.000000) , carla.Location(x=382.441040, y=-212.488907, z=0.000000)]


        self.world.debug.draw_string(carla.Location(x=14.905815, y=-135.747452, z=0.000000),
                                     'o', draw_shadow=True,
                                     color=carla.Color(r=0, g=255, b=0), life_time=9999)
        self.world.debug.draw_string(carla.Location(x=14.631096, y=-205.746918, z=0.000000),
                                     'o', draw_shadow=True,
                                     color=carla.Color(r=0, g=255, b=0), life_time=9999)
        # ---------------------------#

        self.world.debug.draw_string(carla.Location(x=172.745468, y=-364.531799, z=0.000000),
                                     'o', draw_shadow=True,
                                     color=carla.Color(r=0, g=255, b=0), life_time=9999)
        self.world.debug.draw_string(carla.Location(x=232.962860, y=-364.149139, z=0.000000),
                                     'o', draw_shadow=True,
                                     color=carla.Color(r=0, g=255, b=0), life_time=9999)
        # ---------------------------#
        self.world.debug.draw_string(carla.Location(x=376.542816, y=-10.352980, z=0.000000),
                                     'o', draw_shadow=True,
                                     color=carla.Color(r=0, g=255, b=0), life_time=9999)
        self.world.debug.draw_string(carla.Location(x=382.441040, y=-212.488907, z=0.000000),
                                     'o', draw_shadow=True,
                                     color=carla.Color(r=0, g=255, b=0), life_time=9999)
        lane_distance_between_points = []
        for i in range(len(self.lane_finished_point)):
            lane_distance_between_points.append(((self.lane_start_point[i].x - self.lane_finished_point[i].x) ** 2 + (
                        self.lane_start_point[i].y - self.lane_finished_point[i].y) ** 2 + (
                                                            self.lane_start_point[i].z - self.lane_finished_point[i].z) ** 2)**0.5)



        while True:
            if Keyboardcontrol.parse_events(client, self, clock):
                return
            self.spectator.set_transform(
                carla.Transform(self.player.get_transform().location + carla.Location(z=50),
                                carla.Rotation(pitch=-90)))
            self.camera_rgb.render(display)
            self.hud.render(display)
            pygame.display.flip()
            # self.world.debug.draw_string(self.player.get_transform().location,
            #                              'o', draw_shadow=True,
            #                              color=carla.Color(r=255, g=255, b=0), life_time=9999)
            ## Get max lane ##
            print("start get lane")
            if self.section < len(lane_distance_between_points):
                self.lane_distance_between_start = (
                        (self.player.get_transform().location.x - self.lane_start_point[self.section].x) ** 2 +
                        (self.player.get_transform().location.y - self.lane_start_point[self.section].y) ** 2)**0.5
                self.lane_distance_between_end = (
                        (self.player.get_transform().location.x - self.lane_finished_point[self.section].x) ** 2 +
                        (self.player.get_transform().location.y - self.lane_finished_point[self.section].y) ** 2)**0.5

                # print("self.lane_distance_between_start : ",self.lane_distance_between_start,"self.lane_distance_between_end :",self.lane_distance_between_end, "lane_distance_between_points[section]",lane_distance_between_points[self.section],"section :", self.section)
                if max(lane_distance_between_points[self.section], self.lane_distance_between_start, self.lane_distance_between_end) == \
                        lane_distance_between_points[self.section]:
                    self.max_Lane_num = 3
                    print("hi1")
                    # world.debug.draw_string(self.player.get_transform().location, 'o', draw_shadow = True,
                    #                                 color = carla.Color(r=255, g=255, b=0), life_time = 999)

                elif max(lane_distance_between_points[self.section], self.lane_distance_between_start, self.lane_distance_between_end) == \
                        self.lane_distance_between_start and self.max_Lane_num ==3:
                    self.section+=1
                    self.max_Lane_num=4
                    print("hi2")
                    if self.section >= len(lane_distance_between_points):  # when, section_num = 3
                        self.section = 0
            ## finished get max lane ##
            print("finished get lane")


            self.controller.apply_control(self.decision)
            # self.world.wait_for_tick(10.0)
            clock.tick(30)

            self.hud.tick(self, clock)

    def main(self):
        PATH = "/home/a/RL_decision/trained_info.tar"
        print(torch.cuda.get_device_name())
        clock = pygame.time.Clock()
        Keyboardcontrol = KeyboardControl(self, False)
        display = pygame.display.set_mode(
            (self.width, self.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.lane_start_point = [carla.Location(x=14.905815, y=-135.747452, z=0.000000),
                            carla.Location(x=172.745468, y=-364.531799, z=0.000000),
                            carla.Location(x=376.542816, y=-10.352980, z=0.000000)]
        self.lane_finished_point = [carla.Location(x=14.631096, y=-205.746918, z=0.000000),
                               carla.Location(x=232.962860, y=-364.149139, z=0.000000),
                               carla.Location(x=382.441040, y=-212.488907, z=0.000000)]

        self.world.debug.draw_string(carla.Location(x=14.905815, y=-135.747452, z=0.000000),
                                     'o', draw_shadow=True,
                                     color=carla.Color(r=0, g=255, b=0), life_time=9999)
        self.world.debug.draw_string(carla.Location(x=14.631096, y=-205.746918, z=0.000000),
                                     'o', draw_shadow=True,
                                     color=carla.Color(r=0, g=255, b=0), life_time=9999)
        # ---------------------------#

        self.world.debug.draw_string(carla.Location(x=172.745468, y=-364.531799, z=0.000000),
                                     'o', draw_shadow=True,
                                     color=carla.Color(r=0, g=255, b=0), life_time=9999)
        self.world.debug.draw_string(carla.Location(x=232.962860, y=-364.149139, z=0.000000),
                                     'o', draw_shadow=True,
                                     color=carla.Color(r=0, g=255, b=0), life_time=9999)
        # ---------------------------#
        self.world.debug.draw_string(carla.Location(x=376.542816, y=-10.352980, z=0.000000),
                                     'o', draw_shadow=True,
                                     color=carla.Color(r=0, g=255, b=0), life_time=9999)
        self.world.debug.draw_string(carla.Location(x=382.441040, y=-212.488907, z=0.000000),
                                     'o', draw_shadow=True,
                                     color=carla.Color(r=0, g=255, b=0), life_time=9999)
        lane_distance_between_points = []
        for i in range(len(self.lane_finished_point)):
            lane_distance_between_points.append((self.lane_start_point[i].x - self.lane_finished_point[i].x) ** 2 + (
                    self.lane_start_point[i].y - self.lane_finished_point[i].y) ** 2 + (
                                                        self.lane_start_point[i].z - self.lane_finished_point[i].z) ** 2)


        # if(os.path.exists(PATH)):
        #     print("저장된 가중치 불러옴")
        #     torch.load(PATH)
        #     self.agent.model.load_state_dict(self.save_dir['model'])
        #     device = torch.device('cuda')
        #     self.agent.model.to(device)
        #     self.agent.optimizer.load_state_dict(self.save_dir['optimizer'])
            # self.agent.buffer.
            # 'memorybuffer': self.agent.buffer

            # self.is_first_time = False
        # controller = VehicleControl(self, True)
        # self.player.apply_control(carla.Vehicle
        # Control(throttle=1.0, steer=0.0, brake=0.0))
        # vehicles = self.world.get_actors().filter('vehicle.*')
        # for i in vehicles:
        #     print(i.id)
        try:
            n_iters = 10000
            is_error = 0

            for epoch in range(n_iters):
                # cnt = 0
                first_time_print = True

                state = self.get_next_state() #초기 상태 s0 초기화
                while(True):
                    if Keyboardcontrol.parse_events(client, self, clock):
                        return

                    self.camera_rgb.render(display)
                    self.hud.render(display)
                    pygame.display.flip()

                    self.spectator.set_transform(
                        carla.Transform(self.player.get_transform().location + carla.Location(z=100),
                                        carla.Rotation(pitch=-90)))

                    ## Get max lane ##
                    if self.section < len(lane_distance_between_points):
                        self.lane_distance_between_start = (
                                (self.player.get_transform().location.x - self.lane_start_point[self.section].x) ** 2 +
                                (self.player.get_transform().location.y - self.lane_start_point[self.section].y) ** 2)
                        self.lane_distance_between_end = (
                                (self.player.get_transform().location.x - self.lane_finished_point[self.ection].x) ** 2 +
                                (self.player.get_transform().location.y - self.lane_finished_point[self.section].y) ** 2)

                        print("self.lane_distance_between_start : ",self.lane_distance_between_start,"self.lane_distance_between_end :",self.lane_distance_between_end, "lane_distance_between_points[self.section]",lane_distance_between_points[self.section],"self.section :", self.section)
                        if max(lane_distance_between_points[self.section], self.lane_distance_between_start,
                               self.lane_distance_between_end) == \
                                lane_distance_between_points[self.section]:
                            self.max_Lane_num = 3
                            # world.debug.draw_string(self.player.get_transform().location, 'o', draw_shadow = True,
                            #                                 color = carla.Color(r=255, g=255, b=0), life_time = 999)

                        elif max(lane_distance_between_points[self.section], self.lane_distance_between_start,
                                 self.lane_distance_between_end) == \
                                self.lane_distance_between_start and self.max_Lane_num == 3:
                            self.section += 1
                            self.max_Lane_num = 4
                            if self.section >= len(lane_distance_between_points): # when, section_num = 3
                                self.section = 0
                    ## finished get max lane ##

                    [self.ROI_extra_list, self.ROI_extra_dl_list] = self.agent.search_extra_in_ROI(self.extra_lists,
                                                                                                   self.player,
                                                                                                   self.extra_dl_list)

                    if self.agent.is_training:
                        ##dqn 과정##
                        # 가중치 초기화 (pytroch 내부)
                        # 입실론-그리디 행동 탐색 (act function)
                        # 메모리 버퍼에 MDP 튜플 얻기   ㅡ (step function)
                        # 메모리 버퍼에 MDP 튜플 저장   ㅡ
                        # optimal Q 추정             ㅡ   (learning function)
                        # Loss 계산                  ㅡ
                        # 가중치 업데이트              ㅡ

                        if epoch % 10 == 0:
                            # [w, b] = self.agent.model.parameters()  # unpack parameters
                            self.save_dir=torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.agent.model.state_dict(),
                                'optimizer_state_dict': self.agent.optimizer.state_dict(),
                                'memorybuffer': self.agent.buffer}, PATH)


                        if self.decision is not None:
                            next_state = self.get_next_state(self.decision)


                            sample = [state, self.decision, reward, next_state, done]
                            self.agent.buffer.append(sample)


                        self.decision = self.agent.act(state)


                        # print(decision)
                        # if self.decision ==1 and self.max_Lane_num==self.ego_Lane:
                        #     print( " ")
                        self.decision = self.safety_check(self.decision)
                        # print("판단 :", self.decision, "차선 :", self.ego_Lane, "최대 차선 :", self.max_Lane_num)

                        is_error = self.controller.apply_control(self.decision)

                        # print("extra_controller 개수 :", len(self.extra_controller_list))
                        # for i in range(len(self.extra_controller_list)):
                        #     self.extra_controller_list[i].apply_control()



                        [state, decision, reward, _, done] =  self.step(self.decision)

                        if done:
                            print("epsilon :",self.agent.epsilon)
                            print("epoch : ", epoch , "누적 보상 : ",self.accumulated_reward)
                            writer.add_scalar('누적 보상 ',self.accumulated_reward,epoch)

                            if len(self.agent.buffer.size())>self.agent.batch_size:

                                for i in range(300):
                                    self.agent.learning()

                            client.set_timeout(10)
                            self.restart()
                            break

                    else:
                        self.agent.act(state)

                    clock.tick(30)

        finally:
            print('\ndestroying %d vehicles' % len(self.extra_list))
            # client.apply_batch([carla.command.DestroyActor(x) for x in self.extra_list])

            print('destroying actors.')
            for actor in self.actor_list:
                # print("finally 에서 actor 제거 :", self.actor_list)
                actor.destroy()
            for extra in self.extra_list:
                # print("finally 에서 actor 제거 :", self.extra_list)
                extra.destroy()

            # pygame.quit()
            print('done.')

if __name__ == '__main__':
    try:
        client = carla.Client('localhost', 2000)
        # client.set_timeout(4.0)
        client.load_world('Town04')
        # client.set_timeout(10.0)

        weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            sun_altitude_angle=5.0)


        world = client.get_world()
        world.set_weather(weather)



        CarlaEnv(world)


    except:
        pass




    # except KeyboardInterrupt:
        # print('\nCancelled by user. Bye!')
