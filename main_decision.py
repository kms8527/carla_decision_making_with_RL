import time
import glob
import os
import sys


# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass
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
import torch
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
        self.start_epoch = True
        self.input_size = 4  # dr dv da dl
        self.output_size = 3
        n_iters = 100
        self.width = 800
        self.height = 600
        #센서 종류
        self.actor_list = []
        self.extra_list = []
        self.extra_list = []
        self.extra_controller_list = []
        self.extra_dl_list = []
        self.extra_dl_list = []
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
        self.max_Lane_num = 50
        self.ego_Lane = 2
        self.agent = None
        self.controller = None
        self.is_first_time = True
        self.decision = None
        self.simul_time = time.time()
        self.accumulated_reward = 0
        self.end_point = 0
        self.ROI_length = 1000 #(meters)
        self.safe_lane_change_distance = 0
        self.search_radius = None
        self.left_search_radius = None
        self.right_search_radius = None
        self.decision_changed = False


        ## visualize all waypoints ##
        # for n, p in enumerate(self.waypoints):
        #     world.debug.draw_string(p.transform.location, 'o', draw_shadow=True,
        #                             color=carla.Color(r=255, g=255, b=255), life_time=999)

        settings = self.world.get_settings()
        # settings.no_rendering_mode = True
        # settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        self.extra_num = 4
        self.scenario = "scenario2" #"random"
        self.pilot_style = "manual" # "auto"

        self.section = 0
        self.lane_distance_between_start =None
        self.lane_distance_between_end = None
        self.lane_change_point = None
        self.episode_start = None
        self.index = 0
        self.pre_max_Lane_num = self.max_Lane_num

        self.restart()
        self.main()

    def restart(self):
        self.decision = None
        self.simul_time = time.time()

        self.lane_change_time = time.time()-100.0
        self.max_Lane_num = 4
        self.ego_Lane = 2
        self.controller = None
        self.accumulated_reward = 0
        self.section = 0
        self.episode_start = time.time()
        self.pre_max_Lane_num = self.max_Lane_num
        self.index = 0
        print('start destroying actors.')

        if len(self.actor_list) != 0:
            # print('destroying actors.')
            # print("actor 제거 :", self.actor_list)

            for actor in self.actor_list:
                # print("finally 에서 actor 제거 :", self.actor_list)
                if actor.is_alive:
                    actor.destroy()
            print("finshed actor destroy")
            # for x in self.actor_list:
            #     try:
            #         client.apply_batch([carla.command.DestroyActors(x.id)])
            #     except:
            #         continue
        if len(self.extra_list) != 0:
            # client.apply_batch([carla.command.DestoryActors(x.id) for x in self.extra_list])
            # print("extra 제거 :", self.extra_list)
            for x in self.extra_list:
                try:
                    client.apply_batch([carla.command.DestroyActor(x.id)])
                except:
                    continue
            # for extra in self.extra_list:
            #     # print("finally 에서 actor 제거 :", self.extra_list)
            #     print(extra.is_alive)
            #     if extra.is_alive:
            #         extra.destroy()
            print("finshed extra destroy")

            # for x in self.extra_list:
            #     try:
            #         client.apply_batch([carla.command.DestroyActor(x.id)])
            #     except:
            #         continue
        print('finished destroying actors.')

        self.actor_list = []
        self.extra_list = []
        self.ROI_extra_list = []
        self.extra_controller_list = []
        self.extra_dl_list = []
        self.ROI_extra_dl_list = []
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
        # self.end_point = self.waypoint.next(400)[0].transform.location
        # print(self.waypoint.transform)
        ## Ego vehicle의 global Route 출력 ##
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
        spawn_point = None

        if self.scenario == "random":
            for i in range(10, 10 * (self.extra_num) + 1, 10):
                dl=random.choice([-1,0,1])
                self.extra_dl_list.append(dl)
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
            for extra in self.extra_list:
                if self.pilot_style == "auto":
                    extra.set_autopilot()
                elif self.pilot_style == "manual":
                    extra.set_autopilot(True)
                    # controller = Pure_puresuit_controller(extra, self.waypoint, None, 30)  # km/h
                    # self.extra_controller_list.append(controller)
                    target_velocity = 30  # random.randrange(10, 40) # km/h
                    extra.enable_constant_velocity(extra.get_transform().get_right_vector() * target_velocity / 3.6)
                    # traffic_manager.auto_lane_change(extra,False)
        elif self.scenario == "scenario1":
            self.extra_num = 3
            d = 15
            self.extra_dl_list = [-1, 0, 1]

            for i in range(3):
                if i==0:
                    spawn_point = self.waypoint.next(d)[0].get_left_lane().transform
                elif i==1:
                    spawn_point = self.waypoint.next(d)[0].transform
                elif i==2:
                    spawn_point = self.waypoint.next(d)[0].get_right_lane().transform
                else:
                    print("Except ")
                spawn_point = carla.Transform((spawn_point.location + carla.Location(z=1)), spawn_point.rotation)
                spawn_points.append(spawn_point)
                blueprint = random.choice(blueprints)
                # print(blueprint.has_attribute('color'))
                if blueprint.has_attribute('color'):
                    # color = random.choice(blueprint.get_attribute('color').recommended_values)
                    # print(blueprint.get_attribute('color').recommended_values)
                    color = '255,255,255'
                    blueprint.set_attribute('color', color)
                extra = self.world.spawn_actor(blueprint, spawn_point)
                self.extra_list.append(extra)
            for extra in self.extra_list:
                if self.pilot_style == "auto":
                    extra.set_autopilot()
                elif self.pilot_style == "manual":
                    extra.set_autopilot(True)
                    # controller = Pure_puresuit_controller(extra, self.waypoint, None, 30)  # km/h
                    # self.extra_controller_list.append(controller)
                    target_velocity = 30  # random.randrange(10, 40) # km/h
                    extra.enable_constant_velocity(extra.get_transform().get_right_vector() * target_velocity / 3.6)
                    # traffic_manager.auto_lane_change(extra,False)
        elif self.scenario == "scenario2":
            self.extra_num = 4
            d = 15
            self.extra_dl_list = [-1, 0, 1, 2]

            for i in range(4):
                if i==0:
                    spawn_point = self.waypoint.next(d)[0].get_left_lane().transform
                elif i==1:
                    spawn_point = self.waypoint.next(d)[0].transform
                elif i==2:
                    spawn_point = self.waypoint.next(d)[0].get_right_lane().transform
                elif i ==3:
                    spawn_point = self.waypoint.next(200)[0].get_right_lane().get_right_lane().transform
                else:
                    print("Except ")
                spawn_point = carla.Transform((spawn_point.location + carla.Location(z=1)), spawn_point.rotation)
                spawn_points.append(spawn_point)
                blueprint = random.choice(blueprints)
                # print(blueprint.has_attribute('color'))
                if blueprint.has_attribute('color'):
                    # color = random.choice(blueprint.get_attribute('color').recommended_values)
                    # print(blueprint.get_attribute('color').recommended_values)
                    color = '255,255,255'
                    blueprint.set_attribute('color', color)
                extra = self.world.spawn_actor(blueprint, spawn_point)
                self.extra_list.append(extra)
            for extra in self.extra_list:
                if self.pilot_style == "auto":
                    extra.set_autopilot()
                elif self.pilot_style == "manual":
                    extra.set_autopilot(True)
                    # controller = Pure_puresuit_controller(extra, self.waypoint, None, 30)  # km/h
                    # self.extra_controller_list.append(controller)
                    if extra == self.extra_list[-1]:
                        target_velocity = 0.0
                    else:
                        target_velocity = 30.0  # random.randrange(10, 40) # km/h
                    extra.enable_constant_velocity(extra.get_transform().get_right_vector() * target_velocity / 3.6)
                    # traffic_manager.auto_lane_change(extra,False)
        elif self.scenario == "scenario3":
            self.extra_num = 4
            self.extra_dl_list = [-1, 0, 1, 2]

            for i in range(4):
                if i == 0:
                    spawn_point = self.waypoint.next(100)[0].get_left_lane().transform
                elif i == 1:
                    spawn_point = self.waypoint.next(150)[0].transform
                elif i == 2:
                    spawn_point = self.waypoint.next(100)[0].get_right_lane().transform
                elif i == 3:
                    spawn_point = self.waypoint.next(150)[0].get_right_lane().get_right_lane().transform
                elif i == 4:
                    spawn_point
                else:
                    print("Except ")
                spawn_point = carla.Transform((spawn_point.location + carla.Location(z=1)), spawn_point.rotation)
                spawn_points.append(spawn_point)
                blueprint = random.choice(blueprints)
                # print(blueprint.has_attribute('color'))
                if blueprint.has_attribute('color'):
                    # color = random.choice(blueprint.get_attribute('color').recommended_values)
                    # print(blueprint.get_attribute('color').recommended_values)
                    color = '255,255,255'
                    blueprint.set_attribute('color', color)
                extra = self.world.spawn_actor(blueprint, spawn_point)
                self.extra_list.append(extra)
            for extra in self.extra_list:
                if self.pilot_style == "auto":
                    extra.set_autopilot()
                elif self.pilot_style == "manual":
                    extra.set_autopilot(True)
                    # controller = Pure_puresuit_controller(extra, self.waypoint, None, 30)  # km/h
                    # self.extra_controller_list.append(controller)
                    target_velocity = 20.0

                    extra.enable_constant_velocity(extra.get_transform().get_right_vector() * target_velocity / 3.6)
                    # traffic_manager.auto_lane_change(extra,False)
        print('Extra Genration Finished')


        self.spectator.set_transform(carla.Transform(self.player.get_transform().location + carla.Location(z=100),
                            carla.Rotation(pitch=-90)))

        # extra_target_velocity = 10
        port = 8000
        # traffic_manager = client.get_trafficmanager(port)
        # traffic_manager.set_global_distance_to_leading_vehicle(100.0)
        # traffic_manager.set_synchronous_mode(True) # for std::out_of_range eeror
        # tm_port = traffic_manager.get_port()

        print("tm setting finished")
        # self.player.set_autopilot(True,tm_port)
        # traffic_manager.auto_lane_change(self.player, False)
        self.controller = Pure_puresuit_controller(self.player, self.waypoint, self.extra_list, 70)  # km/h
        print("controller setting finished")
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

        # for response in client.apply_batch_sync(batch):
        #     if response.error:
        #         logging.error(response.error)
        #     else:
        #         self.extra_list.append(response.actor_id)

    # def restart(self):
    #     self.controller = Pure_puresuit_controller(self.player, self.waypoint, self.extra_list, 70)  # km/h


    def is_extra_front_than_ego(self,extra_pos):

        extra_pos_tensor = torch.tensor([[extra_pos.x, extra_pos.y, extra_pos.z, 1.0]])

        theta_z =  math.radians(self.player.get_transform().rotation.yaw)
        trans = self.player.get_transform().location

        n = torch.tensor([[math.cos(theta_z), math.sin(theta_z), 0]])
        o = torch.tensor([[-math.sin(theta_z),  math.cos(theta_z), 0]])
        a = torch.tensor([[0.0, 0.0, 1.0]])
        p = torch.tensor([[trans.x, trans.y, trans.z]])

        m41 = -torch.dot(torch.squeeze(n),torch.squeeze(p)).unsqueeze(0).unsqueeze(1)
        m42 = -torch.dot(torch.squeeze(o),torch.squeeze(p)).unsqueeze(0).unsqueeze(1)
        m43 = -torch.dot(torch.squeeze(a), torch.squeeze(p)).unsqueeze(0).unsqueeze(1)
        m = torch.tensor([[0.0, 0.0, 0.0, 1.0]])


        Global_T_relative =\
            torch.cat((torch.cat((n,m41),1),torch.cat((o,m42),1),torch.cat((a,m43),1),m),0)

        output = torch.matmul(Global_T_relative, extra_pos_tensor.T)
        if output[0]>0:
            return True
        else:
            return False
        # print(Global_T_relative)
        # print("output = ", Global_T_relative*[[trans.x], [trans.y], [trans.z], [1]])


    def step(self, decision):
        SECONDS_PER_EPISODE = 1000
        plc = 1
        # decision = None
        '''
        # Simple Action (action number: 3)
        action_test = action -1
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=action_test*self.STEER_AMT))
        '''

        # Complex Action
        if decision == 0:  # LK
            plc = 0
        # elif action == -1:  # left
        #     plc += 10
        #     decision = -1
        # elif action == 1:  # right
        #     plc += 10
        #     decision = 1

        '''
        if len(self.collision_hist) != 0:
            done = True
            reward = -200 + (time.time()-self.episode_start)*15/SECONDS_PER_EPISODE
        else:
            done = False
            reward = 1
        '''
        # end_length=math.sqrt((self.end_point.x - self.player.get_location().x)**2+(self.end_point.y - self.player.get_location().y)**2)
        done = False
        if len(self.collision_sensor.history) != 0:
            done = True
            reward = -1
        elif time.time()-self.simul_time > 15:
            print("done")
            done = True
            reward = 1
        elif self.max_Lane_num < self.ego_Lane:
            done = True
            reward = -1
        else:
            reward = - abs(self.controller.desired_vel-self.controller.velocity)/self.controller.desired_vel-plc
        # print(reward)
        self.accumulated_reward += reward

        if time.time() > self.episode_start + SECONDS_PER_EPISODE:
            done = True

        #state length = 4 * num of extra vehicles + 1

        state, x_static = self.get_next_state(decision) #get now state

        return state, x_static, decision ,reward, None, None , done
                                        # Next State 표현 필요
    def search_distance_between_vehicles(self,extra_forth_lane_waypoint,from_,to_):
        """
        :param search_raidus:
        :param extra: extra actor
        :param from_: data type = 'extra actor' or ego_forth_lane_waypoint
        :param to_:
        :return: distance from_ to to_
        """

        distance=9999
        sign = 0
        search_raidus = 5
        i = 0.01

        # self.world.debug.draw_string(ego_lane_waypoint.transform.location,
        #                                   'o', draw_shadow=True,
        #                                  color=carla.Color(r=0, g=0, b=255), life_time=-1)

        if from_ == extra_forth_lane_waypoint:
            from_waypoint = extra_forth_lane_waypoint
            to_waypoint =  to_
            sign = -1
        else:
            from_waypoint = to_
            to_waypoint = extra_forth_lane_waypoint
            sign = 1

        while search_raidus <= distance:
            # pre_distance = distance
            distance = ((from_waypoint.next(i)[0].transform.location.x - to_waypoint.transform.location.x) ** 2 +
                        (from_waypoint.next(i)[0].transform.location.y - to_waypoint.transform.location.y) ** 2 +
                        (from_waypoint.next(i)[0].transform.location.z - to_waypoint.transform.location.z) ** 2)**0.5

            if round(distance - search_raidus) > 0:
                i += round(distance - search_raidus)
                print("i : " ,i)
                # break
            else:
                # self.world.debug.draw_string(from_waypoint.next(distance + i)[0].transform.location,
                #                              'o', draw_shadow=True, color=carla.Color(r=255, g=255, b=255), life_time=-1)
                print((distance + i)*sign)
                print("finish")
                return (distance + i)*sign

    def get_dr(self):

        distance = 999
        dr= np.zeros(len(self.extra_list))
        ego_forth_lane = self.get_waypoint_of_forth_lane(self.player)

        self.world.debug.draw_string(ego_forth_lane.transform.location,
                                     'o', draw_shadow=True,
                                     color=carla.Color(r=255, g=0, b=0), life_time=-1)
        for index, extra in enumerate(self.extra_list):

            extra_pos = extra.get_transform().location
            extra_forth_lane_waypoint = self.get_waypoint_of_forth_lane(extra)
            self.world.debug.draw_string(extra_forth_lane_waypoint.transform.location,
                                             'o', draw_shadow=True,
                                             color=carla.Color(r=255, g=0, b=0), life_time=-1)
            if self.is_extra_front_than_ego(extra_pos) == False:
                dr[index] = self.search_distance_between_vehicles(extra_forth_lane_waypoint,extra_forth_lane_waypoint,ego_forth_lane)
            else:
                dr[index] = self.search_distance_between_vehicles(extra_forth_lane_waypoint,ego_forth_lane,extra_forth_lane_waypoint)
        return dr

                # print("lane id : ", last_lane_waypoint.lane_id)

    def get_next_state(self,decision=None):
        """
        dl : relative lane num after ching the lane
        dr, dv, da : now state
        """
        state = []
        for x, extra in enumerate(self.extra_list):

            if decision == 1:
                self.extra_dl_list[x] =self.extra_dl_list[x]+1
                # self.ROI_extra_dl_list[x] =self.ROI_extra_dl_list[x]+1

            elif decision == -1:
                self.extra_dl_list[x] = self.extra_dl_list[x]-1
                # self.ROI_extra_dl_list[x] = self.ROI_extra_dl_list[x]-1

            else:
                pass

            extra_pos = extra.get_transform().location
            extra_vel = extra.get_velocity()
            extra_acel = extra.get_acceleration()
            sign = 0
            if self.is_extra_front_than_ego(extra_pos) == True:
                sign = 1
            else:
                sign = -1
            dr = sign * ((extra_pos.x - self.player.get_transform().location.x) ** 2 +
                         (extra_pos.y - self.player.get_transform().location.y) ** 2 +
                         (extra_pos.z - self.player.get_transform().location.z) ** 2)**0.5 - abs(self.waypoint.lane_width*(self.extra_dl_list[x]))

            dv =(self.player.get_velocity().x** 2 + self.player.get_velocity().y** 2 + self.player.get_velocity().z** 2) ** 0.5- (extra_vel.x ** 2 + extra_vel.y ** 2 + extra_vel.z ** 2) ** 0.5

            length = 2 * extra.bounding_box.extent.x/10 # length of extra vehicle

            # da = ((extra_acel.x - self.player.get_acceleration().x) ** 2 + (
            #         extra_acel.y - self.player.get_acceleration().y) ** 2 +
            #       (extra_acel.z - self.player.get_acceleration().z) ** 2) ** 0.5

            state_dyn = [dr, dv, length, self.extra_dl_list[x]]
            state.append(state_dyn)
            # state.append(dr)

            # state.append(dv)
            # state.append(da)
            # state.append(self.extra_dl_list[x])

        if decision == 1:
            self.ego_Lane +=1
        elif decision ==-1:
            self.ego_Lane +=-1
        else:
            pass
        x_static= []
        x_static.append([self.ego_Lane, self.controller.velocity, self.search_distance_valid()])


        # state.append(x_static)

        # state.append(self.ego_Lane)
        out = [state, x_static]
        return out

    def get_waypoint_of_forth_lane(self,actor):
        last_lane_waypoint = self.map.get_waypoint(actor.get_transform().location,
                                                   lane_type=carla.LaneType.Driving)  # self.controller.waypoint
        try:
            while last_lane_waypoint.lane_id > -4:  # get waypoint of forth's lane
                last_lane_waypoint = last_lane_waypoint.get_right_lane()
            # print("lane id : ", last_lane_waypoint.lane_id)
        except:
            self.world.debug.draw_string(last_lane_waypoint.transfrom.location, 'o', draw_shadow=True,
                                         color=carla.Color(r=255, g=255, b=255), life_time=1)
            print("4-th lane not found")
        return last_lane_waypoint

    def search_distance_valid(self):
            # index= 4-self.ego_Lane
            # print("index :", index)
            search_raidus = 5
            distance = 9999
            # pre_distance = 0
            i = 0.01
            last_lane_waypoint= self.get_waypoint_of_forth_lane(self.player)

            # self.world.debug.draw_string(last_lane_waypoint.transform.location,
            #                                  '4', draw_shadow=True,
            #                                  color=carla.Color(r=0, g=255, b=255), life_time=9999)

            # print(self.ego_Lane)

            if self.max_Lane_num != self.pre_max_Lane_num:
                self.index +=1
                if self.index >5:
                    self.index = 0
                self.pre_max_Lane_num = self.max_Lane_num
            # print(self.index)
            if self.index % 2 == 0:
                while search_raidus <= distance:
                    # pre_distance = distance
                    distance = ((last_lane_waypoint.next(i)[0].transform.location.x-self.lane_change_point[self.index].x)**2+(last_lane_waypoint.next(i)[0].transform.location.y-self.lane_change_point[self.index].y)**2+(last_lane_waypoint.next(i)[0].transform.location.z-self.lane_change_point[self.index].z)**2)**0.5
                    # if pre_distance <= distance:
                    #     print("wrong")
                    #     return distance + i
                    if round(distance - search_raidus) > 0:
                        i +=round(distance-search_raidus)
                        # break
                    else:
                        return distance +i
            else:
                distance = math.hypot(last_lane_waypoint.transform.location.x-self.lane_change_point[self.index].x,
                           last_lane_waypoint.transform.location.y-self.lane_change_point[self.index].y)
                return -distance

            # if self.max_Lane_num ==4:
            #         while search_raidus <= distance:
            #             pre_distance = distance
            #             # distance = ((last_lane_waypoint.next(i)[0].x-self.lane_start_point.x)**2+(last_lane_waypoint.next(i)[0].y-self.lane_start_point.y)**2+(last_lane_waypoint.next(i)[0].z-self.lane_start_point.z)**2)**0.5
            #             distance = ((last_lane_waypoint.next(i)[0].transform.location.x-self.lane_start_point[self.section].x)**2+(last_lane_waypoint.next(i)[0].transform.location.y-self.lane_start_point[self.section].y)**2+(last_lane_waypoint.next(i)[0].transform.location.z-self.lane_start_point[self.section].z)**2)**0.5
            #
            #             # print("distance :",distance , "i :", i)
            #             if pre_distance <= distance:
            #                 print("pre_distance <= distance error")
            #                 break
            #             elif round(distance - search_raidus) > 0:
            #                 i += round(distance - search_raidus)
            #             else:
            #                 return min(distance + i, self.ROI_length)
            #             # print("i updated :", i)
            #         # self.max_Lane_num =3
            # elif self.max_Lane_num == 3:
            #     while search_raidus <= distance:
            #         distance = ((last_lane_waypoint.next(i)[0].transform.location.x - self.lane_finished_point[
            #             self.section].x) ** 2 + (last_lane_waypoint.next(i)[0].transform.location.y -
            #                                      self.lane_finished_point[self.section].y) ** 2 + (
            #                             last_lane_waypoint.next(i)[0].transform.location.z -
            #                             self.lane_finished_point[self.section].z) ** 2) ** 0.5
            #         self.world.debug.draw_string(last_lane_waypoint.next(i)[0].transform.location,
            #                                      'o', draw_shadow=True,
            #                                      color=carla.Color(r=255, g=255, b=0), life_time=9999)
            #         if pre_distance <= distance:
            #             print("pre_distance <= distance error")
            #             break
            #         elif round(distance - search_raidus) > 0:
            #             i += round(distance - search_raidus)
            #         else:
            #             return min(distance + i, self.ROI_length)
            #
            #         if round(distance - search_raidus) > 0:
            #             i += round(distance - search_raidus)
            #         else:
            #             return max(-(distance + i), -self.ROI_length)
    def can_lane_change(self,decision,state):

        """

        :param decision:
        :param state:  state_dyn = [dr/self.ROI_length, dv, length, self.extra_dl_list[x]]
        :return:
        """
        self.waypoint = self.map.get_waypoint(self.player.get_location(),lane_type=carla.LaneType.Driving)
        self.safe_lane_change_distance = int(self.controller.velocity/3.0+10.0)
        result = True

        self.world.debug.draw_string(self.waypoint.next(self.safe_lane_change_distance)[0].transform.location, 'o',
                                     draw_shadow=True, color=carla.Color(r=255, g=0, b=0),
                                     life_time=-1)

        if decision ==0:
            # print("tried strait in safety check")
            return True
        for i, state_dyn in enumerate(state):

            if state_dyn[3] == decision:
                if state_dyn[0] >= 0.0 and state_dyn[0] <= self.safe_lane_change_distance:
                    print("not enough side front distance")
                    return False
                elif state_dyn[0] <= 0.0 and state_dyn[1] <= 0.0 and abs(state_dyn[0]) <= self.safe_lane_change_distance:
                    print("not enough behind distance")
                    return False
                elif state_dyn[0] <= 0 and abs(state_dyn[0]) <= self.safe_lane_change_distance/2.0:
                    print("not enough behind distance")
                    return False
            elif state_dyn[3] == 0:

                if state_dyn[0] > 0.0 and state_dyn[0]  <= self.safe_lane_change_distance:
                    print("not engouth leading distance")
                    return False
        return True
        # i = 0
        # for d, extra in self.hud.vehicles:
        #
        #     if d > 40:
        #         break
        #     extra_pos = extra.get_location()
        #     # is_front_extra_vehicle = False
        #     # if self.is_extra_front_than_ego(extra_pos):
        #     #     print("front extra vehicle")
        #     #     is_front_extra_vehicle=True
        #     #     pass
        #     # else:
        #     #     print("behind extra vehicle")
        #     #     extra_pos = self.player.get_location()
        #     #     self.waypoint = self.map.get_waypoint(extra_pos, lane_type=carla.LaneType.Driving)
        #     self.world.debug.draw_string(self.waypoint.transform.location, str(i), draw_shadow=True,
        #                                  color=carla.Color(r=255, g=255, b=0), life_time=1)
        #
        #     # if is_front_extra_vehicle==False:
        #     #     self.world.debug.draw_string(self.waypoint.transform.location,str(i), draw_shadow=True,color=carla.Color(r=255, g=255, b=0), life_time=1)
        #     # else:
        #     #     self.world.debug.draw_string(extra_pos,str(i), draw_shadow=True,color=carla.Color(r=255, g=255, b=0), life_time=1)
        #     i += 1
        #     for x in range(1, self.safe_lane_change_distance + 1 - int(self.waypoint.lane_width), 1):
        #         self.search_radius = ((extra_pos.x - self.waypoint.next(x)[
        #             0].transform.location.x) ** 2 + (extra_pos.y - self.waypoint.next(x)[
        #             0].transform.location.y) ** 2) ** 0.5
        #         self.world.debug.draw_string(self.waypoint.next(x)[0].transform.location, 'o', draw_shadow=True, color=carla.Color(r=0, g=255, b=0),
        #                                      life_time=1)
        #
        #         if self.search_radius <= self.waypoint.lane_width / 2:
        #             print("distance between front :",x)
        #             self.world.debug.draw_string(self.waypoint.next(x)[0].transform.location, 'x', draw_shadow=True,
        #                                          color=carla.Color(r=255, g=0, b=0),
        #                                          life_time=1)
        #             print("front vehicle found")
        #             if x>=self.safe_lane_change_distance*2.0/3.0:
        #                 pass
        #             else:
        #                 print("false1")
        #                 return False
        #
        #         if decision ==1:
        #             self.right_search_radius = ((extra_pos.x - self.waypoint.next(x)[
        #                 0].get_right_lane().transform.location.x) ** 2 + (extra_pos.y - self.waypoint.next(x)[
        #                 0].get_right_lane().transform.location.y) ** 2) ** 0.5
        #             self.world.debug.draw_string(self.waypoint.next(x)[0].get_right_lane().transform.location, 'o', draw_shadow=True,
        #                                          color=carla.Color(r=0, g=255, b=0),
        #                                          life_time=1)
        #             if self.right_search_radius <= self.waypoint.lane_width / 2.0:
        #                 print("right distance with vehicle :", x)
        #                 print("right vehicle found")
        #                 return False
        #         elif decision ==-1:
        #             try:
        #                 self.left_search_radius = ((extra_pos.x - self.waypoint.next(x)[
        #                     0].get_left_lane().transform.location.x) ** 2 + (extra_pos.y - self.waypoint.next(x)[
        #                     0].get_left_lane().transform.location.y) ** 2) ** 0.5
        #             except:
        #                 return False
        #             self.world.debug.draw_string(self.waypoint.next(x)[0].get_left_lane().transform.location, 'o',
        #                                          draw_shadow=True,
        #                                          color=carla.Color(r=0, g=255, b=0),
        #                                          life_time=1)
        #             if self.left_search_radius <= self.waypoint.lane_width / 2.0:
        #                 self.world.debug.draw_string(self.waypoint.next(x)[0].transform.location, 'o', draw_shadow=True,
        #                                              color=carla.Color(r=255, g=0, b=0),
        #                                              life_time=1)
        #                 print("left vehicle found")
        #                 return False
        # print("possible lane change")
        # return True




    def safety_check(self,decision,state, safe_lane_change_again_time=3):
        remained_action_list = None
        action = decision
        if decision !=0 and decision !=None:
            # if (time.time() - self.lane_change_time) > safe_lane_change_again_time and decision == -1 and self.ego_Lane ==1:
            #     print("here")
            if (time.time()-self.lane_change_time) <= safe_lane_change_again_time: # dont change frequently
                self.decision_changed = True
                return 0 #즉 직진
            elif self.agent.selection_method == 'random' and decision == 1 and self.ego_Lane ==self.max_Lane_num: #dont leave max lane
                action = random.randrange(-1,1)

            elif self.agent.selection_method == 'random' and decision == -1 and self.ego_Lane ==1: #dont leave min lane
                action = random.randrange(0, 2)

            elif self.ego_Lane == self.max_Lane_num and decision == 1:
                self.decision_changed = True
                remained_action_list = self.agent.q_value[0][0:2]
                print("remained_action_list:",remained_action_list)
                action = int(remained_action_list.argmax().item())-1

            elif self.ego_Lane == 1 and decision == -1:
                self.decision_changed = True
                remained_action_list =  self.agent.q_value[0][1:3]
                print("remained_action_list:",remained_action_list)

                action =  int(self.agent.q_value[0][1:3].argmax().item())
                #not max exception


            if action != 0:# and self.can_lane_change(action,state):
                self.lane_change_time = time.time()
                return action
            elif remained_action_list is not None:
                action = int(remained_action_list.argmin().item())
                return action
                # if self.can_lane_change(action, state):
                #     return action
                # else:
                #     print("safety exception occured, action :", action, "before decision:",decision)
            else:
                return 0
            # elif decision ==1: # and self.ego_Lane != self.max_Lane_num:
            #     self.lane_change_time = time.time()
            #     return decision
            # elif decision ==-1: # and self.ego_Lane != 1:
            #     self.lane_change_time = time.time()
            #     return decision

            # else: #3차선에서 우 차도 판단, 1차선에서 좌차선 판단
            #     if self.agent.selection_method == 'random' and decision == 1 and self.ego_Lane == self.max_Lane_num:
            #         action = random.randrange(-1, 1)
            #         return action
            #     elif self.agent.selection_method == 'random' and decision == -1 and self.ego_Lane == 1:
            #         action = random.randrange(0, 2)
            #         return action
            #     elif self.ego_Lane == self.max_Lane_num and decision == 1:
            #         return int(self.agent.q_value[0][0:2].argmax().item()) - 1
            #     elif self.ego_Lane == 1 and decision == -1:
            #         return int(self.agent.q_value[0][1:3].argmax().item())
        else:
            return decision


    def main_test(self):
        restart_time = time.time()
        PATH = "/home/a/RL_decision/trained_info.tar"
        print(torch.cuda.get_device_name())
        clock = pygame.time.Clock()
        Keyboardcontrol = KeyboardControl(self, False)
        display = pygame.display.set_mode(
            (self.width, self.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.lane_start_point = [carla.Location(x=14.905815, y=-135.747452, z=0.000000),carla.Location(x=172.745468, y=-364.531799, z=0.000000) , carla.Location(x=382.441040, y=-212.488907, z=0.000000)]
        self.lane_finished_point = [carla.Location(x=14.631096, y=-205.746918, z=0.000000),carla.Location(x=232.962860, y=-364.149139, z=0.000000) ,carla.Location(x=376.542816, y=-10.352980, z=0.000000) ]
        self.lane_change_point = [carla.Location(x=14.905815, y=-135.747452, z=0.000000),carla.Location(x=14.631096, y=-205.746918, z=0.000000),carla.Location(x=172.745468, y=-364.531799, z=0.000000) ,carla.Location(x=232.962860, y=-364.149139, z=0.000000), carla.Location(x=382.441040, y=-212.488907, z=0.000000),carla.Location(x=376.542816, y=-10.352980, z=0.000000)]

        for x in self.lane_change_point:
            self.world.debug.draw_string(x,'o', draw_shadow=True,color=carla.Color(r=0, g=255, b=0), life_time=9999)

        lane_distance_between_points = []
        for i in range(len(self.lane_finished_point)):
            lane_distance_between_points.append(((self.lane_start_point[i].x - self.lane_finished_point[i].x) ** 2 + (
                        self.lane_start_point[i].y - self.lane_finished_point[i].y) ** 2 + (
                                                            self.lane_start_point[i].z - self.lane_finished_point[i].z) ** 2)**0.5)

        pre_decision = None
        while True:
            # if time.time()-restart_time > 10:
            #     print("restart")
            #     self.restart()

            if Keyboardcontrol.parse_events(client, self, clock):
                return

            self.spectator.set_transform(
                carla.Transform(self.player.get_transform().location + carla.Location(z=50),
                                carla.Rotation(pitch=-90)))
            self.camera_rgb.render(display)
            self.hud.render(display)
            pygame.display.flip()

            extra_pos = self.extra_list[0].get_transform().location
            # print(self.is_extra_front_than_ego(extra_pos))

            ## Get max lane ##
            # print("start get lane")
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
                    # world.debug.draw_string(self.player.get_transform().location, 'o', draw_shadow = True,
                    #                                 color = carla.Color(r=255, g=255, b=0), life_time = 999)

                elif max(lane_distance_between_points[self.section], self.lane_distance_between_start, self.lane_distance_between_end) == \
                        self.lane_distance_between_start and self.max_Lane_num ==3:
                    self.section+=1
                    self.max_Lane_num=4
                    if self.section >= len(lane_distance_between_points):  # when, section_num = 3
                        self.section = 0
            ## finished get max lane ##
            # print("finished get lane")

            # self.search_distance_valid()
            # print("distance :" ,,"max_lane_num : ", self.max_Lane_num)

            # if self.max_Lane_num==3:
                # self.world.debug.draw_string(self.player.get_transform().location,
                #                              'o', draw_shadow=True,
                #                              color=carla.Color(r=255, g=255, b=0), life_time=9999)
            if time.time()-self.simul_time > 20:
                for extra in self.extra_list:
                    extra.enable_constant_velocity(extra.get_transform().get_right_vector() * 0.0)

            # if time.time()-self.lane_change_time >10:
            #     self.lane_change_time = time.time()
            #     if pre_decision is None:
            #         self.decision = 1
            #         # self.decision = self.safety_check(self.decision)
            #
            #         self.ego_Lane+=1
            #         pre_decision = 1
            #
            #
            #     elif pre_decision ==1:
            #         pre_decision = -1
            #         self.ego_Lane+=-1
            #         self.decision = -1
            #
            #     elif pre_decision ==-1:
            #         self.decision = 1
            #         self.ego_Lane+=1
            #         pre_decision =1
            #
            # else:
            #     self.decision = 0
            self.decision= 0
            self.controller.apply_control(self.decision)
            # self.world.wait_for_tick(10.0)
            clock.tick(40)

            self.hud.tick(self, clock)

    def main(self):

        PATH = "/home/a/RL_decision/trained_info.tar"
        print(torch.cuda.get_device_name())
        clock = pygame.time.Clock()
        Keyboardcontrol = KeyboardControl(self, False)
        # display = pygame.display.set_mode(
        #     (self.width, self.height),
        #     pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.agent = decision_driving_Agent(self.input_size,self.output_size,True,1000,self.extra_num)

        # # 모델의 state_dict 출력
        # print("Model's state_dict:")
        # for param_tensor in self.agent.model.state_dict():
        #     print(param_tensor, "\t", self.agent.model.state_dict()[param_tensor].size())
        #
        # # 옵티마이저의 state_dict 출력
        # print("Optimizer's state_dict:")
        # for var_name in self.agent.optimizer.state_dict():
        #     print(var_name, "\t", self.agent.optimizer.state_dict()[var_name])

        self.lane_start_point = [carla.Location(x=14.905815, y=-135.747452, z=0.000000),
                            carla.Location(x=172.745468, y=-364.531799, z=0.000000),
                            carla.Location(x=382.441040, y=-212.488907, z=0.000000)]
        self.lane_finished_point = [carla.Location(x=14.631096, y=-205.746918, z=0.000000),
                               carla.Location(x=232.962860, y=-364.149139, z=0.000000),
                               carla.Location(x=376.542816, y=-10.352980, z=0.000000)]
        self.lane_change_point = [carla.Location(x=14.905815, y=-135.747452, z=0.000000),carla.Location(x=14.631096, y=-205.746918, z=0.000000),carla.Location(x=172.745468, y=-364.531799, z=0.000000) ,carla.Location(x=232.962860, y=-364.149139, z=0.000000), carla.Location(x=382.441040, y=-212.488907, z=0.000000),carla.Location(x=376.542816, y=-10.352980, z=0.000000)]

        for x in self.lane_change_point:
            self.world.debug.draw_string(x,'o', draw_shadow=True,
                                         color=carla.Color(r=0, g=255, b=0), life_time=9999)

        lane_distance_between_points = []
        for i in range(len(self.lane_finished_point)):
            lane_distance_between_points.append((self.lane_start_point[i].x - self.lane_finished_point[i].x) ** 2 + (
                    self.lane_start_point[i].y - self.lane_finished_point[i].y) ** 2 + (
                                                        self.lane_start_point[i].z - self.lane_finished_point[i].z) ** 2)

        epoch_init = 0

        if(os.path.exists(PATH)):
            print("저장된 가중치 불러옴")
            checkpoint = torch.load(PATH)
            device = torch.device('cuda')
            self.agent.model.load_state_dict(checkpoint['model_state_dict'])
            self.agent.model.to(device)
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # self.agent.buffer.buffer = checkpoint['memorybuffer']
            epoch_init = checkpoint['epoch']
            self.agent.epsilon = checkpoint['epsilon']


        # self.agent.is_training = False
        print("epoch : ",epoch_init)
        # print(self.agent.buffer.size)

            # self.agent.buffer.load_state_dict(self.save_dir['memorybuffer'])

            # self.is_first_time = False
        # controller = VehicleControl(self, True)
        # self.player.apply_control(carla.Vehicle
        # Control(throttle=1.0, steer=0.0, brake=0.0))
        # vehicles = self.world.get_actors().filter('vehicle.*')
        # for i in vehicles:
        #     print(i.id)
        # try:

        n_iters = 2000
        is_error = 0

        for epoch in range(epoch_init + 1, n_iters + 1):
            # cnt = 0
            print("start epoch!")
            [state, x_static] = self.get_next_state()  # 초기 상태 s0 초기화
            self.start_epoch = True
            while (self.start_epoch == True):
                # self.hud.tick(self, clock)

                if Keyboardcontrol.parse_events(client, self, clock):
                    return

                # self.camera_rgb.render(display)
                # self.hud.render(display)
                # pygame.display.flip()

                self.spectator.set_transform(
                    carla.Transform(self.player.get_transform().location + carla.Location(z=100),
                                    carla.Rotation(pitch=-90)))

                ## Get max lane ##
                if self.section < len(lane_distance_between_points):
                    self.lane_distance_between_start = (
                            (self.player.get_transform().location.x - self.lane_start_point[self.section].x) ** 2 +
                            (self.player.get_transform().location.y - self.lane_start_point[self.section].y) ** 2)
                    self.lane_distance_between_end = (
                            (self.player.get_transform().location.x - self.lane_finished_point[self.section].x) ** 2 +
                            (self.player.get_transform().location.y - self.lane_finished_point[self.section].y) ** 2)

                    # print("self.lane_distance_between_start : ",self.lane_distance_between_start,"self.lane_distance_between_end :",self.lane_distance_between_end, "lane_distance_between_points[self.section]",lane_distance_between_points[self.section],"self.section :", self.section)
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
                        if self.section >= len(lane_distance_between_points):  # when, section_num = 3
                            self.section = 0
                ## finished get max lane ##
                ##visualize when, max lane ==3 ##
                # if self.max_Lane_num == 3:
                #     self.world.debug.draw_string(self.waypoint.transform.location,
                #                                  'o', draw_shadow=True,
                #                                  color=carla.Color(r=255, g=255, b=255), life_time=9999)
                # [self.extra_list, self.extra_dl_list] = self.agent.search_extra_in_ROI(self.extra_list,self.player,self.extra_dl_list)

                ## finished to visualize ##

                if self.agent.is_training:
                    ##dqn 과정##
                    # 가중치 초기화 (pytroch 내부)
                    # 입실론-그리디 행동 탐색 (act function)
                    # 메모리 버퍼에 MDP 튜플 얻기   ㅡ (step function)
                    # 메모리 버퍼에 MDP 튜플 저장   ㅡ
                    # optimal Q 추정             ㅡ   (learning function)
                    # Loss 계산                  ㅡ
                    # 가중치 업데이트              ㅡ

                    if epoch % 5 == 0:
                        # [w, b] = self.agent.model.parameters()  # unpack parameters
                        self.save_dir = torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.agent.model.state_dict(),
                            'optimizer_state_dict': self.agent.optimizer.state_dict(),
                            # 'memorybuffer': self.agent.buffer.buffer,
                            'epsilon': self.agent.epsilon}, PATH)
                    # print(clock.get_fps())

                    if self.decision is not None:
                        [next_state, next_x_static] = self.get_next_state()
                        if self.decision_changed == True:
                            sample = [state, x_static, self.decision, reward-1, next_state, next_x_static, done]
                        else:
                            sample = [state, x_static, self.decision, reward, next_state, next_x_static, done]

                        self.decision_changed = False

                        # print("sample, final action : ", self.decision)
                        if time.time() - self.simul_time > 7 and time.time() - self.simul_time < 8 and clock.get_fps() < 15:
                            time.sleep(5)
                            self.restart()
                            self.start_epoch = False
                        self.agent.buffer.append(sample)

                    # print("befor act")
                    self.decision = self.agent.act(state, x_static)
                    # print("after act")

                    # print(decision)
                    # if self.decision ==1 and self.max_Lane_num==self.ego_Lane:
                    #     print( " ")
                    # print("before safte check/ 판단 :", self.decision, "최대 차선 :", self.max_Lane_num, "ego lane :",self.ego_Lane)
                    # tmp = self.decision
                    # tmp2 = self.ego_Lane
                    self.decision = self.safety_check(self.decision, state)
                    # print("after safety check 판단 :", self.decision)

                    # print("before")
                    is_error = self.controller.apply_control(self.decision)
                    # print("extra_controller 개수 :", len(self.extra_controller_list))
                    # for i in range(len(self.extra_controller_list)):
                    #     self.extra_controller_list[i].apply_control()

                    # if self.decision == -1 and self.ego_Lane == 1:
                    #     print("decision :", tmp, "ego_lane : ", tmp2)

                    # print("after")

                    clock.tick(40)

                    # print("before step")
                    [state, x_static, decision, reward, __, _, done] = self.step(self.decision)
                    # print("after step")

                    if done:
                        print("epsilon :", self.agent.epsilon)
                        print("epoch : ", epoch, "누적 보상 : ", self.accumulated_reward)
                        writer.add_scalar('누적 보상 ', self.accumulated_reward, epoch)
                        if len(self.agent.buffer.size()) > self.agent.batch_size:
                            # print("start learning")
                            for i in range(100):
                                self.agent.learning()
                        # print("finished learning")
                        client.set_timeout(10)
                        time.sleep(5)
                        self.restart()
                        print("restart finished")
                        self.camera_rgb.toggle_camera()
                        print("toggle_camera finished")
                        self.start_epoch = False
                        print("start epoch")

                else:
                    self.decision = self.agent.act(state, x_static)
                    print(self.decision)

                    self.decision = self.safety_check(self.decision, state)

                    is_error = self.controller.apply_control(self.decision)
                    if is_error:
                        time.sleep(5)
                        self.restart()
                        self.start_epoch = False

                    [state, x_static, decision, reward, __, _, done] = self.step(self.decision)

                    if done:
                        print("epoch : ", epoch, "누적 보상 : ", self.accumulated_reward)
                        client.set_timeout(10)
                        self.restart()

        # finally:
        #     print('\ndestroying %d vehicles' % len(self.extra_list))
        #     # client.apply_batch([carla.command.DestroyActor(x) for x in self.extra_list])
        #
        #     print('destroying actors.')
        #     # client.apply_batch([carla.command.DestroyActors(x.id) for x in self.actor_list])
        #     # client.apply_batch([carla.command.DestroyActors(x.id) for x in self.extra_list])
        #     for actor in self.actor_list:
        #         # print("finally 에서 actor 제거 :", self.actor_list)
        #         if actor.is_alive:
        #             actor.destroy()
        #     for extra in self.extra_list:
        #         # print("finally 에서 actor 제거 :", self.extra_list)
        #         if extra.is_alive:
        #             extra.destroy()
        #
        #     # pygame.quit()
        #     print('done.')

if __name__ == '__main__':
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        client.load_world('Town04')
        client.set_timeout(10.0)

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

