import time
import glob
import os.path
import sys

tmp = ['/home/zz/carmaker_ros/ros/ros1_ws/devel/lib/python2.7/dist-packages',
       '/opt/ros/ros1/lib/python2.7/dist-packages', '/opt/ros/kinetic/lib/python2.7/dist-packages',
       '/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla-0.9.6-py3.5-linux-x86_64.egg',
       '/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg_FILES']
for index in range(0, len(tmp)):
    if tmp[index] in sys.path:
        sys.path.remove(tmp[index])

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
        self.extra_controller_list = []
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
        self.max_Lane_num = 3
        self.ego_Lane = 2
        self.agent = None
        self.controller = None
        self.is_first_time = True
        self.decision = None
        self.simul_time = time.time()
        self.accumulated_reward = 0
        self.end_point = 0

        ## visualize all waypoints ##
        # for n, p in enumerate(self.waypoints):
            # if n>1000:
            #     break
            # world.debug.draw_string(p.transform.location, 'o', draw_shadow=True,
            #                         color=carla.Color(r=255, g=255, b=255), life_time=30)

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        self.extra_num = 4

        self.episode_start = None
        self.restart()
        self.main()

    def restart(self):
        self.decision = None
        self.simul_time = time.time()
        self.lane_change_time = time.time()
        self.max_Lane_num = 3
        self.ego_Lane = 2
        self.controller = None
        self.accumulated_reward = 0

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

        self.load_traj()

        self.waypoint = self.map.get_waypoint(start_pose.location,lane_type=carla.LaneType.Driving)
        self.end_point = self.waypoint.next(400)[0].transform.location
        # print(self.waypoint.transform)
        ## Ego vehicle의 global Route 출력##
        world.debug.draw_string(self.waypoint.next(400)[0].transform.location, 'o', draw_shadow=True,
                                color=carla.Color(r=0, g=255, b=255), life_time=100)


        # print(start_pose)
        # print(self.waypoint.transform)

        # self.controller = MPCController.Controller

        self.spectator.set_transform(carla.Transform(start_pose.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

        self.player = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.bmw.grandtourer')),
            start_pose)

        self.actor_list.append(self.player)

        self.camera_rgb =RGBSensor(self.player, self.hud)
        self.actor_list.append(self.camera_rgb.sensor)

        self.camera_semseg = SegmentationCamera(self.player,self.hud)
        self.actor_list.append(self.camera_semseg.sensor)

        self.collision_sensor = CollisionSensor(self.player, self.hud)  # 충돌 여부 판단하는 센서
        self.actor_list.append(self.collision_sensor.sensor)

        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)  # lane 침입 여부 확인하는 센서
        self.actor_list.append(self.lane_invasion_sensor.sensor)

        self.gnss_sensor = GnssSensor(self.player)
        self.actor_list.append(self.gnss_sensor.sensor)

        # --------------
        # Spawn Surrounding vehicles
        # --------------


        # spawn_points = carla.Transform(start_pose.location+carla.Location(0,10,0),start_pose.rotation)  #world.get_map().get_spawn_points()
        # spawn_points = self.map.get_waypoint(spawn_points.location,lane_type=carla.LaneType.Driving)
        spawn_points = world.get_map().get_spawn_points()[85:]
        self.world.debug.draw_string(spawn_points[85+4].location, 'o', draw_shadow=True,
                                     color=carla.Color(r=255, g=255, b=255), life_time=30)
        number_of_spawn_points=len(self.world.get_map().get_spawn_points())#len(spawn_points)

        # for i in range(10,int(10+(30*self.extra_num/3)+2),30):
        #
        #     # spawn_points.append(self.waypoint.next(i)[0].transform)
        #     p1 = self.waypoint.next(i)[0].get_right_lane().transform
        #     p2 = self.waypoint.next(i)[0].get_left_lane().transform
        #     spawn_points.append(p1)
        #     spawn_points.append(p2)
        #     self.world.debug.draw_string(p1.location, 'o', draw_shadow=True,
        #                                 color=carla.Color(r=255, g=255, b=255), life_time=30)
        #
        #     self.world.debug.draw_string(p2.location, 'o', draw_shadow=True,
        #                          color=carla.Color(r=255, g=255, b=255), life_time=30)


        if self.extra_num <= number_of_spawn_points:
            # random.shuffle(spawn_points)
            pass
        elif self.extra_num > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            print(msg)
            # logging.warning(msg, self.extra_num, number_of_spawn_points)
            self.extra_num = number_of_spawn_points

        # blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        # blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]

        # print(number_of_spawn_points)
        for n, transform in enumerate(spawn_points):

            ## surrounding vehicle 차량 spawn 지점 visualize
        #     world.debug.draw_string(transform.location, 'o', draw_shadow=True,
        #                             color=carla.Color(r=255, g=255, b=255), life_time=30)

            if n >= self.extra_num:
                break
            if transform == start_pose:
                continue
            # blueprint = random.choice(blueprints)
            # if blueprint.has_attribute('color'):
            #     color = random.choice(blueprint.get_attribute('color').recommended_values)
            #     blueprint.set_attribute('color', color)
            # if blueprint.has_attribute('driver_id'):
            #     driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            #     blueprint.set_attribute('driver_id', driver_id)
            # blueprint.set_attribute('role_name', 'autopilot')
            # batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

            extra = self.world.spawn_actor(random.choice(blueprint_library.filter('vehicle.bmw.grandtourer')), transform)
            extra_pose = self.map.get_waypoint(transform.location,lane_type=carla.LaneType.Driving)
            self.extra_controller_list.append(Pure_puresuit_controller(extra, extra_pose , None , 30))  # km/h
            # extra.set_autopilot(True)
            self.extra_list.append(extra)
        self.controller = Pure_puresuit_controller(self.player, self.waypoint, self.extra_list, 50)  # km/h

        ROI=1000
 #        for actor in self.extra_list:
 #            for x in range(1, ROI + 1, int(self.waypoint.lane_width)):
 #
 #                extra_pos = actor.get_transform().location
 #                # extra_vel = actor.get_velocity()
 #                # extra_acel = actor.get_acceleration()
 #
 #                own_search_radius = ((extra_pos.x - self.waypoint.next(x)[
 #                    0].transform.location.x) ** 2 + (extra_pos.y - self.waypoint.next(x)[
 #                    0].transform.location.y) ** 2) ** 0.5
 #                right_search_radius = ((extra_pos.x - self.waypoint.next(x)[
 #                    0].get_right_lane().transform.location.x) ** 2 + (extra_pos.y - self.waypoint.next(x)[
 #                    0].get_right_lane().transform.location.y) ** 2) ** 0.5
 #                left_search_radius = ((extra_pos.x - self.waypoint.next(x)[
 #                    0].get_left_lane().transform.location.x) ** 2 + (extra_pos.y - self.waypoint.next(x)[
 #                    0].get_left_lane().transform.location.y) ** 2) ** 0.5
 #
 #
 #                try:
 #                    right_right_search_radius = ((extra_pos.x - self.waypoint.next(x)[
 #                        0].get_right_lane().get_right_lane().transform.location.x) ** 2 + (
 #                                                             extra_pos.y - self.waypoint.next(x)[
 #                                                         0].get_right_lane().get_right_lane().transform.location.y) ** 2) ** 0.5
 #                    # self.world.debug.draw_string(self.waypoint.next(x)[
 #                    #                                  0].get_right_lane().get_right_lane().transform.location, 'o',
 #                    #                              draw_shadow=True,
 #                    #                              color=carla.Color(r=255, g=255, b=255), life_time=30)
 #
 #                except AttributeError:
 #                    pass
 # ##2차선에서만 적용 됨##
 #                self.world.debug.draw_string(extra_pos, 'o', draw_shadow=True,
 #                                           color=carla.Color(r=255, g=255, b=255), life_time=1000)
 #                self.world.debug.draw_string(self.waypoint.next(x)[0].transform.location, 'o', draw_shadow=True,
 #                                             color=carla.Color(r=0, g=min((20*x),255), b=0), life_time=1000)
 #
 #                if own_search_radius <= self.waypoint.lane_width:
 #                    dl = 0
 #                    self.extra_dl_list.append(dl)
 #                    break
 #                elif right_search_radius <= self.waypoint.lane_width:
 #                    dl = 1
 #                    self.extra_dl_list.append(dl)
 #                    break
 #                elif left_search_radius <= self.waypoint.lane_width:
 #                    dl = -1
 #                    self.extra_dl_list.append(dl)
 #                    break
 #                elif right_right_search_radius <= self.waypoint.lane_width:
 #                    dl = 2
 #                    self.extra_dl_list.append(dl)
 #                    break
 #
 #            print("a")
 #            print("a") # [-1,1,0,1]

        self.extra_dl_list = [-1,1,0,-1]
        print(self.extra_dl_list)
        self.input_size = (self.extra_num)*4 + 1
        self.output_size = 3
        time.sleep(1)
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

        state = self.get_next_state()
        # if len(state) == 29:
        #     print(" ")
        return state, decision ,reward, None , done
                                        # Next State 표현 필요
    def get_next_state(self,decision=None):
        state = []
        for x, actor in enumerate(self.extra_list):
            extra_pos = actor.get_transform().location
            extra_vel = actor.get_velocity()
            extra_acel = actor.get_acceleration()
            dr = ((extra_pos.x - self.player.get_transform().location.x) ** 2 + (
                        extra_pos.y - self.player.get_transform().location.y) ** 2 + (
                              (extra_pos.z - self.player.get_transform().location.z) ** 2)) ** 0.5
            dv = ((extra_vel.x - self.player.get_velocity().x) ** 2 + (
                    extra_vel.y - self.player.get_velocity().y) ** 2 + (
                          (extra_vel.z - self.player.get_transform().location.z) ** 2)) ** 0.5
            da = ((extra_acel.x - self.player.get_acceleration().x) ** 2 + (
                    extra_acel.y - self.player.get_acceleration().y) ** 2 +
                  (extra_acel.z - self.player.get_acceleration().z) ** 2) ** 0.5


            if decision == 1:
                self.extra_dl_list[x] =self.extra_dl_list[x]+1
            elif decision == -1:
                self.extra_dl_list[x] = self.extra_dl_list[x]-1
            else:
                pass

            state.append(dr)
            state.append(dv)
            state.append(da)
            state.append(self.extra_dl_list[x])

        if decision == 1:
            self.ego_Lane +=1
        elif decision ==-1:
            self.ego_Lane +=-1
        else:
            pass
        state.append(self.ego_Lane)
        return state

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

    def main(self):
        PATH = "/home/zz/Downloads/RL_test/all.tar"

        clock = pygame.time.Clock()
        # display = pygame.display.set_mode(
        #     (self.width, self.height),
        #     pygame.HWSURFACE | pygame.DOUBLEBUF)
        Keyboardcontrol = KeyboardControl(self, False)
        # if self.is_first_time == True:
        self.agent = decision_driving_Agent(self.input_size,self.output_size , True)

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

                    if time.time()-self.simul_time>=15 and first_time_print==True:
                        print("차선 추가됨")
                        first_time_print=False
                        self.max_Lane_num =4
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

                        # if len(state)==29:
                        #     print(" ")
                        self.decision = self.agent.act(state)
                        # print(decision)
                        # if self.decision ==1 and self.max_Lane_num==self.ego_Lane:
                        #     print( " ")
                        self.decision = self.safety_check(self.decision)
                        # print("판단 :", self.decision, "차선 :", self.ego_Lane, "최대 차선 :", self.max_Lane_num)

                        is_error = self.controller.apply_control(self.decision)
                        # print("extra_controller 개수 :", len(self.extra_controller_list))
                        for i in range(len(self.extra_controller_list)):
                            self.extra_controller_list[i].apply_control()

                        self.spectator.set_transform(
                            carla.Transform(self.player.get_transform().location + carla.Location(z=50),
                                            carla.Rotation(pitch=-90)))

                        [state, decision, reward, _, done] =  self.step(self.decision)

                        if done:
                            print("epsilon :",self.agent.epsilon)
                            print("epoch : ", epoch , "누적 보상 : ",self.accumulated_reward)
                            writer.add_scalar('누적 보상 ',self.accumulated_reward,epoch )

                            if len(self.agent.buffer.size())>self.agent.batch_size:

                                for i in range(300):
                                    self.agent.learning()

                            client.set_timeout(5)
                            self.restart()
                            break

                    else:
                        self.agent.act(state)

                    clock.tick(20)




        #     # Create a synchronous mode context.
        #     with CarlaSyncMode(world, self.camera_rgb.sensor, self.camera_semseg.sensor, fps=40) as sync_mode:
        #         while True:
        #             if Keyboardcontrol.parse_events(client, self, clock):
        #                 return
        #             clock.tick()
        #
        #
        #             cnt+=1
        #             if cnt ==1000:
        #                 print('수해유 ㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠ')
        #                 decision = "right"
        #                 aa = controller.apply_control(decision)
        #             else:
        #                 aa = controller.apply_control()
        #
        #             for i in range(len(self.extra_controller_list)):
        #                 self.extra_controller_list[i].apply_control()
        #
        #             # world.debug.draw_string(aa, 'o', draw_shadow=True,
        #             #                         color=carla.Color(r=255, g=255, b=255), life_time=-1)
        #             # print(self.waypoint.transform)
        #             # print(self.player.get_transform())
        #             # print(dist)
        #
        #             self.hud.tick(self, clock)
        #             # print( clock.get_fps())
        #             # Advance the simulation and wait for the data.
        #             snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=0.1)
        #
        #             # Choose the next waypoint and update the car location.
        #             # self.waypoint = random.choice(self.waypoint.next(1.5))
        #             # self.player.set_transform(self.waypoint.transform)
        #             # self.world.debug.draw_point(self.waypoint.transform.location,size=0.1,color=carla.Color(r=255, g=255, b=255),life_time=1)
        #
        #             ## player trajactory history visualize ## -> project_to_road = false 시 차량의 궤적, True 시 차선센터 출력
        #             # curent_location=self.map.get_waypoint(self.player.get_transform().location,project_to_road=True)
        #             # world.debug.draw_string(curent_location.transform.location, 'o', draw_shadow=True,
        #             #                         color=carla.Color(r=0, g=255, b=255), life_time=100)
        #             ##이동궤적저장
        #             # self.save_traj(curent_location)
        #
        #             # self.world.debug.draw_point(ww.transform.location,size=0.1,color=carla.Color(r=255, g=255, b=255),life_time=1)
        #
        #             self.spectator.set_transform(
        #                 carla.Transform(self.player.get_transform().location + carla.Location(z=100), carla.Rotation(pitch=-90)))
        #
        #             image_semseg.convert(carla.ColorConverter.CityScapesPalette)
        #             gray_image = self.get_gray_segimg(image_semseg)
        #             # fps = round(1.0 / snapshot.timestamp.delta_seconds)
        #
        #             # Draw the display.
        #             self.draw_image(display, image_rgb)
        #             # self.draw_image(display, image_semseg, blend=False)
        #             # pygame.gfxdraw.filled_circle(display,int(self.waypoint.transform.location.x),int(self.waypoint.transform.location.y),5,(255,255,255))
        #             self.hud.render(display)
        #
        #             pygame.display.flip()

        finally:
            print('\ndestroying %d vehicles' % len(self.extra_list))
            # client.apply_batch([carla.command.DestroyActor(x) for x in self.extra_list])

            print('destroying actors.')
            for actor in self.actor_list:
                print("finally 에서 actor 제거 :", self.actor_list)
                actor.destroy()
            for extra in self.extra_list:
                print("finally 에서 actor 제거 :", self.extra_list)

                extra.destroy()

            # pygame.quit()
            print('done.')

if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    client.set_timeout(4.0)
    client.load_world('Town04')
    client.set_timeout(10.0)

    weather = carla.WeatherParameters(
        cloudyness=0.0,
        precipitation=30.0,
        sun_altitude_angle=90.0)

    world = client.get_world()
    world.set_weather(weather)
    CarlaEnv(world)




    # except KeyboardInterrupt:
        # print('\nCancelled by user. Bye!')
