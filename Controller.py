import carla
import math
import numpy as np
import time
import random
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


class Pure_puresuit_controller:

    def __init__(self,player,waypoint,extra_actors=None,desired_vel=40):
        self.player = player
        self.world = self.player.get_world()
        self.waypoint = waypoint
        self.desired_vel = desired_vel
        self.velocity = 0

        self.error_v_pre = 0
        self.error_v = 0
        self.error_v_dot = 0
        self.error_v_int = 0

        self.error_acc_pre = 0
        self.error_acc = 0
        self.error_acc_dot = 0
        self.error_acc_int = 0

        # self.error_pre = 0
        # self.error = 0
        # self.error_dot = 0
        # self.error_int = 0

        self.k_v = [0.5, -0.004, 0.003] #60km/h : [0.03, 0.004, 0.003] #kp, kd, ki
        self.k_acc = [0.2, -0.001, 0.001]
        self.cnt = 0
        self.t = time.time()
        self.pos_pre = self.player.get_location()
        self.pos = None
        self.ld = 0
        self.heading = None
        self.extra_actors = extra_actors
        self.safe_distance = 10
        self.leading_vehicle = None
        self.search_radius = None

    def apply_control(self,decision=None):
        dt = time.time() - self.t
        # print(1/dt)
        ## waypoint update ##
        if self.ld < self.velocity/15:
            try:
                self.waypoint =self.waypoint.next(10)[0]

            except:
                print("직진 차선, waypoint 존재 x")
                return -1
            self.world.debug.draw_string(self.waypoint.transform.location, 'o', draw_shadow=True,
                                     color=carla.Color(r=255, g=255, b=255), life_time=1)
        if decision == 1:
            print("차오른쪽 차선 변경 수행")
            self.leading_vehicle = None
            self.player.set_autopilot(False)
            try:
                self.waypoint = self.waypoint.next(25)[0]
                self.waypoint = self.waypoint.get_right_lane()
            except:
                print("오른쪽 판단, waypoint 존재 x")
                return -1
        elif decision == -1:
            print("왼쪽 차선 변경 수행")
            self.leading_vehicle = None
            self.player.set_autopilot(False)
            try:
                self.waypoint = random.choice(self.waypoint.next(25))
                self.waypoint = self.waypoint.get_left_lane()
            except:
                print("왼쪽 판단, waypoint 존재 x")
                return -1

        if self.player.is_alive:
            self.pos = self.player.get_location()
            self.heading = self.pos- self.pos_pre
            self.pos_pre = self.pos
        else:
            print("이미 죽은 actor")
            return -1
        # print(self.heading)

        ## lateral control ##
        self.ld = ((self.pos.x-self.waypoint.transform.location.x)**2+(self.pos.y-self.waypoint.transform.location.y)**2+(self.pos.z-self.waypoint.transform.location.z)**2)**0.5
        eld = -(self.heading.y*self.waypoint.transform.location.x-self.heading.x*self.waypoint.transform.location.y+self.heading.x*self.pos.y-self.heading.y*self.pos.x)/(self.heading.x**2+self.heading.y**2+0.001)**0.5
        self.velocity = (self.player.get_velocity().x**2+self.player.get_velocity().y**2+self.player.get_velocity().z**2)**0.5*3.6
        # km/h

        k = 3
        car_length = 3
        steer = math.atan2(2 * car_length * eld, k*self.velocity+0.01) * 180 / math.pi * 1 / 59  # 59 -> 1
        # print("eld : ", eld, "ld : ", self.ld, "heading :", self.heading, "steer : ",steer)


        ## 전방 차량 시각화 ##
        # if self.leading_vehicle is not None:
        #     self.world.debug.draw_string(self.leading_vehicle.량get_transform().location, '전방 차량', draw_shadow=True,
        #                                  color=carla.Color(r=255, g=255, b=255), life_time=1)

        if self.leading_vehicle == None   : #전방 차량이 없거나 없어졌을 때 다시 전방 차량을 찾아줌. 없으면 None값으로 초기화
            if self.extra_actors is not None: #ego vehicle 만 수행
                for actor in self.extra_actors:
                    extra_pos = actor.get_transform().location
                    try:
                        for x in range(1, self.safe_distance+1-int(self.waypoint.lane_width), int(self.waypoint.lane_width)):
                            # self.world.debug.draw_string(self.waypoint.next(x)[0].transform.location, 'o', draw_shadow=True,
                            #                         color=carla.Color(r=255, g=255, b=255), life_time=1)
                            self.search_radius = ((extra_pos.x - self.waypoint.next(x)[
                                0].transform.location.x) ** 2 + (extra_pos.y - self.waypoint.next(x)[
                                0].transform.location.y) ** 2) ** 0.5
                            if self.search_radius <= self.waypoint.lane_width:
                                # print("추종 시작")
                                self.leading_vehicle = actor
                                break
                    except IndexError:
                        print("IndexError 발생")
                        pass
        else:
            self.error_acc = ((self.leading_vehicle.get_transform().location.x - self.player.get_transform().location.x) ** 2 + (
                                        self.leading_vehicle.get_transform().location.y - self.player.get_transform().location.y) ** 2 + (
                                        self.leading_vehicle.get_transform().location.z - self.player.get_transform().location.z) ** 2) ** 0.5  \
                                       - self.safe_distance
            if self.error_acc > 0:
                self.leading_vehicle = None

        # if self.leading_vehicle == None:
        #     # 종방향 제어 에러
        #     self.error_v_pre = self.error_v
        #     self.error_v = self.desired_vel - self.velocity
        #     self.error_v_dot = (self.error_v - self.error_v_pre) / dt
        #     if self.error_v <0.1:
        #         self.cnt =1
        #     if self.cnt !=1:
        #         self.error_v_int += self.error_v * dt
        #     else:
        #         pass
        #
        #         # print("종방향 돔")
        #     self.error = [self.error_v, self.error_v_dot, self.error_v_int ]
        #     # print("V error : " ,self.error[0])
        #
        #     self.control_input(self.k_v,self.error,steer)
        # else: ## 거리로 ACC 제어 ##
        #     # print("ACC")
        #     self.error_v_pre = self.error_v
        #
        #     self.error_acc_pre = self.error_acc
        #     self.error_acc = ((self.leading_vehicle.get_transform().location.x - self.player.get_transform().location.x) ** 2 + (
        #             self.leading_vehicle.get_transform().location.y - self.player.get_transform().location.y) ** 2 + (
        #             self.leading_vehicle.get_transform().location.z - self.player.get_transform().location.z) ** 2) ** 0.5  \
        #                      - self.safe_distance
        #
        #     self.error_acc_dot = (self.error_acc - self.error_acc_pre) / dt
        #     self.error_acc_int += self.error_acc * dt
        #     if self.error_acc < 0.5:
        #         self.cnt = 2
        #     if self.cnt != 2:
        #         self.error_v_int += self.error_v * dt
        #     self.error = [self.error_acc, self.error_acc_dot, self.error_acc_int ]
        #     print("ACC error : ", self.error[0])
        #     self.control_input(self.k_acc, self.error, steer)
        #
        #     self.error_v = self.desired_vel - self.velocity
        if self.leading_vehicle is None:
            self.error_v_pre = self.error_v
            self.error_v = self.desired_vel - self.velocity
            self.error_v_dot = (self.error_v - self.error_v_pre) / dt
            if self.error_v < 0.1:
                self.cnt = 1
            if self.cnt != 1:
                self.error_v_int += self.error_v * dt
            else:
                pass

                # print("종방향 돔")
            self.error = [self.error_v, self.error_v_dot, self.error_v_int]
            # print("V error : " ,self.error[0])

            self.control_input(self.k_v, self.error, steer)
        else:
            self.error_v_pre = self.error_v
            self.error_v = 30 - self.velocity
            self.error_v_dot = (self.error_v - self.error_v_pre) / dt
            if self.error_v < 0.1:
                self.cnt = 1
            if self.cnt != 1:
                self.error_v_int += self.error_v * dt
            else:
                pass

                # print("종방향 돔")
            self.error = [self.error_v, self.error_v_dot, self.error_v_int]
            # print("V error : " ,self.error[0])

            self.control_input(self.k_v, self.error, steer)


        # print(self.waypoint.section_id)
        # print(self.waypoint.lane_width)
        self.t = time.time()

    def control_input(self,k,error,steer):

        if error[0]>0:
            a = k[0] * error[0] + k[1] * error[1] + k[2] * error[2]
            a =min(a,1)
            self.player.apply_control(carla.VehicleControl(throttle=a, steer=steer, brake=0.0))


        else:

            b = k[0] * error[0] + k[1] * error[1] + k[2] * error[2]
            # print(error[0])


            b = min(b, 1)
            self.player.apply_control(carla.VehicleControl(throttle=0, steer=steer, brake=b))
