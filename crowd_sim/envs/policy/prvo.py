# import numpy as np
#
# from crowd_sim.envs.policy.rvo.vector import Vector2
# # from rvo.vector import Vector2
# from crowd_sim.envs.policy.rvo.simulator import Simulator
# from crowd_sim.envs.policy.policy import Policy
# from crowd_sim.envs.utils.action import ActionXY, ActionRot
#
#
# class ORCA(Policy):
#     def __init__(self):
#         super().__init__()
#         self.name = 'ORCA'
#         self.simulator_ = None
#         self.trainable = False
#         self.multiagent_training = None
#         self.kinematics = 'holonomic'
#         self.safety_space = 0
#         self.neighbor_dist = 10
#         self.max_neighbors = 10
#         self.time_horizon = 5
#         self.time_horizon_obst = 5
#         self.radius = 0.3
#         self.max_speed = 1
#         self.goals_ = []
#
#     def configure(self, config):
#         return
#
#     def set_phase(self, phase):
#         return
#
#     def predict(self, state, deterministic):
#         self_state = state.self_state
#
#         if self.simulator_ is None:
#             self.simulator_ = Simulator()
#             self.simulator_.set_time_step(self.time_step)
#             # neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed, velocity
#             self.simulator_.set_agent_defaults(self.neighbor_dist, self.max_neighbors, self.time_horizon,
#                                                self.time_horizon_obst, self.radius, self.max_speed, Vector2(0.0, 0.0))
#             # print(self_state.position)
#             x, y = self_state.position
#             # print(x,y)
#             self.simulator_.add_agent(Vector2(x, y))
#
#             for other_human_state in state.human_states:
#                 px, py = other_human_state.position
#                 self.simulator_.add_agent(Vector2(px, py))
#                 # self.goals_.append(other_human_state.position)
#
#         velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
#         speed = np.linalg.norm(velocity)
#         pref_vel = velocity / speed if speed > 1 else velocity
#         # self.simulator_.agents_
#         for i, human_state in enumerate(state.human_states):
#             # unknown goal position of other humans
#             print(pref_vel)
#             self.simulator_.set_agent_pref_velocity(i, pref_vel)
#         self.simulator_.step()
#         # print(self.simulator_.agents_[0].velocity_)
#         action = ActionXY(self.simulator_.agents_[0].velocity_.x, self.simulator_.agents_[0].velocity_.y)
#         print(action)
#         return action
