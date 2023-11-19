import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY, ActionRot


class SOCIAL_FORCE(Policy):
    def __init__(self):
        super().__init__()
        self.A = 2.
        self.B = 1
        self.KI = 1
        self.name = 'social_force'
        self.kinematics = 'holonomic'
        self.trainable = False
        self.multiagent_training = None
        self.safety_space = 0
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 5
        self.time_horizon_obst = 5
        self.radius = 0.3
        self.max_speed = 1
        self.sim = None

    def predict(self, state, deterministic):
        """
        Produce action for agent with circular specification of social force model.
        """
        # Pull force to goal
        delta_x = state.self_state.gx - state.self_state.px
        delta_y = state.self_state.gy - state.self_state.py
        dist_to_goal = np.sqrt(delta_x ** 2 + delta_y ** 2)
        desired_vx = (delta_x / dist_to_goal) * state.self_state.v_pref
        desired_vy = (delta_y / dist_to_goal) * state.self_state.v_pref
        KI = self.KI  # Inverse of relocation time K_i
        curr_delta_vx = KI * (desired_vx - state.self_state.vx)
        curr_delta_vy = KI * (desired_vy - state.self_state.vy)
        # print(state.self_state.position)
        # Push force(s) from other agents
        A = self.A  # Other observations' interaction strength: 1.5
        B = self.B  # Other observations' interaction range: 1.0
        interaction_vx = 0
        interaction_vy = 0
        for other_human_state in state.human_states:
            delta_x = state.self_state.px - other_human_state.px
            delta_y = state.self_state.py - other_human_state.py
            dist_to_human = np.sqrt(delta_x ** 2 + delta_y ** 2)
            interaction_vx += A * np.exp((state.self_state.radius + other_human_state.radius - dist_to_human) / B) * (
                    delta_x / dist_to_human)
            interaction_vy += A * np.exp((state.self_state.radius + other_human_state.radius - dist_to_human) / B) * (
                    delta_y / dist_to_human)

        # Sum of push & pull forces
        total_delta_vx = (curr_delta_vx + interaction_vx) * self.time_step
        total_delta_vy = (curr_delta_vy + interaction_vy) * self.time_step

        # clip the speed so that sqrt(vx^2 + vy^2) <= v_pref
        new_vx = state.self_state.vx + total_delta_vx
        new_vy = state.self_state.vy + total_delta_vy
        act_norm = np.linalg.norm([new_vx, new_vy])
        self.last_state = state
        if act_norm > state.self_state.v_pref:
            return ActionXY(new_vx / act_norm * state.self_state.v_pref, new_vy / act_norm * state.self_state.v_pref)
        else:
            return ActionXY(new_vx, new_vy)
