import numpy as np

from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Robot(Agent):
    def __init__(self, config, section):
        super(Robot, self).__init__(config, section)
        self.state = None
        self.config = config

    def act(self, ob, deterministic):
        # print(ob)
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        self.state = JointState(self.get_full_state(), ob)
        # print(self.policy)
        action = self.policy.predict(self.state, deterministic)
        # print(action)

        return action

    def return_state(self):
        return self.get_full_state()

    def clip_action(self, raw_action, v_pref, prev_v, time_step, a_pref=1.):
        """
        Input state is the joint state of robot concatenated by the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """
        # quantize the action
        holonomic = True if self.kinematics == 'holonomic' else False
        # clip the action
        if isinstance(raw_action, ActionXY) or isinstance(raw_action, ActionRot):
            return raw_action
        else:
            raw_action = raw_action[0].cpu().numpy()
            if holonomic:
                raw_action = np.array(raw_action)
                # clip acceleration
                # a_norm = np.linalg.norm(raw_action - prev_v)
                # a_norm = np.linalg.norm((raw_action - prev_v))
                # if a_norm > a_pref:
                #     v_action = np.zeros(2)
                #     raw_ax = raw_action[0] - prev_v[0]
                #     raw_ay = raw_action[1] - prev_v[1]
                #     v_action[0] = (raw_ax / a_norm * a_pref) * time_step + prev_v[0]
                #     v_action[1] = (raw_ay / a_norm * a_pref) * time_step + prev_v[1]
                # else:
                #     v_action = raw_action
                # print(ActionXY(raw_action[0], raw_action[1]))
                # clip velocity
                v_norm = np.linalg.norm(raw_action)
                # v_pref = 0.5
                if v_norm > v_pref:
                    raw_action[0] = raw_action[0] / v_norm * v_pref
                    raw_action[1] = raw_action[1] / v_norm * v_pref
                    # print(raw_action[0], raw_action[1])
                return ActionXY(raw_action[0], raw_action[1])
            else:
                # for sim2real
                # raw_action[0] = np.clip(raw_action[0], 0, 1)  # action[0] is change of v
                raw_action[0] = (raw_action[0] + 1) / 2  # action[0] is change of v
                raw_action[1] = np.clip(raw_action[1], -1, 1)
                # raw[0, 1] = np.clip(raw[0, 1], -0.25, 0.25) # action[1] is change of w
                # action[0] is v
                # raw_action[1] = np.clip(raw_action[1], -0.25, 0.25)  # action[1] is change of theta
                # print(raw_action)
                return ActionRot(raw_action[0], raw_action[1])
