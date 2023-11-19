from crowd_sim.envs.policy.linear import Linear
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.social_force import SOCIAL_FORCE


def none_policy():
    return None


policy_factory = dict()
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['social_force'] = SOCIAL_FORCE
policy_factory['none'] = none_policy
