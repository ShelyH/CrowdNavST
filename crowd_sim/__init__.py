from gym.envs.registration import register

register(
    id='CrowdSim-v0',
    entry_point='crowd_sim.envs:CrowdSim',
)
register(
    id='CrowdSim-v1',
    entry_point='crowd_sim.envs:Crowdsim2real',
)
register(
    id='CrowdSim-v2',
    entry_point='crowd_sim.envs:Crowdsim2real_V2',
)
register(
    id='CrowdSimstemporal-v0',
    entry_point='crowd_sim.envs:CrowdSim_Stemporal',
)
