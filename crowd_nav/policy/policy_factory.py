import sys

from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.sac_lstm import RNNSAC
from crowd_nav.policy.sarl_sac import SARLSAC

from crowd_sim.envs.policy.policy_factory import policy_factory

sys.path.append('..')
policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['rnnsac'] = RNNSAC
policy_factory['sarl_sac'] = SARLSAC

