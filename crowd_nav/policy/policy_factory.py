import sys

from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.sac_rl import USAC
from crowd_nav.policy.sac_lstm import RNNSAC
from crowd_nav.policy.sarl_sac import SARLSAC
from crowd_nav.policy.tf_sac import TFSAC
from crowd_nav.policy.stftrl import TFRNNSAC
from crowd_nav.policy.crossTFRL import STRL
from crowd_nav.policy.PER_SAC_STRL import PER_STRL
from crowd_sim.envs.policy.policy_factory import policy_factory

sys.path.append('..')
policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['sac_rl'] = USAC
policy_factory['rnnsac'] = RNNSAC
policy_factory['sarl_sac'] = SARLSAC
policy_factory['tf_sac'] = TFSAC
policy_factory['tf_rnn_sac'] = TFRNNSAC
policy_factory['STRL'] = STRL
policy_factory['PER_STRL'] = PER_STRL
