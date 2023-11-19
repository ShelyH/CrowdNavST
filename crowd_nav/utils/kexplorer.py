import logging

import torch
from tensorboardX import SummaryWriter

from crowd_nav.utils.utils_sac import fullstateTotensor, transform_h, transform_rh, Transition


def kexplorer(num_updates, is_rl, env, robot, policy, env_config, args, up):
    training_step = 0
    base_dir = args.log_path + args.env_id + "/ORCA_exp{}".format(args.seed)
    print(base_dir)
    writer = SummaryWriter(base_dir)
    for i in range(num_updates):
        ob = env.reset('train')
        done = False
        tol_reward = 0
        while not done:
            action = robot.act(ob, args.deterministic)
            # print(action)
            fullstate = robot.return_state()
            fullstate = fullstateTotensor(fullstate)

            state = transform_h(ob)
            state = torch.cat((state, fullstate))
            # print(env_config.getfloat('robot', 'v_pref'))
            # actiontoObj = robot.clip_action(action, env_config.getfloat('robot', 'v_pref'))
            # print('1', state)
            # actiontoObj = ActionXY(action[0][0], action[0][1])
            # print(action)
            ob, reward, done, info, state_ = env.step(action)

            state_ = transform_rh(state_)
            up.replay_pool.push(Transition(state, action, reward, state_, done))
            # print('2', state_)
            tol_reward += reward
            # env.render()
            if done:
                writer.add_scalar("reward", scalar_value=tol_reward, global_step=training_step)
                if is_rl:
                    # print(up.replay_pool._size)

                    if up.replay_pool._size > 10000:
                        q1_loss, q2_loss, pi_loss, alpha_loss = up.optim_sac()
                        writer.add_scalar("loss/pi_loss", scalar_value=pi_loss, global_step=training_step)
                        writer.add_scalar("loss/alpha_loss", scalar_value=alpha_loss, global_step=training_step)
                        writer.add_scalar("loss/q1_loss", scalar_value=q1_loss, global_step=training_step)
                        writer.add_scalar("loss/q2_loss", scalar_value=q2_loss, global_step=training_step)
                    # writer.add_scalar("reward", scalar_value=tol_reward, global_step=training_step)
                    logging.info('episode:{}, reward:{}, memory size:{}, time:{},'
                                 ' info:{}'.format(i, ('%.2f' % tol_reward), up.replay_pool._size,
                                                   ('%.2f' % env.global_time), info))
                    if i % args.save_interval == 0 or i == num_updates - 1:
                        policy.save_load_model('save', args.output_dir)
                else:

                    logging.info('episode:{}, reward:{}, memory size:{}, time:{},'
                                 ' info:{}'.format(i, ('%.2f' % tol_reward), up.replay_pool._size,
                                                   ('%.2f' % env.global_time), info))
                training_step += 1
                break
            # env.render(update=True)
        # print(up.replay_pool._size)
        # if i == 500:
        #     break
