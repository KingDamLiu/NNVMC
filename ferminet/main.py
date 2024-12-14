# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core training loop for neural QMC in JAX."""

from ferminet.init import *

def write_status(cfg, status):
    with open(cfg.log.save_path+'/status.yaml', 'w') as f:
        yaml.dump(status, f)

def read_status(cfg):
    with open(cfg.log.save_path+'/status.yaml', 'r') as f:  
        status = yaml.load(f, Loader=yaml.FullLoader)
    return status

def pre_train(cfg: ml_collections.ConfigDict, status):
    cfg.system.states = cfg.system.states_total
    # 初始化配置和设备
    (cfg, key, num_devices, num_hosts, num_states, host_batch_size,
     total_host_batch_size, device_batch_size, data_shape, atoms,
     charges, nspins, batch_atoms, batch_charges, core_electrons, ecp) = init_config(cfg)
    
    ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path+'/pretrain/')

    # 初始化波函数和网络
    (params, network, networks_list_, signed_network, logabs_network,
        batch_network, log_network, batch_network_mcmc, log_network_for_loss, key) = init_wavefunction(cfg, key, charges, nspins, core_electrons, ecp)

    # 初始化参考波函数
    hartree_fock = init_reference_wavefunction(cfg, key, charges, nspins, core_electrons, ecp)

    # 初始化数据和检查点
    data = init_data(cfg, key, data_shape, total_host_batch_size, batch_atoms,
              batch_charges, core_electrons)
    
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)

    # 预训练
    states=cfg.system.states
    batch_orbitals = jax.vmap(
        network.orbitals, in_axes=(None, 0, 0, 0, 0), out_axes=0
    )
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    optimizer = optax.adam(3.e-4)
    opt_state_pt = constants.pmap(optimizer.init)(params)

    pretrain_step = pretrain.make_pretrain_step(
        batch_orbitals,
        batch_network,
        optimizer.update,
        mcmc_target = cfg.mcmc.target,
        electrons=cfg.system.electrons,
        batch_size=device_batch_size,
        full_det=network.options.full_det,
        scf_fraction=cfg.pretrain.get('scf_fraction', 0.0),
        states=cfg.system.states,
    )
    pretrain_step = constants.pmap(pretrain_step)

    if cfg.mcmc.target == 'Psi':
        pass
    elif cfg.mcmc.target == 'diag':
        positions = data.positions
        atoms=data.atoms
        charges=data.charges
        # batch_spins = jnp.tile(data.spins[0, 0][None], [positions.shape[1], 1])
        # pmap_spins = kfac_jax.utils.replicate_all_local_devices(batch_spins)
        data = networks.FermiNetData(
            positions=jnp.reshape(positions, positions.shape[:2]+(states, -1,)), spins=jnp.reshape(data.spins, positions.shape[:2]+(states, -1,)), atoms=atoms, charges=charges
        )
    # positions = data.positions
    # atoms=data.atoms
    # charges=data.charges
    # spins = data.spins[0,0]

    # batch_spins = jnp.tile(spins[None], [positions.shape[1], 1])
    # pmap_spins = kfac_jax.utils.replicate_all_local_devices(batch_spins)
    # if cfg.mcmc.target == 'Psi':
    #     data = networks.FermiNetData(
    #         positions=positions, spins=pmap_spins, atoms=atoms, charges=charges
    #     )
    # elif cfg.mcmc.target == 'diag':
    #     data = networks.FermiNetData(
    #         positions=jnp.reshape(positions, positions.shape[:2]+(states, -1,)), spins=jnp.reshape(pmap_spins, positions.shape[:2]+(states, -1,)), atoms=atoms, charges=charges
    #     )
    
    # 增加日志文件
    out_file_name = os.path.join(ckpt_save_path, 'pretrain_stats'+'.csv')
    start_time = time.time()
    for t in range(cfg.pretrain.iterations):
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        data, params, opt_state_pt, loss, pmove = pretrain_step(
            data, params, opt_state_pt, subkeys, hartree_fock)
        writer_kwargs = dict(
            loss=loss[0],
            pmove=pmove[0],
            dt = time.time() - start_time,
        )
        start_time = time.time()
        if t % 100 == 0:
            logging.info('Pretrain iter %05d: %g %g', t, loss[0], pmove[0])
        if t>0 or os.path.exists(out_file_name):
            df = pd.DataFrame(writer_kwargs, index=[t])
            df.to_csv(out_file_name, mode='a', header=False)
        elif t == 0:
            df = pd.DataFrame(writer_kwargs, index=[t])
            df.to_csv(out_file_name, header=True, mode = 'w')

    # 保存参数
    checkpoint.save(ckpt_save_path, t, data, params, opt_state=None, mcmc_width=None) 
    status['pretrain']['status'] = True
    status['pretrain']['Epochs'] = t
    write_status(cfg, status)

def ipf_train(cfg: ml_collections.ConfigDict):
    continue_train = False
    for i in range(cfg.system.states_total):
        # 读取状态文件
        with open(cfg.log.save_path+'/status.yaml', 'r') as f:  
            status = yaml.load(f, Loader=yaml.FullLoader)
        # 查找是否已经完成预训练
        if status['pretrain']['status'] == False:
            pre_train(cfg, status)
        else:
            logging.info('Pretrain has been completed, skip pretrain.')
            # 查找当前训练状态
            for i, key in enumerate(status['vmctrain'].keys()):
                if status['vmctrain'][key]['status'] == False:
                    if status['vmctrain'][key]['Epochs'] == 0:
                        # 从当前能级开始训练
                        cfg.system.states = i+1
                        cfg.system.states_update = 1
                        cfg.log.restore_path = cfg.log.save_path+'/vmctrain/states_{}/'.format(i-1)
                        break
                    elif status['vmctrain'][key]['Epochs'] < cfg.optim.iterations:
                        # 从当前能级继续训练
                        cfg.system.states = i
                        cfg.system.states_update = 1
                        cfg.log.restore_path = cfg.log.save_path+'/vmctrain/states_{}/'.format(i)
                        continue_train = True
                        break

        # 初始化配置和设备
        (cfg, key, num_devices, num_hosts, num_states, host_batch_size,
         total_host_batch_size, device_batch_size, data_shape, atoms,
         charges, nspins, batch_atoms, batch_charges, core_electrons, ecp) = init_config(cfg)

        # 初始化波函数和网络
        (params, network, network_list, signed_network, logabs_network,
        batch_network, log_network, batch_network_mcmc, log_network_for_loss, key) = init_wavefunction(cfg, key, charges, nspins, core_electrons, ecp)

        # 加载预训练初始化参数
        ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path+'/pretrain/')
        ckpt_restore_filename = checkpoint.find_last_checkpoint(ckpt_save_path)
        if ckpt_restore_filename:
            logging.info('Restoring parameters from %s', ckpt_restore_filename)
            (t_init,
            data,
            data_init,
            opt_state_ckpt,
            mcmc_width_ckpt,
            density_state_ckpt) = checkpoint.restore(
                ckpt_restore_filename, host_batch_size)
            
            for k in params.keys():
                params[k] = params_init[k]
            
            t_init = 0
            opt_state_ckpt = None
            mcmc_width_ckpt = None
            density_state_ckpt = None

        # 加载已有波函数参数
        ckpt_restore_filename = checkpoint.find_last_checkpoint(checkpoint.create_save_path(cfg.log.restore_path))
        if ckpt_restore_filename:
            logging.info('Restoring parameters from %s', ckpt_restore_filename)
            t_init, data_last, params_init, opt_state_ckpt, mcmc_width_ckpt, density_state_ckpt = checkpoint.restore(ckpt_restore_filename, host_batch_size)
            # 覆盖参数和数据
            for k in data_last.keys():
                params[k] = data_last[k]

            if continue_train==False:
                t_init = 0
                opt_state_ckpt = None
                mcmc_width_ckpt = None
                density_state_ckpt = None

        ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path+'/vmctrain/states_{}/'.format(i))

        data.positions = jnp.concatenate([data_init.positions, data.positions[:, :, cfg.system.states-1:cfg.system.states, :]], axis=2)
        data.spins = jnp.concatenate([data_init.spins, data.spins[:, :, cfg.system.states-1:cfg.system.states, :]], axis=2) 

        # 初始化日志和观测量
        (train_schema, observable_fns, observable_states, energy_matrix_file,
         energy_matrix_std_file, energy_matrix_clip_file,
         energy_matrix_clip_std_file, s2_matrix_file, dipole_matrix_file,
         density_matrix_file, sharded_key, density_update) = init_log(
            cfg, ckpt_save_path, signed_network, nspins, data,
            density_state_ckpt, key)

        # 初始化MCMC步骤
        mcmc_step = init_mcmc_step(cfg, batch_network_mcmc, device_batch_size, atoms, num_states)

        # 初始化局部能量函数
        local_energy_fn = init_local_energy_fn(cfg, signed_network, network_list, charges, nspins, cfg.network.get('complex', False))

        # 初始化损失函数
        evaluate_loss = init_evaluate_loss(cfg, log_network, network_list, log_network_for_loss, logabs_network, signed_network, local_energy_fn, cfg.network.get('complex', False))

        # 初始化学习率调度
        learning_rate_schedule = init_learning_rate_schedule(cfg)

        # 初始化优化器
        optimizer, opt_state, sharded_key = init_optimizer(cfg, evaluate_loss, learning_rate_schedule, params,
                                                       sharded_key, data, opt_state_ckpt)

        # 初始化变量
        time_of_last_ckpt = time.time()
        weighted_stats = None
        num_resets = 0  # 用于 reset_if_nan 为真时
        if mcmc_width_ckpt is not None:
            mcmc_width = kfac_jax.utils.replicate_all_local_devices(mcmc_width_ckpt[0])
        else:
            mcmc_width = kfac_jax.utils.replicate_all_local_devices(
                jnp.asarray(cfg.mcmc.move_width))
        pmoves = np.zeros(cfg.mcmc.adapt_frequency)
        if cfg.mcmc.target == 'diag':
            data.positions = jnp.reshape(data.positions, data_shape + (cfg.system.states, -1,))
            data.spins = jnp.reshape(data.spins, data_shape + (cfg.system.states, -1,))
        else:
            data.positions = jnp.reshape(data.positions, data_shape + (-1,))
            data.spins = jnp.reshape(data.spins, data_shape + (-1,))
        # 执行burn-in步骤
        if t_init == 0:
            logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)
            burn_in_step = make_training_step(
                mcmc_step=mcmc_step, optimizer_step=null_update)
            for _ in range(cfg.mcmc.burn_in):
                sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
                data, params, *_ = burn_in_step(
                    data,
                    params,
                    state=None,
                    key=subkeys,
                    mcmc_width=mcmc_width)
            logging.info('Completed burn-in MCMC steps')
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            ptotal_energy = constants.pmap(evaluate_loss)
            initial_energy, _ = ptotal_energy(params, subkeys, data)
            logging.info('Initial energy: %03.4f E_h', initial_energy[0])

        # 设置训练步骤函数
        if not optimizer:
            step = make_training_step(
                mcmc_step=mcmc_step,
                optimizer_step=make_loss_step(evaluate_loss))
        elif isinstance(optimizer, optax.GradientTransformation):
            step = make_training_step(
                mcmc_step=mcmc_step,
                optimizer_step=make_opt_update_step(evaluate_loss, optimizer),
                reset_if_nan=cfg.optim.reset_if_nan)
        elif isinstance(optimizer, kfac_jax.Optimizer):
            step = make_kfac_training_step(
                mcmc_step=mcmc_step,
                damping=cfg.optim.kfac.damping,
                optimizer=optimizer,
                reset_if_nan=cfg.optim.reset_if_nan)
        else:
            raise ValueError(f'Unknown optimizer: {optimizer}')

        # 主训练循环
        out_file_name = os.path.join(ckpt_save_path, 'train_stats'+ datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S') + '.csv')
        start_time = time.time()

        E_means = []
        E_stds = []
        for t in range(t_init, cfg.optim.iterations):
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            data, params, opt_state, stats, pmove = step(
                data,
                params,
                opt_state,
                subkeys,
                mcmc_width)

            # 处理统计信息和日志记录
            loss = stats['loss'][0]
            aux_data = stats['aux']
            weighted_stats = statistics.exponentialy_weighted_stats(
                alpha=0.1, observation=loss, previous_stats=weighted_stats)
            pmove = pmove[0]

            # 更新观测量
            observable_data = {
                key: fn(params, data, observable_states[key])
                for key, fn in observable_fns.items()
            }
            if cfg.observables.density:
                sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
                observable_states['density'] = density_update(
                    subkeys, params, data, observable_states['density'])

            # 更新MCMC移动宽度
            mcmc_width, pmoves = mcmc.update_mcmc_width(
                t, mcmc_width, cfg.mcmc.adapt_frequency, pmove, pmoves)

            # 检查NaN并处理
            if cfg.debug.check_nan:
                tree = {'params': params, 'loss': loss}
                if cfg.optim.optimizer != 'none':
                    tree['optim'] = opt_state
                try:
                    chex.assert_tree_all_finite(tree)
                    num_resets = 0
                except AssertionError as e:
                    if cfg.optim.reset_if_nan:
                        num_resets += 1
                        if num_resets > 100:
                            raise e
                    else:
                        raise e

            # 日志记录
            if t % cfg.log.stats_frequency == 0:
                logging_str = ('Step %05d: '
                            '%03.4f s, %03.4f E_h, %03.4f E_h^2, exp. variance=%03.4f E_h^2, pmove=%0.2f')
                logging_args = t, time.time()-start_time, loss, aux_data.variance[0].sum(), weighted_stats.variance, pmove
                writer_kwargs = dict(
                    step =  t,
                    time = time.time()-start_time,
                    energy = np.asarray(loss),
                    var = np.asarray(aux_data.variance[0].sum()),
                    ewmean = np.asarray(weighted_stats.mean),
                    ewvar = np.asarray(weighted_stats.variance),
                    pmove = np.asarray(pmove),
                    grad_norm = stats['grad_norm'][0],
                    learning_rate = stats['learning_rate'][0],
                    momentum = stats['momentum'][0],
                    param_norm = stats['param_norm'][0],
                    precon_grad_norm = stats['precon_grad_norm'][0],
                    update_norm = stats['update_norm'][0],
                )
                start_time = time.time()
                for key in observable_data:
                    obs_data = observable_data[key]
                    if cfg.system.states:
                        obs_data = np.trace(obs_data, axis1=-1, axis2=-2)
                    if key == 'dipole':
                        writer_kwargs['mu_x'] = obs_data[0]
                        writer_kwargs['mu_y'] = obs_data[1]
                        writer_kwargs['mu_z'] = obs_data[2]
                    elif key == 'density':
                        pass
                    elif key == 's2':
                        writer_kwargs[key] = obs_data
                        logging_str += ', <S^2>=%03.4f'
                        logging_args += obs_data,
                logging.info(logging_str, *logging_args)
                if t>0 or os.path.exists(out_file_name):
                    df = pd.DataFrame(writer_kwargs, index=[0])
                    df.to_csv(out_file_name, mode='a', header=False)
                elif t == 0:
                    df = pd.DataFrame(writer_kwargs, index=[0])
                    df.to_csv(out_file_name, header=True, mode = 'w')

            # 处理过大的观测量数据，直接保存为.npy文件
            if cfg.system.states:
                if cfg.run_model != 'test':
                    energy_matrix = aux_data.local_energy_mat
                    energy_matrixs = energy_matrix.reshape((cfg.batch_size,)+energy_matrix.shape[2:])
                    energy_matrixs_clip = clip_local_energys(energy_matrixs)
                    energy_matrix = np.nanmean(energy_matrixs, axis=0)
                    energy_matrix_std = np.nanstd(energy_matrixs, axis=0)
                    energy_matrix_clip = np.nanmean(energy_matrixs_clip, axis=0)
                    energy_matrix_clip_std = np.nanstd(energy_matrixs_clip, axis=0)
                    np.save(energy_matrix_file, energy_matrix)
                    np.save(energy_matrix_std_file, energy_matrix_std)
                    np.save(energy_matrix_clip_file, energy_matrix_clip)
                    np.save(energy_matrix_clip_std_file, energy_matrix_clip_std)
                else:
                    energy_matrix = aux_data.local_energy_mat
                    np.save(energy_matrix_file, energy_matrix)
                if cfg.observables.s2:
                    np.save(s2_matrix_file, observable_data['s2'])
                if cfg.observables.dipole:
                    np.save(dipole_matrix_file, observable_data['dipole'])
            if cfg.observables.density:
                np.save(density_matrix_file, observable_data['density'])

            E_means.append(energy_matrix_clip)
            E_stds.append(energy_matrix_clip_std)
            if len(E_means) > 100:
                E_means.pop(0)
                E_stds.pop(0)

            # 检查点保存
            if time.time() - time_of_last_ckpt > cfg.log.save_frequency * 60:
                checkpoint.save(ckpt_save_path, t, data, params, opt_state, mcmc_width)
                time_of_last_ckpt = time.time()
                status['vmctrain']['states_{}'.format(cfg.system.states-1)]['Epochs'] = t
                with open(cfg.log.save_path+'/status.yaml', 'w') as f:
                    yaml.dump(status, f)
            
            # 满足收敛判据时保存参数
            if jnp.mean(E_stds) < 0.01 and jnp.std(E_means) < 0.01:
                checkpoint.save(ckpt_save_path, t, data, params, opt_state, mcmc_width)
                time_of_last_ckpt = time.time()
                break

        # 关闭日志
        if cfg.system.states:
            energy_matrix_file.close()
            if cfg.observables.s2:
                s2_matrix_file.close()
            if cfg.observables.dipole:
                dipole_matrix_file.close()
        if cfg.observables.density:
            density_matrix_file.close()

def vmc_train(cfg: ml_collections.ConfigDict, writer_manager=None):
    """Runs training loop for QMC.

    Args:
        cfg: ConfigDict containing the system and training parameters to run on. See
          base_config.default for more details.
        writer_manager: context manager with a write method for logging output. If
          None, a default writer (ferminet.utils.writers.Writer) is used.

    Raises:
        ValueError: if an illegal or unsupported value in cfg is detected.
    """
    # 初始化配置和设备
    (cfg, key, num_devices, num_hosts, num_states, host_batch_size,
     total_host_batch_size, device_batch_size, data_shape, atoms,
     charges, nspins, batch_atoms, batch_charges, core_electrons, ecp) = init_config(cfg)

    # 初始化波函数和网络
    (params, network, network_list, signed_network, logabs_network,
    batch_network, log_network, batch_network_mcmc, log_network_for_loss, key) = init_wavefunction(cfg, key, charges, nspins, core_electrons, ecp)

    # 初始化数据和检查点
    (t_init, data, params, opt_state_ckpt, mcmc_width_ckpt,
     density_state_ckpt, ckpt_save_path, key) = init_datas(
        cfg, key, data_shape, total_host_batch_size, batch_atoms,
        batch_charges, core_electrons, params, host_batch_size)
    
    data.positions = data.positions[:, :, :cfg.system.states, :]
    data.spins = data.spins[:, :, :cfg.system.states, :]

    # 初始化日志和观测量
    (train_schema, observable_fns, observable_states, energy_matrix_file,
     energy_matrix_std_file, energy_matrix_clip_file,
     energy_matrix_clip_std_file, s2_matrix_file, dipole_matrix_file,
     density_matrix_file, sharded_key, density_update) = init_log(
        cfg, ckpt_save_path, signed_network, nspins, data,
        density_state_ckpt, key)

    # 初始化MCMC步骤
    mcmc_step = init_mcmc_step(cfg, batch_network_mcmc, device_batch_size, atoms, num_states)

    # 初始化局部能量函数
    local_energy_fn = init_local_energy_fn(cfg, signed_network, charges, nspins, cfg.network.get('complex', False))

    # 初始化损失函数
    evaluate_loss = init_evaluate_loss(cfg, log_network, log_network_for_loss, logabs_network, signed_network, local_energy_fn, cfg.network.get('complex', False))

    # 初始化学习率调度
    learning_rate_schedule = init_learning_rate_schedule(cfg)

    # 初始化优化器
    optimizer, opt_state, sharded_key = init_optimizer(cfg, evaluate_loss, learning_rate_schedule, params,
                                                       sharded_key, data, opt_state_ckpt)

    # 初始化变量
    time_of_last_ckpt = time.time()
    weighted_stats = None
    num_resets = 0  # 用于 reset_if_nan 为真时
    if mcmc_width_ckpt is not None:
        mcmc_width = kfac_jax.utils.replicate_all_local_devices(mcmc_width_ckpt[0])
    else:
        mcmc_width = kfac_jax.utils.replicate_all_local_devices(
            jnp.asarray(cfg.mcmc.move_width))
    pmoves = np.zeros(cfg.mcmc.adapt_frequency)
    if cfg.mcmc.target == 'diag':
        data.positions = jnp.reshape(data.positions, data_shape + (cfg.system.states, -1,))
        data.spins = jnp.reshape(data.spins, data_shape + (cfg.system.states, -1,))
    else:
        data.positions = jnp.reshape(data.positions, data_shape + (-1,))
        data.spins = jnp.reshape(data.spins, data_shape + (-1,))
    # 执行burn-in步骤
    if t_init == 0:
        logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)
        burn_in_step = make_training_step(
            mcmc_step=mcmc_step, optimizer_step=null_update)
        for _ in range(cfg.mcmc.burn_in):
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            data, params, *_ = burn_in_step(
                data,
                params,
                state=None,
                key=subkeys,
                mcmc_width=mcmc_width)
        logging.info('Completed burn-in MCMC steps')
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        ptotal_energy = constants.pmap(evaluate_loss)
        initial_energy, _ = ptotal_energy(params, subkeys, data)
        logging.info('Initial energy: %03.4f E_h', initial_energy[0])

    # 设置训练步骤函数
    if not optimizer:
        step = make_training_step(
            mcmc_step=mcmc_step,
            optimizer_step=make_loss_step(evaluate_loss))
    elif isinstance(optimizer, optax.GradientTransformation):
        step = make_training_step(
            mcmc_step=mcmc_step,
            optimizer_step=make_opt_update_step(evaluate_loss, optimizer),
            reset_if_nan=cfg.optim.reset_if_nan)
    elif isinstance(optimizer, kfac_jax.Optimizer):
        step = make_kfac_training_step(
            mcmc_step=mcmc_step,
            damping=cfg.optim.kfac.damping,
            optimizer=optimizer,
            reset_if_nan=cfg.optim.reset_if_nan)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer}')

    # 主训练循环
    out_file_name = os.path.join(ckpt_save_path, 'train_stats'+ datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S') + '.csv')
    start_time = time.time()

    for t in range(t_init, cfg.optim.iterations):
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        data, params, opt_state, stats, pmove = step(
            data,
            params,
            opt_state,
            subkeys,
            mcmc_width)

        # 处理统计信息和日志记录
        loss = stats['loss'][0]
        aux_data = stats['aux']
        weighted_stats = statistics.exponentialy_weighted_stats(
            alpha=0.1, observation=loss, previous_stats=weighted_stats)
        pmove = pmove[0]

        # 更新观测量
        observable_data = {
            key: fn(params, data, observable_states[key])
            for key, fn in observable_fns.items()
        }
        if cfg.observables.density:
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            observable_states['density'] = density_update(
                subkeys, params, data, observable_states['density'])

        # 更新MCMC移动宽度
        mcmc_width, pmoves = mcmc.update_mcmc_width(
            t, mcmc_width, cfg.mcmc.adapt_frequency, pmove, pmoves)

        # 检查NaN并处理
        if cfg.debug.check_nan:
            tree = {'params': params, 'loss': loss}
            if cfg.optim.optimizer != 'none':
                tree['optim'] = opt_state
            try:
                chex.assert_tree_all_finite(tree)
                num_resets = 0
            except AssertionError as e:
                if cfg.optim.reset_if_nan:
                    num_resets += 1
                    if num_resets > 100:
                        raise e
                else:
                    raise e

        # 日志记录
        if t % cfg.log.stats_frequency == 0:
            logging_str = ('Step %05d: '
                            '%03.4f s, %03.4f E_h, %03.4f E_h^2, exp. variance=%03.4f E_h^2, pmove=%0.2f')
            logging_args = t, time.time()-start_time, loss, aux_data.variance[0].sum(), weighted_stats.variance, pmove
            writer_kwargs = dict(
                step =  t,
                time = time.time()-start_time,
                energy = np.asarray(loss),
                var = np.asarray(aux_data.variance[0].sum()),
                ewmean = np.asarray(weighted_stats.mean),
                ewvar = np.asarray(weighted_stats.variance),
                pmove = np.asarray(pmove),
                grad_norm = stats['grad_norm'][0],
                learning_rate = stats['learning_rate'][0],
                momentum = stats['momentum'][0],
                param_norm = stats['param_norm'][0],
                precon_grad_norm = stats['precon_grad_norm'][0],
                update_norm = stats['update_norm'][0],
            )
            start_time = time.time()
            for key in observable_data:
                obs_data = observable_data[key]
                if cfg.system.states:
                    obs_data = np.trace(obs_data, axis1=-1, axis2=-2)
                if key == 'dipole':
                    writer_kwargs['mu_x'] = obs_data[0]
                    writer_kwargs['mu_y'] = obs_data[1]
                    writer_kwargs['mu_z'] = obs_data[2]
                elif key == 'density':
                    pass
                elif key == 's2':
                    writer_kwargs[key] = obs_data
                    logging_str += ', <S^2>=%03.4f'
                    logging_args += obs_data,
            logging.info(logging_str, *logging_args)
            if t>0 or os.path.exists(out_file_name):
                df = pd.DataFrame(writer_kwargs, index=[0])
                df.to_csv(out_file_name, mode='a', header=False)
            elif t == 0:
                df = pd.DataFrame(writer_kwargs, index=[0])
                df.to_csv(out_file_name, header=True, mode = 'w')

        # 处理过大的观测量数据，直接保存为.npy文件
        if cfg.system.states:
            if cfg.run_model != 'test':
                energy_matrix = aux_data.local_energy_mat
                energy_matrixs = energy_matrix.reshape((cfg.batch_size,)+energy_matrix.shape[2:])
                energy_matrixs_clip = clip_local_energys(energy_matrixs)
                energy_matrix = np.nanmean(energy_matrixs, axis=0)
                energy_matrix_std = np.nanstd(energy_matrixs, axis=0)
                energy_matrix_clip = np.nanmean(energy_matrixs_clip, axis=0)
                energy_matrix_clip_std = np.nanstd(energy_matrixs_clip, axis=0)
                np.save(energy_matrix_file, energy_matrix)
                np.save(energy_matrix_std_file, energy_matrix_std)
                np.save(energy_matrix_clip_file, energy_matrix_clip)
                np.save(energy_matrix_clip_std_file, energy_matrix_clip_std)
            else:
                energy_matrix = aux_data.local_energy_mat
                np.save(energy_matrix_file, energy_matrix)
            if cfg.observables.s2:
                np.save(s2_matrix_file, observable_data['s2'])
            if cfg.observables.dipole:
                np.save(dipole_matrix_file, observable_data['dipole'])
        if cfg.observables.density:
            np.save(density_matrix_file, observable_data['density'])

        # 检查点保存
        if time.time() - time_of_last_ckpt > cfg.log.save_frequency * 60:
            checkpoint.save(ckpt_save_path, t, data, params, opt_state, mcmc_width)
            time_of_last_ckpt = time.time()

    # 关闭日志
    if cfg.system.states:
        energy_matrix_file.close()
        if cfg.observables.s2:
            s2_matrix_file.close()
        if cfg.observables.dipole:
            dipole_matrix_file.close()
    if cfg.observables.density:
        density_matrix_file.close()

def post_process(cfg: ml_collections.ConfigDict, writer_manager=None):
    """Runs training loop for QMC.

    Args:
        cfg: ConfigDict containing the system and training parameters to run on. See
          base_config.default for more details.
        writer_manager: context manager with a write method for logging output. If
          None, a default writer (ferminet.utils.writers.Writer) is used.

    Raises:
        ValueError: if an illegal or unsupported value in cfg is detected.
    """
    cfg.observables.s2 = True  # spin magnitude
    cfg.observables.density = True  # density matrix
    cfg.observables.density_basis = 'def2-tzvpd'  # basis used for DM calculation
    cfg.observables.dipole = True  # dipole moment
    # 初始化配置和设备
    (cfg, key, num_devices, num_hosts, num_states, host_batch_size,
     total_host_batch_size, device_batch_size, data_shape, atoms,
     charges, nspins, batch_atoms, batch_charges, core_electrons, ecp) = init_config(cfg)

    # 初始化波函数和网络
    (params, network, signed_network, logabs_network,
     batch_network, log_network, batch_network_mcmc, log_network_for_loss, key) = init_wavefunction_single(cfg, key, charges, nspins, core_electrons, ecp)
    
    # 初始化数据和检查点
    (t_init, data, params, opt_state_ckpt, mcmc_width_ckpt,
     density_state_ckpt, ckpt_save_path, key) = init_datas(
        cfg, key, data_shape, total_host_batch_size, batch_atoms,
        batch_charges, core_electrons, params, host_batch_size)
    
    ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path+'/postprocess/')
    # 初始化日志和观测量
    (train_schema, observable_fns, observable_states, energy_matrix_file,
     energy_matrix_std_file, energy_matrix_clip_file,
     energy_matrix_clip_std_file, s2_matrix_file, dipole_matrix_file,
     density_matrix_file, sharded_key, density_update) = init_log(
        cfg, ckpt_save_path, signed_network, nspins, data,
        density_state_ckpt, key)

    # 初始化MCMC步骤
    mcmc_step = init_mcmc_step(cfg, batch_network_mcmc, device_batch_size, atoms, num_states)

    # 初始化局部能量函数
    local_energy_fn = init_local_energy_fn(cfg, signed_network, charges, nspins, cfg.network.get('complex', False))

    # 初始化损失函数
    evaluate_loss = init_evaluate_loss(cfg, log_network, log_network_for_loss, logabs_network, signed_network, local_energy_fn, cfg.network.get('complex', False))

    # 初始化变量
    time_of_last_ckpt = time.time()
    weighted_stats = None

    if mcmc_width_ckpt is not None:
        mcmc_width = kfac_jax.utils.replicate_all_local_devices(mcmc_width_ckpt[0])
    else:
        mcmc_width = kfac_jax.utils.replicate_all_local_devices(
            jnp.asarray(cfg.mcmc.move_width))
    pmoves = np.zeros(cfg.mcmc.adapt_frequency)
    if cfg.mcmc.target == 'diag':
        data.positions = jnp.reshape(data.positions, data_shape + (cfg.system.states, -1,))
        data.spins = jnp.reshape(data.spins, data_shape + (cfg.system.states, -1,))
    # 执行burn-in步骤
    if t_init == 0:
        logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)
        burn_in_step = make_training_step(
            mcmc_step=mcmc_step, optimizer_step=null_update)
        for _ in range(cfg.mcmc.burn_in):
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            data, params, *_ = burn_in_step(
                data,
                params,
                state=None,
                key=subkeys,
                mcmc_width=mcmc_width)
        logging.info('Completed burn-in MCMC steps')
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        ptotal_energy = constants.pmap(evaluate_loss)
        initial_energy, _ = ptotal_energy(params, subkeys, data)
        logging.info('Initial energy: %03.4f E_h', initial_energy[0])

    # 主训练循环
    out_file_name = os.path.join(ckpt_save_path, 'train_stats'+ datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S') + '.csv')
    start_time = time.time()

    for t in range(t_init, cfg.optim.iterations):
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        data, params, _, _, _, pmove = burn_in_step(
            data,
            params,
            state=None,
            key=subkeys,
            mcmc_width=mcmc_width)

        ptotal_energy = constants.pmap(evaluate_loss)
        initial_energy, stats = ptotal_energy(params, subkeys, data)

        # 处理统计信息和日志记录
        loss = initial_energy[0]
        aux_data = stats
        weighted_stats = statistics.exponentialy_weighted_stats(
            alpha=0.1, observation=loss, previous_stats=weighted_stats)
        pmove = pmove[0]

        # 更新观测量
        observable_data = {
            key: fn(params, data, observable_states[key])
            for key, fn in observable_fns.items()
        }
        if cfg.observables.density:
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            observable_states['density'] = density_update(
                subkeys, params, data, observable_states['density'])

        # 更新MCMC移动宽度
        mcmc_width, pmoves = mcmc.update_mcmc_width(
            t, mcmc_width, cfg.mcmc.adapt_frequency, pmove, pmoves)

        # 日志记录
        if t % cfg.log.stats_frequency == 0:
            logging_str = ('Step %05d: '
                            '%03.4f s, %03.4f E_h, %03.4f E_h^2, exp. variance=%03.4f E_h^2, pmove=%0.2f')
            logging_args = t, time.time()-start_time, loss, aux_data.variance[0].sum(), weighted_stats.variance, pmove
            writer_kwargs = dict(
                step =  t,
                time = time.time()-start_time,
                energy = np.asarray(loss),
                var = np.asarray(aux_data.variance[0].sum()),
                ewmean = np.asarray(weighted_stats.mean),
                ewvar = np.asarray(weighted_stats.variance),
                pmove = np.asarray(pmove),
            )
            start_time = time.time()
            for key in observable_data:
                obs_data = observable_data[key]
                if cfg.system.states:
                    obs_data = np.trace(obs_data, axis1=-1, axis2=-2)
                if key == 'dipole':
                    writer_kwargs['mu_x'] = obs_data[0]
                    writer_kwargs['mu_y'] = obs_data[1]
                    writer_kwargs['mu_z'] = obs_data[2]
                elif key == 'density':
                    pass
                elif key == 's2':
                    writer_kwargs[key] = obs_data
                    logging_str += ', <S^2>=%03.4f'
                    logging_args += obs_data,
            logging.info(logging_str, *logging_args)
            if t>0 or os.path.exists(out_file_name):
                df = pd.DataFrame(writer_kwargs, index=[0])
                df.to_csv(out_file_name, mode='a', header=False)
            elif t == 0:
                df = pd.DataFrame(writer_kwargs, index=[0])
                df.to_csv(out_file_name, header=True, mode = 'w')

        # 处理过大的观测量数据，直接保存为.npy文件
        if cfg.system.states:
            energy_matrix = aux_data.local_energy_mat
            np.save(energy_matrix_file, energy_matrix)
            if cfg.observables.s2:
                np.save(s2_matrix_file, observable_data['s2'])
            if cfg.observables.dipole:
                np.save(dipole_matrix_file, observable_data['dipole'])
        if cfg.observables.density:
            np.save(density_matrix_file, observable_data['density'])

        # 检查点保存
        if time.time() - time_of_last_ckpt > cfg.log.save_frequency * 60:
            checkpoint.save(ckpt_save_path, t, data, params, None, mcmc_width)
            time_of_last_ckpt = time.time()

    # 关闭日志
    if cfg.system.states:
        energy_matrix_file.close()
        if cfg.observables.s2:
            s2_matrix_file.close()
        if cfg.observables.dipole:
            dipole_matrix_file.close()
    if cfg.observables.density:
        density_matrix_file.close()
