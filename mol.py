# import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置使用 GPU 0
from ferminet import base_config
from ferminet import train, main
import os
import jax
from absl import logging
import yaml

os.environ['NVIDIA_TF32_OVERRIDE']="0"
# jax.config.update("jax_enable_x64", False)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'

# Optional, for also printing training progress to STDOUT.
# If running a script, you can also just use the --alsologtostderr flag.
# logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

from pyscf import gto
LiH_mol = gto.Mole()
LiH_mol.build(
    atom=f'Li 0 0 0; H 1.595 0 0',
    basis={'Li': 'ccpvdz', 'H': 'ccpvdz'},
    spin=0)

BeH_mol = gto.Mole()
BeH_mol.build(
    atom=f'Be 0 0 0; H 1.326903 0 0',
    basis={'Be': 'ccpvdz', 'H': 'ccpvdz'},
    # ecp={'Be': 'ccecp'},
    spin=1)

BH_mol = gto.Mole()
BH_mol.build(
    atom=f'B 0 0 0; H 1.2324 0 0',
    basis={'B': 'ccpvdz', 'H': 'ccpvdz'},
    spin=0)

H2O_mol = gto.Mole()
H2O_mol.build(
    atom=f'O 0.000000, 0.000000, -0.069903; H 0.000000, 0.757532, 0.518435; H 0.000000, -0.757532, 0.518435',
    basis={'O': 'ccpvdz', 'H': 'ccpvdz'},
    spin=0)

cfg = base_config.default()
cfg.system.pyscf_mol = BH_mol

# Set training parameters
cfg.optim.laplacian = 'folx'
cfg.optim.reset_if_nan = True
cfg.optim.spin_energy = 1.0
# cfg.optim.optimizer = 'sgd'
cfg.optim.lr.rate=0.05
cfg.debug.check_nan = True
cfg.system.states = 1
cfg.system.states_total = 5
cfg.system.states_update = 1
cfg.batch_size = 4096
cfg.optim.iterations = 200000

cfg.mcmc.target = 'diag' # 'diag'

# cfg.system.use_pp = True  # Enable pseudopotentials
# cfg.system.pp.symbols = ['O']  # Indicate which atoms to apply PP to
cfg.pretrain.iterations = 10000
cfg.pretrain.method = 'casci'
cfg.network.type = 'multi'# 'single'
# cfg.network.network_type = 'psiformer'
# cfg.network.psiformer.num_layers = 2
# cfg.network.psiformer.num_heads = 4
# cfg.network.psiformer.heads_dim = 32
# cfg.network.psiformer.mlp_hidden_dims = (128,)
# cfg.network.determinants = 16
# cfg.network.full_det = False
cfg.optim.objective='vmc_ipf'
cfg.pretrain.excitation_type = 'ordered' # 'random'
cfg.log.save_path = '../IPF/ferminet/'+'BH_4090'
cfg.run_model = 'run'
# cfg.log.restore_path = ''

status = dict(
    total_states = 5,
    batch_size = 4096,
    pretrain = dict(
        status = False,
        Epochs = 0,
    ),
    vmctrain = dict(),
    posttrain = dict(
        status = False,
        Epochs = 0,)
)

for i in range(cfg.system.states_total):
    status['vmctrain']['state_'+str(i)] = dict(
        status = False,
        Epochs = 0,
    )
if not os.path.exists(cfg.log.save_path):
    os.makedirs(cfg.log.save_path)

if not os.path.exists(cfg.log.save_path+'/status.yaml'):
    with open(cfg.log.save_path+'/status.yaml', 'w') as f:
        yaml.dump(status, f)
else:
    with open(cfg.log.save_path+'/status.yaml', 'r') as f:
        status = yaml.load(f, Loader=yaml.FullLoader)

# cfg.observables.s2 = True  # spin magnitude
# cfg.observables.density = True  # density matrix
# cfg.observables.density_basis = 'def2-tzvpd'  # basis used for DM calculation
# cfg.observables.dipole = True  # dipole moment

if status['pretrain']['status'] == False:
    main.pre_train(cfg, status)
main.ipf_train(cfg)

# if status
# # main.vmc_train(cfg)
# main.ipf_train(cfg)
# cfg.log.restore_path = '/home/liujinde/data/home/liujinde/deep_wavafunction/Excited_Calculate/Test/ferminet/BH_casscf_diag/pretrain/qmcjax_ckpt_009999.npz'
# main.post_process(cfg)