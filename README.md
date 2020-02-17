# GOKU - Deep Generative ODE Modelling with Known Unknowns

This repository is an implementation of the GOKU paper (link will be added soon).

### Data creation
To create the datasets used in the paper run:
* Lotka-Volterra experiment:  `python3 create_data.py --model LV`
* Friction-less pixel-pendulum:  `python3 create_data.py --model Pendulum`
* Fricition pixel-pendulum: `python3 create_data.py --model Pendulum --friction`
* Cardiovascular system: `python3 create_data.py --model CVS`

The data would be created using default arguments. To view / modify them check the file `config.py`, and `create_data.py`.

### Training
To train the GOKU model run: `python3 goku_train.py --model <lv/pixel_pendulum/pixel_pendulum_friction/cvs>`

To train baselines:

* Latent-ODE: `python3 latent_ode_train.py --model <lv/pixel_pendulum/pixel_pendulum/friction/cvs>`.
* For Grounded Latent-ODE (L-ODE+), run Latent-ODE with the parameter `--grounding-loss`.
* LSTM: `python3 lstm_train.py --model <lv/pixel_pendulum/pixel_pendulum/friction/cvs>`.
* Direct-Identification (DI) has 3 different files for the different datasets (it cannot run the friction pendulum, since it needs the entire ODE functional form):
  * LV: `python3 di_baseline_lv.py`
  * Pendulum: `python3 di_baseline_pendulum.py`
  * CVS: `python3 di_baseline_cvs.py`
  
### Requirements:
* python 3
* pytorch
* numpy
* gym (for the pendulum experiment)
