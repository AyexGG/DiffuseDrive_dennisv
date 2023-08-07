# DiffuseDrive 
#### Robot Learning Course - Spring Semester 2023, ETH Zurich
### Dennis Vilgertshofer
---
## Abstract
> Autonomous navigation has gained increasing attention in recent years. However, much of the recent work published on the topic focuses on increases in driving performance, using increasingly complex systems. As such it often becomes hard to pinpoint the root cause of arising issues. While some of the recent work has also considered the interpretability concerns of autonomous driving, they aren't end-to-end and also rely on many manually designed heuristics. With advances in generative modeling, namely Diffusion models, in the area of generating high-fidelity images as well as their promise in decision-making processes, they may potentially included in autonomous driving pipelines. 
> With this project, an end-to-end autonomous driving architecture is designed that bases its decision making process on future trajectories that are sampled using a Diffusion model. The Diffusion process is conditioned using information about the past trajectory of the vehicle, as well as additional information. While the model struggles with highly complex traffic situations, it is capable of turning as well as following and changing lanes in simpler scenarios. Despite its issues, it shows promise regarding the inclusion of diffusion models in autonomous navigation systems and may well lead to better interpretability in the future. 
---
## How to reproduce our project
* Install the requirements in ```requirement.txt``` with a virtual environment 
* Follow the same installation described in the [Interfuser repository](https://github.com/opendilab/InterFuser) for CARLA simulator and data collection
* To run the training routine, run ```scripts/train.py``` with Python 3.8 or above.
* Configuration of training is in ```config/carla.py```
* To evaluate the model after training, run the script ```leaderboard/scripts/run_evaluation.sh```
* A [trained checkpoint](https://polybox.ethz.ch/index.php/s/PgJ4S3TFXyitjbk) of the model is also provided (as described in the report)
---
## Further Comments
* Our repository is based on the [Decision Diffuser](https://github.com/anuragajay/decision-diffuser/tree/main/code). 
* Our adaptations and extensions (e.g. classes or functions), all include "Carla" in the name.
* Data handling is specific to our data collection and therefore needs adaptation
* Further details on motivation, architecture, training, and results, can be found in the project report ```DiffuseDrive.pdf```
