#################################################################################################################
                                            # Import libraries
#################################################################################################################


import os
import random
import numpy as np
import torch


#################################################################################################################
                                            # Seed setting
#################################################################################################################

def seed_setting(random_state=42):
    os.environ["PYTHONHASHSEED"] = str(random_state)
    np.random.seed(random_state)
    random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


###################################################################################################################