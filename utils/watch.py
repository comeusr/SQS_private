import wandb
from composer import Callback, State, Logger, Event

class sparsity(Callback):

    def __init__(self):
        super().__init__()

    def log_mu(self, state:State, event:Event, logger:Logger):
        if event==Event.BATCH_END:
            wandb.log()
