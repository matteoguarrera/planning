import os
import wandb

def init_logging(params, model):
    # Automate tag creation on launch
    wandd_tags = []

    for k, v in params.items():
        wandd_tags.append(f"{k}:{v}")

    # create log directory
    os.makedirs(params['LOG_DIR'], exist_ok=True)

    # initialize wandb
    os.environ['WANDB_START_METHOD'] = 'thread'
    wandb.init(project=params['PROJECT_NAME'],
                entity=params['ENTITY'],
                tags=wandd_tags,
                dir=params['LOG_DIR'],
                name=f'Experiment {params["ID"]}'
    )
    wandb.run.save()
    wandb.watch(model, log='all')

    return wandb.run.get_url()