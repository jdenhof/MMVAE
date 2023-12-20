TRAIN = 'train'
TEST = 'test'
PHASES = ('train', 'test')

def validate(phase):
    if phase not in PHASES:
        raise ValueError(f"Phase {phase} not in {PHASES}")
    return phase
