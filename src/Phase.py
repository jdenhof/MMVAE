class Phase:
    train = 'train'
    test = 'test'
    phases = ('train', 'test')

    def validate(phase):
        if phase not in Phase.phases:
            raise ValueError(f"Phase {phase} not in {Phase.phases}")
        return phase
    