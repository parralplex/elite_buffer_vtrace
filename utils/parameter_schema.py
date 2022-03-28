from schema import Schema, And, Use, Optional, SchemaError


def validate_config(schema, configuration):
    try:
        config = schema.validate(configuration)
        return config, True, ""
    except SchemaError as error:
        return (), False, str(error)


replay_schema = Schema({
    'type': And(Use(str), Use(str.lower), lambda s: s in ('queue', 'standard')),
    'capacity': And(Use(int), lambda n: 1 <= n),
    'sample_ratio': And(Use(float), lambda n: 0 <= n <= 1),
})

custom_replay_schema = Schema({
    'type': And(Use(str), Use(str.lower), lambda s: s in ('custom')),
    'capacity': And(Use(int), lambda n: 1 <= n),
    'sample_ratio': And(Use(float), lambda n: 0 <= n <= 1),
    'dist_function': And(str, Use(str.lower), lambda s: s in ('ln_norm', 'cos_dist', 'kl_div')),
    Optional('p', default=2): And(Use(int), lambda n: 1 <= n),
    Optional('insert_strategy'): And(Use(str), Use(str.lower), lambda s: s in 'elite_insertion'),
    Optional('sample_strategy'): And(Use(str), Use(str.lower), lambda s: s in ('elite_sampling', 'attentive_sampling')),
    Optional('lambda_batch_multiplier', default=1): And(Use(int), lambda n: 1 <= n),
    Optional('alfa_annealing_factor', default=1): And(Use(float), lambda n: 0 < n),
    Optional('elite_sampling_strategy', default='strategy3'): And(Use(str), Use(str.lower), lambda s: s in ('strategy1', 'strategy2', 'strategy3', 'strategy4')),
    Optional('elite_batch_size', default=20): And(Use(int), lambda n: 1 <= n),
})

