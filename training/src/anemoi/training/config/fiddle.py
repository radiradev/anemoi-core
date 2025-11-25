import fiddle as fdl
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.layers.encoder import GNNEncoder
from anemoi.models.layers.processor import GNNProcessor
from anemoi.models.layers.decoder import GNNDecoder
from anemoi.models.layers.residual import IdentityResidualConnection
from anemoi.models.layers.bounding import build_boundings, ReluBounding

def anemoimodelencprocdec_config():
    """Returns a Fiddle configuration for the AnemoiModelEncProcDec model."""

    # Using fdl.Partial to create callables that can be configured with hyperparameters.
    # Runtime arguments (like in_channels, sub_graph, etc.) will be supplied later.
    encoder_callable_cfg = fdl.Partial(
        GNNEncoder,
        hidden_dim=256,
    )
    processor_callable_cfg = fdl.Partial(
        GNNProcessor,
        num_layers=1,
        # other hyperparameters...
    )
    decoder_callable_cfg = fdl.Partial(
        GNNDecoder,
        hidden_dim=256,
    )
    residual_callable_cfg = fdl.Partial(
        IdentityResidualConnection,
    )

    # The `build_boundings` function will be passed as a dependency.
    # The list of bounding layers to be created will be configured here.
    # In a real scenario, the main script would create the bounding configs
    # as they need runtime data like `name_to_index`.
    # For now, we pass the function itself.
    boundings_callable_cfg = fdl.Partial(
        build_boundings
    )

    return fdl.Config(
        AnemoiModelEncProcDec,
        encoder=encoder_callable_cfg,
        processor=processor_callable_cfg,
        decoder=decoder_callable_cfg,
        residual=residual_callable_cfg,
        boundings=boundings_callable_cfg,
    )
