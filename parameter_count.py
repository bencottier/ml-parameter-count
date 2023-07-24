def transformer_parameter_count(n_block=4, d_model=512, d_ff=2048, encoder_only=False, approx=False):
    """
    Calculate the number of parameters in a Transformer.
    Formulas are the following:
        multi-head attention: 4*(d_model^2 + d_model)
            if approx=False, 4*d_model^2 otherwise
        feed-forward: 2*d_model*d_ff + d_model + d_ff 
            if approx=False, 2*d_model*d_ff otherwise
        layer normalization: 2*d_model if approx=False, 0 otherwise

    Encoder block consists of: 
        1 multi-head attention block, 
        1 feed-forward net, and 
        2 layer normalizations.
    Decoder block consists of: 
        2 multi-head attention blocks, 
        1 feed-forward net, and 
        3 layer normalizations.

    :param n_block: (int) number of Transformer blocks (also known as layers)
    :param d_model: (int) model dimensionality
    :param d_ff: (int) internal dimensionality of a feed-forward neural network
    :param encoder: (bool) if True, return the number of parameters of the Encoder, 
        otherwise the full Encoder-Decoder Transformer
    :param approx: (bool) if True, result is approximate (see formulas)
    :return: (int) number of learnable parameters in Transformer
    """

    attention = 4 * (d_model ** 2 + d_model) if not approx else 4 * d_model ** 2
    feed_forward = 2 * d_model * d_ff + d_model + d_ff if not approx else 2 * d_model * d_ff
    layer_norm = 2 * d_model if not approx else 0

    if encoder_only:
        return n_block * (attention + feed_forward + 2 * layer_norm)
    else:
        return n_block * (3 * attention + 2 * feed_forward + 5 * layer_norm)


if __name__ == '__main__':
    # Test cases
    # GPT-3: should be approximately 175B i.e. 1.75e+11
    print(f"Number of parameters in GPT-3: {transformer_parameter_count(n_block=96, d_model=12288, d_ff=49152, encoder_only=True):.2e}")
