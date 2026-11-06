"""Bound MPS/CUDA input graph shapes without truncating or changing valid samples."""


STREAM_AXES = {'local': 2, 'mel': 1, 'song': 1, 'reference': 1, 'intensity': 1}


def pad_input_shapes(inputs, lengths):
    result = dict(inputs)
    for stream, length in lengths.items():
        if stream not in STREAM_AXES or length < 1:
            raise ValueError('Unknown stream or invalid fixed length')
        axis = STREAM_AXES[stream]
        keys = (stream, stream+'_times', stream+'_valid')
        if stream == 'mel':
            # Optional charter-timing features [B,T_mel,F] follow the mel frames.
            keys += tuple(k for k in ('beat', 'beat_valid') if k in inputs)
        for key in keys:
            value = inputs[key]
            key_axis = axis if key == stream else 1
            n = value.shape[key_axis]
            if n > length:
                raise ValueError(f'{key} has {n} entries, exceeding the frozen shape {length}')
            if n == length:
                continue
            shape = list(value.shape); shape[key_axis] = length
            padded = value.new_zeros(shape)
            index = [slice(None)]*value.ndim; index[key_axis] = slice(0, n)
            padded[tuple(index)] = value
            result[key] = padded
            result[key] = padded
    return result


def pad_sequence_shapes(sequence, length):
    """Pad teacher sequences to one graph bucket; padded tokens stay masked out."""
    if length < 1:
        raise ValueError('Invalid sequence bucket')
    result = {}
    for key, value in sequence.items():
        if value.ndim < 2:
            raise ValueError(f'Sequence tensor {key} has no token axis')
        n = value.shape[1]
        if n > length:
            raise ValueError(f'{key} has {n} tokens, exceeding the bucket {length}')
        if n == length:
            result[key] = value
            continue
        shape = list(value.shape); shape[1] = length
        padded = value.new_zeros(shape)
        padded[:, :n] = value
        result[key] = padded
    result['allowed'][..., 0] |= ~result['valid']
    return result
