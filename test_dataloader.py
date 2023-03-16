import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import WhisperProcessor
from transformers.pipelines.base import no_collate_fn
from transformers.pipelines.pt_utils import PipelineChunkIterator, PipelinePackIterator, PipelineIterator


def chunk_iter(inputs, feature_extractor, chunk_len, stride_left, stride_right):
    inputs_len = inputs.shape[0]
    step = chunk_len - stride_left - stride_right
    for chunk_start_idx in range(0, inputs_len, step):
        chunk_end_idx = chunk_start_idx + chunk_len
        chunk = inputs[chunk_start_idx:chunk_end_idx]
        processed = feature_extractor(chunk, sampling_rate=feature_extractor.sampling_rate, return_tensors="np")
        _stride_left = 0 if chunk_start_idx == 0 else stride_left
        # all right strides must be full, otherwise it is the last item
        is_last = chunk_end_idx > inputs_len if stride_right > 0 else chunk_end_idx >= inputs_len
        _stride_right = 0 if is_last else stride_right

        chunk_len = chunk.shape[0]
        stride = (chunk_len, _stride_left, _stride_right)
        if chunk.shape[0] > _stride_left:
            yield {"is_last": is_last, "stride": stride, **processed}
        if is_last:
            break

class Pipeline:
    def __init__(self, checkpoint="openai/whisper-tiny.en"):
        self.checkpoint = checkpoint
        self.processor = WhisperProcessor.from_pretrained(self.checkpoint)
        self.feature_extractor = self.processor.feature_extractor
        self.tokenizer = self.processor.tokenizer

    def preprocess(self, inputs, chunk_length_s=0, stride_length_s=None):
        # TODO(SG): handle more generic input types, currently assume inputs is a dict {"array":, "sampling_rate":}
        array = inputs.get("array")
        in_sampling_rate = inputs.get("sampling_rate")
        stride = inputs.get("stride", None)

        if in_sampling_rate != self.feature_extractor.sampling_rate:
            # TODO(SG): resampling ...
            ratio = self.feature_extractor.sampling_rate / in_sampling_rate
        else:
            ratio = 1

        if stride is not None:
            if stride[0] + stride[1] > inputs.shape[0]:
                raise ValueError("Stride is too large for input")

            # Stride needs to get the chunk length here, it's going to get
            # swallowed by the `feature_extractor` later, and then batching
            # can add extra data in the inputs, so we need to keep track
            # of the original length in the stride so we can cut properly.
            stride = (inputs.shape[0], int(round(stride[0] * ratio)), int(round(stride[1] * ratio)))

        if chunk_length_s:
            if stride_length_s is None:
                stride_length_s = chunk_length_s / 6

            if isinstance(stride_length_s, (int, float)):
                stride_length_s = [stride_length_s, stride_length_s]

            chunk_len = round(chunk_length_s * self.feature_extractor.sampling_rate)
            stride_left = round(stride_length_s[0] * self.feature_extractor.sampling_rate)
            stride_right = round(stride_length_s[1] * self.feature_extractor.sampling_rate)

            if chunk_len < stride_left + stride_right:
                raise ValueError("Chunk length must be superior to stride length")

            for item in chunk_iter(
                    array, self.feature_extractor, chunk_len, stride_left, stride_right,
            ):
                yield item
        else:
            processed = self.feature_extractor(
                array, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="np"
            )
            if stride is not None:
                processed["stride"] = stride
            yield {"is_last": True, **processed}

    def postprocess(
                self, model_outputs, return_timestamps=None, return_language=None
        ):
        # Optional return types
        final_items = []

        for outputs in model_outputs:
            items = np.array(outputs["tokens"])
            final_items.append(items)

        time_precision = self.feature_extractor.chunk_length / self.model.config.max_source_positions
        # Send the chunking back to seconds, it's easier to handle in whisper
        sampling_rate = self.feature_extractor.sampling_rate
        for output in model_outputs:
            if "stride" in output:
                chunk_len, stride_left, stride_right = output["stride"]
                # Go back in seconds
                chunk_len /= sampling_rate
                stride_left /= sampling_rate
                stride_right /= sampling_rate
                output["stride"] = chunk_len, stride_left, stride_right

        text, optional = self.tokenizer._decode_asr(
            model_outputs,
            return_timestamps=return_timestamps,
            return_language=return_language,
            time_precision=time_precision,
        )
        return {"text": text, **optional}

    def forward(self, model_inputs, return_timestamps=False, generate_kwargs=None):
        if generate_kwargs is None:
            generate_kwargs = {}

        if return_timestamps:
            generate_kwargs["return_timestamps"] = return_timestamps
        is_last = model_inputs.pop("is_last")

        pred_ids = model_inputs.pop("input_features")
        out = {"tokens": np.asarray(pred_ids)}

        stride = model_inputs.pop("stride", None)
        if stride is not None:
            out["stride"] = stride

        return {"is_last": is_last, **out}

    def __call__(self, inputs, chunk_length_s=0, stride_length_s=None, return_timestamps=None, return_language=None, generate_kwargs=None, batch_size=4, num_workers=1):
        dataset = PipelineChunkIterator([inputs], self.preprocess, {"chunk_length_s": chunk_length_s, "stride_length_s": stride_length_s})
        collate_fn = no_collate_fn if batch_size == 1 else pad_collate_fn(self.tokenizer, self.feature_extractor)
        # TODO(SG): handle incomplete batches
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn, drop_last=True)
        model_iterator = PipelinePackIterator(dataloader, self.forward, {}, loader_batch_size=batch_size)
        return next(iter(model_iterator))

def _pad(items, key, padding_value, padding_side):
    batch_size = len(items)
    if isinstance(items[0][key], torch.Tensor):
        # Others include `attention_mask` etc...
        shape = items[0][key].shape
        dim = len(shape)
        if key in ["pixel_values", "image"]:
            # This is probable image so padding shouldn't be necessary
            # B, C, H, W
            return torch.cat([item[key] for item in items], dim=0)
        elif dim == 4 and key == "input_features":
            # this is probably a mel spectrogram batched
            return torch.cat([item[key] for item in items], dim=0)
        max_length = max(item[key].shape[1] for item in items)
        min_length = min(item[key].shape[1] for item in items)
        dtype = items[0][key].dtype

        if dim == 2:
            if max_length == min_length:
                # Bypass for `ImageGPT` which doesn't provide a padding value, yet
                # we can consistently pad since the size should be matching
                return torch.cat([item[key] for item in items], dim=0)
            tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
        elif dim == 3:
            tensor = torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value

        for i, item in enumerate(items):
            if dim == 2:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0])] = item[key][0].clone()
            elif dim == 3:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :, :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0]), :] = item[key][0].clone()
        return tensor
    elif isinstance(items[0][key], np.ndarray):
        if key == "input_features":
            # this is probably a mel spectrogram batched
            return np.concatenate([item[key] for item in items], axis=0)
    else:
        return [item[key] for item in items]

def pad_collate_fn(tokenizer, feature_extractor):
    # Tokenizer
    t_padding_side = None
    # Feature extractor
    f_padding_side = None
    if tokenizer is None and feature_extractor is None:
        raise ValueError("Pipeline without tokenizer or feature_extractor cannot do batching")
    if tokenizer is not None:
        if tokenizer.pad_token_id is None:
            raise ValueError(
                "Pipeline with tokenizer without pad_token cannot do batching. You can try to set it with "
                "`pipe.tokenizer.pad_token_id = model.config.eos_token_id`."
            )
        else:
            t_padding_value = tokenizer.pad_token_id
            t_padding_side = tokenizer.padding_side
    if feature_extractor is not None:
        # Feature extractor can be images, where no padding is expected
        f_padding_value = getattr(feature_extractor, "padding_value", None)
        f_padding_side = getattr(feature_extractor, "padding_side", None)

    if t_padding_side is not None and f_padding_side is not None and t_padding_side != f_padding_side:
        raise ValueError(
            f"The feature extractor, and tokenizer don't agree on padding side {t_padding_side} != {f_padding_side}"
        )
    padding_side = "right"
    if t_padding_side is not None:
        padding_side = t_padding_side
    if f_padding_side is not None:
        padding_side = f_padding_side

    def inner(items):
        keys = set(items[0].keys())
        for item in items:
            if set(item.keys()) != keys:
                raise ValueError(
                    f"The elements of the batch contain different keys. Cannot batch them ({set(item.keys())} !="
                    f" {keys})"
                )
        # input_values, input_pixels, input_ids, ...
        padded = {}
        for key in keys:
            if key in {"input_ids"}:
                # ImageGPT uses a feature extractor
                if tokenizer is None and feature_extractor is not None:
                    _padding_value = f_padding_value
                else:
                    _padding_value = t_padding_value
            elif key in {"input_values", "pixel_values", "input_features"}:
                _padding_value = f_padding_value
            elif key in {"p_mask", "special_tokens_mask"}:
                _padding_value = 1
            elif key in {"attention_mask", "token_type_ids"}:
                _padding_value = 0
            else:
                # This is likely another random key maybe even user provided
                _padding_value = 0
            padded[key] = _pad(items, key, _padding_value, padding_side)
        return padded

    return inner


def chunk_iter_with_batch(inputs, feature_extractor, chunk_len, stride_left, stride_right, batch_size):
    inputs_len = inputs.shape[0]
    step = chunk_len - stride_left - stride_right

    chunk_start_idx = np.arange(0, inputs_len, step)
    chunk_end_idx = chunk_start_idx + chunk_len

    chunks = [inputs[chunk_start:chunk_end] for chunk_start, chunk_end in zip(chunk_start_idx, chunk_end_idx)]

    _stride_left = np.where(chunk_start_idx == 0, 0, stride_left)
    is_last = np.where(stride_right > 0, chunk_start_idx > inputs_len, chunk_end_idx >= inputs_len)
    _stride_right = np.where(is_last, 0, stride_right)

    chunk_lens = [chunk.shape[0] for chunk in chunks]
    strides = [(chunk_l, _stride_l, _stride_r) for chunk_l, _stride_l, _stride_r in zip(chunk_lens, _stride_left, _stride_right)]

    num_batches = math.ceil(len(chunks) / batch_size)
    batch_idx = np.array_split(np.arange(len(chunks)), num_batches)

    for idx in batch_idx:
        processed = feature_extractor(chunks[idx], sampling_rate=feature_extractor.sampling_rate, return_tensors="np")
        yield {"stride": strides[idx], **processed}


def preprocess_batch(inputs, feature_extractor, chunk_length_s=0, stride_length_s=None):
        # TODO(SG): handle more generic input types, currently assume inputs is a dict {"array":, "sampling_rate":}
        array = inputs.get("array")
        in_sampling_rate = inputs.get("sampling_rate")
        stride = inputs.get("stride", None)

        if in_sampling_rate != feature_extractor.sampling_rate:
            # TODO(SG): resampling ...
            ratio = feature_extractor.sampling_rate / in_sampling_rate
        else:
            ratio = 1

        if stride is not None:
            if stride[0] + stride[1] > inputs.shape[0]:
                raise ValueError("Stride is too large for input")

            # Stride needs to get the chunk length here, it's going to get
            # swallowed by the `feature_extractor` later, and then batching
            # can add extra data in the inputs, so we need to keep track
            # of the original length in the stride so we can cut properly.
            stride = (inputs.shape[0], int(round(stride[0] * ratio)), int(round(stride[1] * ratio)))

        if chunk_length_s:
            if stride_length_s is None:
                stride_length_s = chunk_length_s / 6

            if isinstance(stride_length_s, (int, float)):
                stride_length_s = [stride_length_s, stride_length_s]

            chunk_len = round(chunk_length_s * feature_extractor.sampling_rate)
            stride_left = round(stride_length_s[0] * feature_extractor.sampling_rate)
            stride_right = round(stride_length_s[1] * feature_extractor.sampling_rate)

            if chunk_len < stride_left + stride_right:
                raise ValueError("Chunk length must be superior to stride length")

            for item in chunk_iter(
                    array, feature_extractor, chunk_len, stride_left, stride_right,
            ):
                yield item
        else:
            processed = feature_extractor(
                array, sampling_rate=feature_extractor.sampling_rate, return_tensors="np"
            )
            if stride is not None:
                processed["stride"] = stride
            yield {"is_last": True, **processed}