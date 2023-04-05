import math
import time

import numpy as np
from torch.utils.data import DataLoader
from transformers import WhisperProcessor
from transformers.pipelines.base import no_collate_fn
from transformers.pipelines.pt_utils import PipelineChunkIterator, PipelinePackIterator


class Pipeline:
    """Relies on the data loaders defined in transformers Pipeline"""

    def __init__(self, checkpoint="openai/whisper-tiny.en"):
        self.checkpoint = checkpoint
        self.processor = WhisperProcessor.from_pretrained(self.checkpoint)
        self.feature_extractor = self.processor.feature_extractor
        self.tokenizer = self.processor.tokenizer

    @staticmethod
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

    def preprocess(self, inputs, chunk_length_s=0, stride_length_s=None):
        array = inputs.get("array")
        in_sampling_rate = inputs.get("sampling_rate")
        stride = inputs.get("stride", None)

        if in_sampling_rate != self.feature_extractor.sampling_rate:
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

            for item in self.chunk_iter(
                array,
                self.feature_extractor,
                chunk_len,
                stride_left,
                stride_right,
            ):
                yield item
        else:
            processed = self.feature_extractor(
                array, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="np"
            )
            if stride is not None:
                processed["stride"] = stride
            yield {"is_last": True, **processed}

    def forward(self, model_inputs, return_timestamps=False, generate_kwargs=None):
        print(model_inputs["stride"])
        if generate_kwargs is None:
            generate_kwargs = {}

        if return_timestamps:
            generate_kwargs["return_timestamps"] = return_timestamps
        is_last = model_inputs.pop("is_last")

        out = {}
        stride = model_inputs.pop("stride", None)
        if stride is not None:
            out["stride"] = stride

        return {"is_last": is_last, **out}

    def __call__(
        self,
        inputs,
        chunk_length_s=0,
        stride_length_s=None,
        return_timestamps=None,
        return_language=None,
        generate_kwargs=None,
        batch_size=4,
        num_workers=1,
    ):
        dataset = PipelineChunkIterator(
            [inputs], self.preprocess, {"chunk_length_s": chunk_length_s, "stride_length_s": stride_length_s}
        )
        collate_fn = no_collate_fn if batch_size == 1 else pad_collate_fn()
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)
        model_iterator = PipelinePackIterator(dataloader, self.forward, {}, loader_batch_size=batch_size)

        for batch in enumerate(model_iterator):
            continue


def _pad(items, key):
    if isinstance(items[0][key], np.ndarray):
        if key == "input_features":
            # this is probably a mel spectrogram batched
            return np.concatenate([item[key] for item in items], axis=0)
    else:
        return [item[key] for item in items]


def pad_collate_fn():
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
            padded[key] = _pad(items, key)
        return padded

    return inner


class ManualIterator:
    """Manual implementation"""

    def __init__(self, checkpoint="openai/whisper-tiny.en"):
        self.checkpoint = checkpoint
        self.processor = WhisperProcessor.from_pretrained(self.checkpoint)
        self.feature_extractor = self.processor.feature_extractor
        self.tokenizer = self.processor.tokenizer

    @staticmethod
    def chunk_iter_with_batch(inputs, feature_extractor, chunk_len, stride_left, stride_right, batch_size):
        inputs_len = inputs.shape[0]
        step = chunk_len - stride_left - stride_right

        all_chunk_start_idx = np.arange(0, inputs_len, step)
        num_samples = len(all_chunk_start_idx)

        num_batches = math.ceil(num_samples / batch_size)
        batch_idx = np.array_split(np.arange(num_samples), num_batches)

        for i, idx in enumerate(batch_idx):
            chunk_start_idx = all_chunk_start_idx[idx]

            chunk_end_idx = chunk_start_idx + chunk_len

            chunks = [inputs[chunk_start:chunk_end] for chunk_start, chunk_end in zip(chunk_start_idx, chunk_end_idx)]
            processed = feature_extractor(chunks, sampling_rate=feature_extractor.sampling_rate, return_tensors="np")

            _stride_left = np.where(chunk_start_idx == 0, 0, stride_left)
            is_last = np.where(stride_right > 0, chunk_end_idx > inputs_len, chunk_end_idx >= inputs_len)
            _stride_right = np.where(is_last, 0, stride_right)

            chunk_lens = [chunk.shape[0] for chunk in chunks]
            strides = [
                (chunk_l, _stride_l, _stride_r)
                for chunk_l, _stride_l, _stride_r in zip(chunk_lens, _stride_left, _stride_right)
            ]

            yield {"stride": strides, **processed}

    def preprocess_batch(self, inputs, chunk_length_s=0, stride_length_s=None, batch_size=None):
        array = inputs.get("array")
        in_sampling_rate = inputs.get("sampling_rate")
        stride = inputs.get("stride", None)

        if in_sampling_rate != self.feature_extractor.sampling_rate:
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

            for item in self.chunk_iter_with_batch(
                array,
                self.feature_extractor,
                chunk_len,
                stride_left,
                stride_right,
                batch_size,
            ):
                yield item
        else:
            processed = self.feature_extractor(
                array, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="np"
            )
            if stride is not None:
                processed["stride"] = stride
            yield processed

    def forward(self, model_inputs, return_timestamps=False, generate_kwargs=None):
        if generate_kwargs is None:
            generate_kwargs = {}

        if return_timestamps:
            generate_kwargs["return_timestamps"] = return_timestamps

        out = {}
        stride = model_inputs.pop("stride", None)
        if stride is not None:
            out["stride"] = stride

    def __call__(
        self,
        inputs,
        chunk_length_s=0,
        stride_length_s=None,
        return_timestamps=None,
        return_language=None,
        generate_kwargs=None,
        batch_size=4,
        num_workers=1,
    ):
        dataloader = self.preprocess_batch(inputs, chunk_length_s, stride_length_s, batch_size)

        for batch in dataloader:
            print(batch["stride"])
            self.forward(batch)


pipeline = Pipeline()
custom_pipeline = ManualIterator()

powers = np.logspace(0, 6, 7, base=2)
input_lengths = [30, 60, 120, 240]

for input_len in input_lengths:
    print(f"=== Input len {input_len} ===")
    inputs = {"array": np.ones(16000 * (int(input_len))), "sampling_rate": 16000}

    # first run our custom one since it doesn't consume the audio input
    start = time.time()
    custom_pipeline(inputs, chunk_length_s=30)
    runtime = time.time() - start
    print(f"Manual: {runtime}")

    # now run the transformers based one
    start = time.time()
    pipeline(inputs, chunk_length_s=30)
    runtime = time.time() - start
    print(f"Pipeline: {runtime}")


"""
Results:
=== Input len 30.0 ===
Manual: 0.09786629676818848
Pipeline: 0.4056670665740967

=== Input len 60.0 ===
Manual: 0.14911437034606934
Pipeline: 0.6139914989471436

=== Input len 120.0 ===
Manual: 0.29366159439086914
Pipeline: 0.9421713352203369

=== Input len 240.0 ===
Manual: 0.5912315845489502
Pipeline: 1.2646639347076416

=== Input len 480.0 ===
Manual: 1.1709322929382324
Pipeline: 1.6115500926971436

=== Input len 960.0 ===
Manual: 2.373847246170044
Pipeline: 3.2664968967437744

=== Input len 1920.0 ===
Manual: 4.619845151901245
Pipeline: 5.50755500793457
"""
