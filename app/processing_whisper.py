import math

import numpy as np
from transformers import WhisperProcessor


class WhisperPrePostProcessor(WhisperProcessor):
    def chunk_iter_with_batch(self, inputs, chunk_len, stride_left, stride_right, batch_size):
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
            processed = self.feature_extractor(
                chunks, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="np"
            )

            _stride_left = np.where(chunk_start_idx == 0, 0, stride_left)
            is_last = np.where(stride_right > 0, chunk_end_idx > inputs_len, chunk_end_idx >= inputs_len)
            _stride_right = np.where(is_last, 0, stride_right)

            chunk_lens = [chunk.shape[0] for chunk in chunks]
            strides = [
                (int(chunk_l), int(_stride_l), int(_stride_r))
                for chunk_l, _stride_l, _stride_r in zip(chunk_lens, _stride_left, _stride_right)
            ]

            yield {"stride": strides, **processed}

    def preprocess_batch(self, inputs, chunk_length_s=0, stride_length_s=None, batch_size=None):
        stride = None
        if isinstance(inputs, dict):
            stride = inputs.pop("stride", None)
            # Accepting `"array"` which is the key defined in `datasets` for
            # better integration
            if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
                raise ValueError(
                    "When passing a dictionary to FlaxWhisperPipline, the dict needs to contain a "
                    '"raw" or "array" key containing the numpy array representing the audio, and a "sampling_rate" key '
                    "containing the sampling rate associated with the audio array."
                )

            _inputs = inputs.pop("raw", None)
            if _inputs is None:
                # Remove path which will not be used from `datasets`.
                inputs.pop("path", None)
                _inputs = inputs.pop("array", None)
            in_sampling_rate = inputs.pop("sampling_rate")
            inputs = _inputs

            if in_sampling_rate != self.feature_extractor.sampling_rate:
                try:
                    import librosa
                except ImportError as err:
                    raise ImportError(
                        "To support resampling audio files, please install 'librosa' and 'soundfile'."
                    ) from err

                inputs = librosa.resample(
                    inputs, orig_sr=in_sampling_rate, target_sr=self.feature_extractor.sampling_rate
                )
                ratio = self.feature_extractor.sampling_rate / in_sampling_rate
            else:
                ratio = 1

        if not isinstance(inputs, np.ndarray):
            raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`.")
        if len(inputs.shape) != 1:
            raise ValueError(
                f"We expect a single channel audio input for the Flax Whisper API, got {len(inputs.shape)} channels."
            )

        if stride is not None:
            if stride[0] + stride[1] > inputs.shape[0]:
                raise ValueError("Stride is too large for input.")

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
                raise ValueError("Chunk length must be superior to stride length.")

            for item in self.chunk_iter_with_batch(
                inputs,
                chunk_len,
                stride_left,
                stride_right,
                batch_size,
            ):
                yield item
        else:
            processed = self.feature_extractor(
                inputs, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="np"
            )
            if stride is not None:
                processed["stride"] = stride
            yield processed

    def postprocess(self, model_outputs, return_timestamps=None, return_language=None):
        # unpack the outputs from list(dict(list)) to list(dict)
        model_outputs = [dict(zip(output, t)) for output in model_outputs for t in zip(*output.values())]

        time_precision = self.feature_extractor.chunk_length / 1500  # max source positions = 1500
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
