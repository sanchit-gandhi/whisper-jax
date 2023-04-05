import time

from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor


BATCH_SIZES = [4, 8, 16, 32, 64, 128]
NUM_BATCHES = 100
NUM_TOKENS = 25

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
model.to("cuda").half()

# processors/tokenizers are the same for all models, so just load from tiny and preprocess once
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")


def preprocess(batch):
    batch["input_features"] = processor(
        batch["audio"]["array"], sampling_rate=16000, return_tensors="pt"
    ).input_features[0]
    return batch


# load a dataset of 73 audio samples
librispeech = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset_processed = librispeech.map(preprocess, remove_columns=librispeech.column_names)

for batch_size in BATCH_SIZES:
    eval_dataset = dataset_processed.select(range(batch_size // 2))
    eval_dataset = concatenate_datasets([eval_dataset for _ in range(2 * NUM_BATCHES)])

    dataloader = DataLoader(
        dataset=eval_dataset.with_format("torch"), batch_size=batch_size, num_workers=4, pin_memory=True
    )

    # generate
    start = time.time()
    for batch in dataloader:
        input_features = batch["input_features"].to("cuda").half()
        out = model.generate(input_features, max_new_tokens=NUM_TOKENS, min_new_tokens=NUM_TOKENS)
    runtime = time.time() - start
    print(f"{batch_size}: {runtime:.06}")
