#!/usr/bin/env python3
# Simple whisper test app, using decode instead of transcode. by Shervin Emami, Oct 2022.

import whisper
import time
import sys

if len(sys.argv) <= 2:
    print("usage: " + sys.argv[0] + " <model> <audio>")
    print("eg:  python3 " + sys.argv[0] + " medium.en shervstest.wav")
    sys.exit(1)

print("Loading pytorch model '" + sys.argv[1] + "' during startup. Takes a long time. Also expect the first transciption to be slower than usual, since the GPU must load drivers and ramp up its clocks.")
model = whisper.load_model(sys.argv[1])

# Run the same thing twice, since the first time includes initialisation delays
for i in [0, 1]:
    start_inference = time.perf_counter()

    # Load the file and convert it into an audio buffer that's exactly 30 seconds long, since that is what whisper is trained for.
    filename = sys.argv[2]
    audio = whisper.load_audio(filename)
    audio = whisper.pad_or_trim(audio)

    # Make log-mel spectrogram and move it to the same device as the model (GPU)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Whisper allows passing "prompt" that is intended to be the previous sentence or some similar related text, to give a hint
    # about what it should expect. This includes formatting, so for example giving a hint of "40's" can push whisper closer to
    # decoding the phrase "forties" as "40's" instead of "40s".
    # Since I want a lot of commas but not fullstops or capitalising of phrases, I'm using a hint_prompt this way.
    hint_prompt="oh OK yeah sure, in my 40's I mostly benchmarked a profile of ARM CPU core optimisation"

    # Decode the audio.
    # "decode" will use GreedyDecoder (fast) if beam_size=None, or it will use BeamSearch (slower but more reliable) if beam_size=5 or similar.
    options = whisper.DecodingOptions(language="en", fp16=False, prompt=hint_prompt, best_of=None, beam_size=3, temperature=0.0, patience=1.3)
    result = whisper.decode(model, mel, options)

    elapsed_inference = time.perf_counter() - start_inference
    print("[Wall clock time:", elapsed_inference, "seconds]")
    print("  --> ", result.text)
    print()

