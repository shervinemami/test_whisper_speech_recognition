#!/usr/bin/env python3
# Ultra simple whisper test app, by Shervin Emami, Oct 2022.
# Use "benchmark_dictate.py" instead of this app, if you want fast inference!

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
    filename = sys.argv[2]

    start_inference = time.perf_counter()
    # "transcribe" will use GreedyDecoder (fast) if beam_size=None, or BeamSearch (slower but more reliable) if beam_size=5 or similar.
    result = model.transcribe(filename, fp16=False, beam_size=None)
    elapsed_inference = time.perf_counter() - start_inference
    print("[Wall clock time:", elapsed_inference, "seconds]")
    print("  --> ", result["text"])
    print()

