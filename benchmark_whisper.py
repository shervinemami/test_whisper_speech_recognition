#!/usr/bin/env python3
# whisper benchmarking app, by Shervin Emami, Oct 2022.
#
# I realised I can get much faster speed by using "whisper.decode" instead of "whisper.transcribe", and also much faster results on my GPU using FP32 not FP16.
# So this script compares the speed of many different decoding options.

import whisper
import time
import sys

## Debugger
#import ipdb
#ipdb.set_trace()

if len(sys.argv) <= 3:
    print("usage: " + sys.argv[0] + " <model> <audio1> <audio2>")
    print("eg:  python3 benchmark_whisper.py medium.en shervstest.wav shervstest2.wav")
    sys.exit(1)

print("Loading pytorch model '" + sys.argv[1] + "' during startup. Takes a long time. Also expect the first transciption to be slower than usual, since the GPU must load drivers and ramp up its clocks.")
model = whisper.load_model(sys.argv[1])

# Compare many different combinations of settings. Also run the first test atleast twice, since the first execution includes GPU initialisation delays.
for i in range(64):
    print(i)
    start_prog = time.perf_counter()

    # Alternate input files with each iteration, so it's less likely to be getting hidden speedups by caching it across iterations.
    if (i & 1) == 0:
        audio = whisper.load_audio(sys.argv[2])
    else:
        audio = whisper.load_audio(sys.argv[3])
    # Pad/trim it to fit 30 seconds just like the training set.
    audio = whisper.pad_or_trim(audio)

    start_inference = time.perf_counter()

    # Make log-mel spectrogram and move it to the same device as the model (GPU)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect the spoken language
    #_, probs = model.detect_language(mel)
    #print(f"Detected language: {max(probs, key=probs.get)}")

    # Whisper allows passing "prompt" that is intended to be the previous sentence or some similar related text, to give a hint 
    # about what it should expect. This includes formatting, so for example giving a hint of "40's" can push whisper closer to
    # decoding the phrase "forties" as "40's" instead of "40s".
    #hint_prompt="Good Shervin Hello Processors"
    hint_prompt="oh OK yeah sure, in my 40's I mostly benchmarked a profile of ARM CPU core optimisation"

    # Alternate through many modes.
    # beam_size and best_of are mutually-exclusive, so do one or the other. Let's compare beam_size of 2 & 3 & 5, and best_of of 2 & 3 & 5.
    # Since the first iteration is slower than all later iterations, due to driver loading, let's do the first case (beam_size=None) twice.
    # So we have 8 options to alternate (including disabling everything), hence a mask of 8.
    beam_size = None
    best_of = None
    temperature = 0.0
    patience = None
    if (i & 7) == 0:     
        True  # Do nothing
    elif (i & 7) == 1:
        True  # Do nothing
    elif (i & 7) == 2:
        beam_size=2
    elif (i & 7) == 3:
        beam_size=3
    elif (i & 7) == 4:
        beam_size=5
    elif (i & 7) == 5:
        best_of=2
        temperature = 0.1   # Need temperature > 0 when using best_of
    elif (i & 7) == 6:
        best_of=3
        temperature = 0.1   # Need temperature > 0 when using best_of
    elif (i & 7) == 7:
        best_of=5
        temperature = 0.1   # Need temperature > 0 when using best_of

    # Alternate some other modes simply on or off.
    fp16 = False
    if (i & 8) == 0:
        fp16 = True;
    if (i & 16) == 0:
        temperature = 0.3;
    if (i & 32) == 0 and beam_size:   # patience only works when beam_size is given
        patience = 1.3    # I'm not sure but I think patience value needs to be atleast 1.0. See "https://github.com/openai/whisper/discussions/154"

    # Decode the audio
    options = whisper.DecodingOptions(language="en", fp16=fp16, prompt=hint_prompt, best_of=best_of, beam_size=beam_size, temperature=temperature, patience=patience)
    result = whisper.decode(model, mel, options)

    elapsed_inference = time.perf_counter() - start_inference
    elapsed_prog = time.perf_counter() - start_prog
    #print("[Inference clock time:", elapsed_inference, "seconds]")
    print("[Wall clock time:", elapsed_prog, "seconds]")

    # Print the recognized text
    print("  --> ", result.text)
    print()

