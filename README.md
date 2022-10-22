Several simple scripts for experimenting with OpenAI's [whisper](https://github.com/openai/whisper) speech recognition dictation / transcription / translation library.

## Example commands:

```bash
./test_transcribe.py medium.en shervstest.wav

./test_decode.py medium.en shervstest.wav

./benchmark_whisper.py medium.en shervstest.wav shervstest2.wav
```

Note that benchmark_whisper.py requires 2 audio files as arguments, to allow alternating between them. But you can pass the same file to both args if you only have a single audio file.

## Example results:
On my computer with a fast CPU and a slightly old "GTX 1660 Super" GPU with 6GB GDDR, here are some different results I get for different decoder options, all on the same "medium.en" model and "shervstest.wav" 10 second audio input:
```
                                                         TIME:           GPU MEM USED:
    "medium.en" model on GPU, FP32, GreedyDecoder:       1.1 seconds     5693 MiB GDDR
    "medium.en" model on GPU, FP32, BestOf5Decoder:      2.5 seconds     5733 MiB GDDR
    "medium.en" model on GPU, FP32, BeamSize3Decoder:    2.3 seconds     5733 MiB GDDR
    "medium.en" model on GPU, FP32, BeamSize5Decoder:    3.9 seconds     5933 MiB GDDR
    "medium.en" model on GPU, FP32, BeamSize7Decoder:    <crashed. 6GB is not enough GDDR for this!>
    "medium.en" model on GPU, FP16, GreedyDecoder:       3.8 seconds     5693 MiB GDDR
    "medium.en" model on GPU, FP16, BestOf5Decoder:      6.6 seconds     5733 MiB GDDR
    "medium.en" model on GPU, FP16, BeamSize5Decoder:    8.3 seconds     5747 MiB GDDR
    "medium.en" model on GPU, FP16, BeamSize7Decoder:    8.7 seconds     5921 MiB GDDR
```

Note that in my testing with a handful of 10 second English audio files, the `medium.en` model gave very accurate transcription no matter what decoder options I used, the accuracy seems to only change slightly while the speed varies significantly. So I'll personally be going with one the faster decoder options. Whereas using smaller models than `medium.en` (such as `base.en`) did start having a noticeable reduction in accuracy.

The discussions on the whisper GitHub mention that `beam_size=5, temperature=0.0` is considered the most accurate decoder in general, based on their short testing, but it will vary a lot depending on many things. Patience can probably also help a little. So I'd generally recommend to go with the Greedy Decoder (`beam_size=None, temperature=0, best_of=None`) if you want max speed, or use beam_size of atleast 3 if you want a bit more accuracy even if it's a few times slower.

For example, here are a set of options that takes around 2.5 seconds on my GPU for 10 seconds of audio:
```
[Decoding Options:  DecodingOptions(task='transcribe', language='en', temperature=0.0, sample_len=None, best_of=None, beam_size=3, patience=1.3, length_penalty=None, prompt="Profiling ARM CPU and GPU cores in my 40's", prefix=None, suppress_blank=True, suppress_tokens='-1', without_timestamps=False, max_initial_timestamp=1.0, fp16=False) ]
```

(See my longer message at [whisper#391](https://github.com/openai/whisper/discussions/391) ).
