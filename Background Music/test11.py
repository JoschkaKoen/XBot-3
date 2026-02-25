
from pydub import AudioSegment

# Load first audio file
audio1 = AudioSegment.from_wav("x.wav")

fade_out_duration = 1000

# Create 2 seconds of silence
silence = AudioSegment.silent(duration=fade_out_duration)

# Append the silence to audio1
audio1_with_silence = audio1 + silence

# Load second audio file
audio2 = AudioSegment.from_mp3("music.mp3")

# Decrease the volume of audio2 by 10 dB
gain_reduction = -8

audio2 = audio2.apply_gain(gain_reduction)

# If audio2 is longer than audio1_with_silence, reduce its length to match audio1_with_silence
if len(audio2) > len(audio1_with_silence):
    audio2 = audio2[:len(audio1_with_silence)]

# Fade out the last 2 seconds of audio2
audio2 = audio2.fade_out(fade_out_duration)

# Overlay audio1_with_silence and audio2
combined = audio1_with_silence.overlay(audio2)

# Export combined audio
combined.export("combined.wav", format='wav')

