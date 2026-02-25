

X_DURATION=$(ffprobe -i x.wav -show_entries format=duration -v quiet -of csv="p=0")

ffmpeg -i x.wav -af "apad=pad_len=2*44100" x_padded.wav

DELAY=$(bc <<< "$X_DURATION*1000+2000")

ffmpeg -i music.mp3 -af "adelay=$DELAY|$DELAY,afade=t=out:st=$(bc <<< "$X_DURATION+2-2"):d=2" music_delayed_faded.mp3
