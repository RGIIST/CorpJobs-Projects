from pydub import AudioSegment

# Load the original song file
original_song = AudioSegment.from_file("original_song.mp3")

# Change the pitch of the song by -5 semitones (lowering it)
emotional_song = original_song.pitch_shift(-5)

# Save the modified version as a new file
emotional_song.export("emotional_song.mp3", format="mp3")
