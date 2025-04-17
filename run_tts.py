from TTS.api import TTS
#Load the model
tts = TTS(model_name='tts_models/en/ljspeech/glow-tts')
#We transform the text to an audio file
tts.tts_to_file(text="Hello Unity, this is a test.", file_path="test_output.wav")

