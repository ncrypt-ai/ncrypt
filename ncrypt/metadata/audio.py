import whisper

from ncrypt.utils import AUDIO_EXTENSIONS, UnsupportedExtensionError


def extract_subtitles(path: str, extension: str):
    if extension not in AUDIO_EXTENSIONS:
        raise UnsupportedExtensionError(f"The provided file extension is not supported for metadata: .{extension}")

    model = whisper.load_model("base")
    result = model.transcribe(path)

    return result["text"]
