#!flask/bin/python

"""TTS demo server."""

import argparse
import gc
import io
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs

import torch
import torchaudio

try:
    from flask import Flask, render_template, render_template_string, request, send_file, jsonify, Response
except ImportError as e:
    msg = "Server requires requires flask, use `pip install coqui-tts[server]`"
    raise ImportError(msg) from e

from TTS.api import TTS
from TTS.utils.generic_utils import ConsoleFormatter, setup_logger
from TTS.utils.manage import ModelManager
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
logger = logging.getLogger(__name__)
setup_logger("TTS", level=logging.INFO, stream=sys.stdout, formatter=ConsoleFormatter())
xtts_text_cleaner = VoiceBpeTokenizer()

def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="list available pre-trained tts and vocoder models.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Name of one of the pre-trained tts models in format <language>/<dataset>/<model_name>",
    )
    parser.add_argument("--vocoder_name", type=str, default=None, help="Name of one of the released vocoder models.")
    parser.add_argument("--speaker_idx", type=str, default=None, help="Default speaker ID for multi-speaker models.")

    # Args for running custom models
    parser.add_argument("--config_path", default=None, type=str, help="Path to model config file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model file.",
    )
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).",
        default=None,
    )
    parser.add_argument("--vocoder_config_path", type=str, help="Path to vocoder model config file.", default=None)
    parser.add_argument("--speakers_file_path", type=str, help="JSON file for multi-speaker model.", default=None)
    parser.add_argument("--port", type=int, default=5002, help="port to listen on.")
    parser.add_argument("--device", type=str, help="Device to run model on. Choices: cpu, cuda, cuda:0, cuda:1", default="cpu")
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default=False, help="true to use CUDA.")
    parser.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, default=False, help="true to enable Flask debug mode."
    )
    parser.add_argument(
        "--show_details", action=argparse.BooleanOptionalAction, default=False, help="Generate model detail page."
    )
    parser.add_argument("--language_idx", type=str, help="Default language ID for multilingual models.", default="en")
    parser.add_argument("--lowvram", action=argparse.BooleanOptionalAction, default=False, help="Use low vram mode, switches device to cpu when idle.")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False, help="Use streaming mode. Only works with XTTS2.")
    return parser


# parse the args
args = create_argparser().parse_args()

manager = ModelManager(models_file=TTS.get_models_file_path())

# update in-use models to the specified released models.
model_path = None
config_path = None
speakers_file_path = None
vocoder_path = None
vocoder_config_path = None



# CASE1: list pre-trained TTS models
if args.list_models:
    manager.list_models()
    sys.exit(0)

# CASE2: load models
device = args.device
if args.use_cuda:
    warnings.warn("`--use_cuda` is deprecated, use `--device cuda` instead.", DeprecationWarning, stacklevel=2)
    if not "cuda" in device:
        device = "cuda"
current_device = device

model_name = args.model_name if args.model_path is None else None
api = TTS(
    model_name=model_name,
    model_path=args.model_path,
    config_path=args.config_path,
    vocoder_name=args.vocoder_name,
    vocoder_path=args.vocoder_path,
    vocoder_config_path=args.vocoder_config_path,
    speakers_file_path=args.speakers_file_path,
    # language_ids_file_path=args.language_ids_file_path,
).to(device)

# TODO: set this from SpeakerManager
use_gst = api.synthesizer.tts_config.get("use_gst", False)
supports_cloning = api.synthesizer.tts_config.get("model", "") in ["xtts", "bark"]

try:
    import pyi_splash
    pyi_splash.close()
except:
    pass

app = Flask(__name__)


def handle_vram_change(desired_device: str):
    current_device = str(api.synthesizer.tts_model.device)
    if torch.cuda.is_available():
        if "cuda" in desired_device:
            if "cuda" not in current_device:
                api.synthesizer.tts_model.to(desired_device)
                gc.collect()
                current_device = desired_device
        if "cpu" in desired_device:
            if "cpu" not in current_device:
                api.synthesizer.tts_model.to(desired_device)
                torch.cuda.empty_cache()
                gc.collect()
                current_device = desired_device

# Move out of vram if low vram mode until ready to generate
if args.lowvram and "cuda" in device:
    handle_vram_change("cpu")


def style_wav_uri_to_dict(style_wav: str) -> str | dict:
    """Transform an uri style_wav, in either a string (path to wav file to be use for style transfer)
    or a dict (gst tokens/values to be use for styling)

    Args:
        style_wav (str): uri

    Returns:
        Union[str, dict]: path to file (str) or gst style (dict)
    """
    if style_wav:
        if os.path.isfile(style_wav) and style_wav.endswith(".wav"):
            return style_wav  # style_wav is a .wav file located on the server

        style_wav = json.loads(style_wav)
        return style_wav  # style_wav is a gst dictionary with {token1_id : token1_weigth, ...}
    return None


@app.route("/")
def index():
    return render_template(
        "index.html",
        show_details=args.show_details,
        use_multi_speaker=api.is_multi_speaker,
        use_multi_language=api.is_multi_lingual,
        speaker_ids=api.speakers,
        language_ids=api.languages,
        use_gst=use_gst,
        supports_cloning=supports_cloning,
    )


@app.route("/details")
def details():
    model_config = api.synthesizer.tts_config
    vocoder_config = api.synthesizer.vocoder_config or None

    return render_template(
        "details.html",
        show_details=args.show_details,
        model_config=model_config,
        vocoder_config=vocoder_config,
        args=args.__dict__,
    )


lock = Lock()


@app.route("/api/tts", methods=["GET", "POST"])
def tts():
    with lock:
        if args.lowvram:
            handle_vram_change(device)   
        text = request.headers.get("text") or request.values.get("text", "")
        speaker_idx = (
            request.headers.get("speaker-id") or request.values.get("speaker_id", args.speaker_idx)
            if api.is_multi_speaker
            else None
        )
        if speaker_idx == "":
            speaker_idx = None

        language_idx = (
            request.headers.get("language-id") or request.values.get("language_id", args.language_idx)
            if api.is_multi_lingual
            else None
        )
        if language_idx == "":
            language_idx = None

        style_wav = request.headers.get("style-wav") or request.values.get("style_wav", "")
        style_wav = style_wav_uri_to_dict(style_wav)
        speaker_wav = request.headers.get("speaker-wav") or request.values.get("speaker_wav", "")
        
        if not text.strip():
            return {"error": "Text parameter is required"}, 400
            
        logger.info("Model input: %s", text)
        logger.info("Speaker idx: %s", speaker_idx)
        logger.info("Speaker wav: %s", speaker_wav)
        logger.info("Language idx: %s", language_idx)
        # Clean text for xtts if using xtts
        if api.synthesizer.tts_config.get("model", "") == "xtts":
            text = xtts_text_cleaner.preprocess_text(text, language_idx)
        # Use proper segmenter to split sentences for the chosen language
        try:
            api.synthesizer.seg = api.synthesizer._get_segmenter(language_idx)
        except Exception as e:
            logger.info(f"Getting segmenter for language: {language_idx} failed, defaulting to English.  Reason: {e}.")
            api.synthesizer.seg = api.synthesizer._get_segmenter("en")
        try:
            wavs = api.tts(text, speaker=speaker_idx, language=language_idx, style_wav=style_wav, speaker_wav=speaker_wav)
        except Exception as e:
            logger.error("TTS synthesis failed: %s", str(e))
            return {"error": f"TTS synthesis failed: {str(e)}"}, 500

        out = io.BytesIO()
        api.synthesizer.save_wav(wavs, out)
        if args.lowvram:
            handle_vram_change("cpu")
    return send_file(out, mimetype="audio/wav")


# Basic MaryTTS compatibility layer


@app.route("/locales", methods=["GET"])
def mary_tts_api_locales():
    """MaryTTS-compatible /locales endpoint"""
    # NOTE: We currently assume there is only one model active at the same time
    if args.model_name is not None:
        model_details = args.model_name.split("/")
    else:
        model_details = ["", "en", "", "default"]
    return render_template_string("{{ locale }}\n", locale=model_details[1])


@app.route("/voices", methods=["GET"])
def mary_tts_api_voices():
    """MaryTTS-compatible /voices endpoint"""
    # NOTE: We currently assume there is only one model active at the same time
    if args.model_name is not None:
        model_details = args.model_name.split("/")
    else:
        model_details = ["", "en", "", "default"]
    if api.is_multi_speaker:
        return render_template_string(
            "{% for speaker in speakers %}{{ speaker }} {{ locale }} {{ gender }}\n{% endfor %}",
            speakers=api.speakers,
            locale=model_details[1],
            gender="u",
        )
    return render_template_string(
        "{{ name }} {{ locale }} {{ gender }}\n", name=model_details[3], locale=model_details[1], gender="u"
    )


@app.route("/process", methods=["GET", "POST"])
def mary_tts_api_process():
    """MaryTTS-compatible /process endpoint"""
    with lock:
        if args.lowvram:
            handle_vram_change(device)        
        if request.method == "POST":
            data = parse_qs(request.get_data(as_text=True))
            speaker_idx = data.get("VOICE", [args.speaker_idx])[0]
            # NOTE: we ignore parameter LOCALE for now since we have only one active model
            text = data.get("INPUT_TEXT", [""])[0]
        else:
            text = request.args.get("INPUT_TEXT", "")
            speaker_idx = request.args.get("VOICE", args.speaker_idx)

        logger.info("Model input: %s", text)
        logger.info("Speaker idx: %s", speaker_idx)
        wavs = api.tts(text, speaker=speaker_idx)
        out = io.BytesIO()
        api.synthesizer.save_wav(wavs, out)
        if args.lowvram:
            handle_vram_change("cpu")
    return send_file(out, mimetype="audio/wav")


# OpenAI-compatible Speech API
@app.route("/v1/audio/speech", methods=["POST"])
def openai_tts():
    """
    POST /v1/audio/speech
    {
      "model": "tts-1",           # ignored, defaults to args.model_name
      "voice": "alloy",           # required: a speaker ID or a file/folder for voice cloning
      "input": "Hello world!",    # required text to speak
      "response_format": "wav"    # optional: wav, opus, aac, flac, wav, pcm (alternative to format)
    }
    """
    payload = request.get_json(force=True)
    logger.info(payload)
    text = payload.get("input") or ""
    speaker_idx = payload.get("voice", args.speaker_idx) if api.is_multi_speaker else None
    fmt = payload.get("response_format", "mp3").lower()  # OpenAI default is .mp3
    speed = payload.get("speed", 1.0)
    language_idx = args.language_idx if api.is_multi_lingual else None
    stream = payload.get("stream") or args.stream or False

    speaker_wav = None
    if speaker_idx is not None:
        voice_path = Path(speaker_idx)
        if voice_path.exists() and supports_cloning:
            speaker_wav = str(voice_path) if voice_path.is_file() else [str(w) for w in voice_path.glob("*.wav")]
            speaker_idx = None

    # here we ignore payload["model"] since its loaded at startup

    # Get format of output audio
    mimetypes = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "pcm": "audio/L16",
    }
    mimetype = mimetypes.get(fmt, "audio/mpeg")
    
    def _save_audio(waveform, sample_rate, format_args):
        buf = io.BytesIO()
        torchaudio.save(buf, waveform, sample_rate, **format_args)
        buf.seek(0)
        return buf

    def _save_pcm(waveform):
        """Raw PCM (16-bit little-endian)."""
        waveform_int16 = (waveform * 32767).to(torch.int16)
        buf = io.BytesIO()
        buf.write(waveform_int16.numpy().tobytes())
        buf.seek(0)
        return buf

    with lock:
        if args.lowvram:
            handle_vram_change(device)
        logger.info("Model input: %s", text)
        logger.info("Speaker idx: %s", speaker_idx)
        logger.info("Speaker wav: %s", speaker_wav)
        logger.info("Language idx: %s", language_idx)
        # Clean text for xtts if using xtts
        if api.synthesizer.tts_config.get("model", "") == "xtts":
            text = xtts_text_cleaner.preprocess_text(text, language_idx)
        # Use proper segmenter to split sentences for the chosen language
        try:
            api.synthesizer.seg = api.synthesizer._get_segmenter(language_idx)
        except Exception as e:
            logger.info(f"Getting segmenter for language: {language_idx} failed, defaulting to English.  Reason: {e}.")
            api.synthesizer.seg = api.synthesizer._get_segmenter("en")
        # If streaming, generate stream chunk-by-chunk for each sentence
        if stream:
            gpt_cond_latent, speaker_embedding = (None, None)
            # if cloning wav provided, handle either single file or directory
            if speaker_wav:
                if isinstance(speaker_wav, str):
                    voices_list = [speaker_wav]
                else:
                    voices_list = speaker_wav
                gpt_cond_latent, speaker_embedding = (
                    api.synthesizer.tts_model.get_conditioning_latents(
                        audio_path=voices_list
                    )
                )
            # if built in xtts2 voice, generate latents for voice
            else:
                speakers_dir = os.path.join(
                    Path(args.model_path), "speakers_xtts.pth"
                )
                speaker_data = torch.load(speakers_dir)
                speaker = list(speaker_data[speaker_idx].values())
                gpt_cond_latent = speaker[0]
                speaker_embedding = speaker[1]

            # Split the input text into sentences
            sentences = api.synthesizer.split_into_sentences(text)
            logger.info(f"Split text into {len(sentences)} sentences for streaming.")

            def generate_chunks():
                # Loop over each sentence
                for i, sentence in enumerate(sentences):
                    # Skip empty or whitespace-only sentences
                    if not sentence.strip():
                        continue
                    
                    logger.info(f"Streaming sentence {i+1}/{len(sentences)}: '{sentence}'")
                    
                    # Get the audio stream iterator for the current sentence
                    waveform_iterator = api.synthesizer.tts_model.inference_stream(
                        sentence,
                        language_idx,
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        speed=speed,
                        stream_chunk_size=20, # Default XTTS stream chunk size
                    )
                    
                    # Yield all audio chunks from this sentence's stream
                    for chunk in waveform_iterator:
                        # The chunk is a tensor, convert it to the desired format
                        cpu_chunk = chunk.cpu()
                        
                        if cpu_chunk.ndim == 1:
                            cpu_chunk = cpu_chunk.unsqueeze(0)
                            
                        if fmt != "pcm":
                            # This part might be slow for real-time streaming if not using PCM
                            audio_buffer = _save_audio(cpu_chunk, api.synthesizer.output_sample_rate, {"format": fmt})
                        else:
                            audio_buffer = _save_pcm(cpu_chunk)
                        
                        yield audio_buffer.getvalue()
                
                # After all sentences, ensure VRAM is cleared if needed
                if args.lowvram:
                    handle_vram_change("cpu")

            return Response(generate_chunks(), mimetype=mimetype)
        
        # If not streaming, just generate on chunk with normal API
        else:
            wavs = api.tts(text, speaker=speaker_idx, language=language_idx, speaker_wav=speaker_wav, speed=speed)
            out = io.BytesIO()
            api.synthesizer.save_wav(wavs, out)
            out.seek(0)
            waveform, sample_rate = torchaudio.load(out)

            if fmt == "wav":
                out.seek(0)
                if args.lowvram:
                    handle_vram_change("cpu")
                return send_file(out, mimetype=mimetype)

            format_dispatch = {
                "mp3": lambda: _save_audio(waveform, sample_rate, {"format": "mp3"}),
                "opus": lambda: _save_audio(waveform, sample_rate, {"format": "ogg", "encoding": "opus"}),
                "aac": lambda: _save_audio(waveform, sample_rate, {"format": "mp4", "encoding": "aac"}),  # m4a container
                "flac": lambda: _save_audio(waveform, sample_rate, {"format": "flac"}),
                "pcm": lambda: _save_pcm(waveform),
            }

            # Check if format is supported
            if fmt not in format_dispatch:
                return "Unsupported format", 400

            # Generate and send file
            audio_buffer = format_dispatch[fmt]()
            if args.lowvram:
                handle_vram_change("cpu")
            return send_file(audio_buffer, mimetype=mimetype)

@app.route("/v1/models", methods=["GET"])
def openai_list_models():
    """
    GET /v1/models
    Returns a list of “models” – here we only expose our TTS model.
    """
    model_id = args.model_name or os.path.basename(args.model_path or "")
    return jsonify({
        "data": [
            {"id": model_id, "object": "model", "created":1234567890, "owned_by": "coqui-tts"}
        ]
    })


def main():
    app.run(debug=args.debug, host="::", port=args.port)


if __name__ == "__main__":
    main()