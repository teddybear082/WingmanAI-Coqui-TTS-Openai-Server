

# OpenAI-Compatible Server for Coqui-TTS (XTTS2-Focused)

## What It Is

This repo contains an OpenAI-compatible server for [Coqui-TTS](https://github.com/idiap/coqui-ai-TTS), adapted from
[`TTS/server/server.py`](https://github.com/idiap/coqui-ai-TTS/blob/dev/TTS/server/server.py).

## What It Does

This server allows you to use XTTS2 local TTS models as a drop-in replacement for OpenAI TTS models.

The primary use case is integration with **[WingmanAI](https://github.com/ShipBit/wingman-ai)**
([wingman-ai.com](https://www.wingman-ai.com/)), offering:

* Local voice cloning
* Additional TTS options
* No reliance on paid services like ElevenLabs

## Key Enhancements Over Base Repo

* `--lowvram` mode: moves TTS model to CPU when idle (saves \~1.5GB VRAM with XTTS2)
* Ensures correct language segmenter is used for splitting long text
* *(Planned)* Support for pre-made XTTS2 latents in generation

---

## How to Install

You have three installation options:

1. ✅ Premade `.exe` for Windows (Experimental)
2. ⚙️ Use original `idiap/coqui-ai-TTS` server with Python
3. 🛠️ Use custom server from this repo with Python

---

### 🪟 Option 1: Premade `.exe` (Windows Only)

**Pros:**

* No Python/coding knowledge needed
* Mostly pre-packaged
* Quickest setup

**Cons:**

* Antivirus may flag the `.exe`
* Minimal testing
* No auto-updates
* Windows only
* Trust required for the download

**Installation Steps:**

1. [Download ZIP (\~5GB)](https://mega.nz/file/1f8nQRTD#JfjgLrk2Ml1o3CkZj01Rj_Zk70RxaZ8nEdShGwZbI3Y)
2. Unzip anywhere (avoid OneDrive-controlled folders)
3. If warned, click “Keep Anyway”
4. Double-click `run_server.bat`
5. Allow network access when prompted
6. Follow menu to select language and GPU/CPU

You’re now running! 🎉 Proceed to **WingmanAI Configuration**.

---

### 🐍 Option 2: Use `idiap/coqui-ai-TTS` Server with Python

**Pros:**

* Trusted, long-standing repo
* Open source
* Automatic updates
* Works on all OS

**Cons:**

* No `lowvram` mode (uses \~3-4GB VRAM idle on GPU)
* No support for pre-made latents

**Installation Steps:**

1. Create a folder (e.g. `Coqui-TTS-Server`)
2. Install [`pyenv-win`](https://github.com/pyenv-win/pyenv-win)
3. Open terminal in that folder
4. Run:

   ```bash
   pyenv install 3.11.7
   pyenv local 3.11.7
   python -m venv venv
   .\venv\Scripts\activate
   ```
5. (Optional for NVIDIA GPU):

   ```bash
   pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
   ```
6. Then:

   ```bash
   pip install coqui-tts[server,languages]
   ```

**To Run the Server:**

```bash
.\venv\Scripts\activate
tts-server --model_name tts_models/multilingual/multi-dataset/xtts_v2
```

Optional flags:

* Add `--use_cuda` to run on GPU
* Add `--language_idx de` (or other language code)

Example:
```tts-server --model_name tts_models/multilingual/multi-dataset/xtts_v2 --use_cuda --language_idx de```

On the first run, the program should automatically install the TTS model (XTTS2).  You may have to indicate consent to the license during the download process.

You’re now running! 🎉 Proceed to **WingmanAI Configuration**.

---

### 🛠️ Option 3: Use Custom Server from This Repo

**Pros:**

* Custom WingmanAI features (e.g., lowvram)
* Open source
* Cross-platform support

**Cons:**

* Not automatically synced with base repo
* Requires more steps
* Trust needed (or read the code)

**Installation Steps:**

1. [Download this repo as ZIP](https://github.com/your-repo-url)
2. Unzip (avoid OneDrive folders)
3. Install [`pyenv-win`](https://github.com/pyenv-win/pyenv-win)
4. Open terminal in the unzipped folder
5. Run:

   ```bash
   pyenv install 3.11.7
   pyenv local 3.11.7
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
6. Download XTTS2 model files from:
   [huggingface.co/coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2/tree/main)
   into the `xtts_model` folder

**To Run the Server:**

1. Open the project folder
2. Double-click `run_server_with_python.bat`
3. Follow prompts to choose language and GPU/CPU

You’re now running! 🎉 Proceed to **WingmanAI Configuration**.

---

## 🛠️ WingmanAI Configuration

1. Start the TTS server using any method above

2. Open **WingmanAI**

3. Choose a Wingman and click the 🔧 config wrench

4. Under **Text to Speech**, choose `Local OpenAI Compatible TTS`

5. Click the ⚙️ configuration wheel

6. Enter:

   * **URL:** `http://localhost:5002/v1`
   * **Model:** `XTTS2` (or anything else; just a placeholder)

7. Adjust **Speed** to control speech rate

8. Choose a **Voice** from the built-ins, or:

   * Use a `.wav` file (1 voice sample)
   * Or a folder of `.wav` files (multiple samples)

9. Enter the path to the `.wav` or folder (use `/`, not `\`)

10. Save your Wingman

Your Wingman now speaks with **XTTS2**! 🗣️✨
