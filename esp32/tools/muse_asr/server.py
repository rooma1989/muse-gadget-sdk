#!/usr/bin/env python3
"""Speech-to-text and text-to-speech for a Muse gadget without PSRAM.

The AI Passport (ESP32-C3, no PSRAM) reaches Muse through a relay with a long
round trip, and can neither upload a voice note fast enough nor receive the
16 KB frame Muse echoes it back with. Instead it streams the mic here (a nearby
server), gets the transcript back, and sends Muse plain text.

Listens on plain HTTP behind nginx (see protocol.py). POST /asr: SenseVoice-
Small through sherpa-onnx transcribes the mic. POST /tts: VITS voices through
sherpa-onnx read a reply aloud, sentence by sentence, as 16 kHz IMA ADPCM.

  MUSE_ASR_KEY_FILE  hex AES-256 key shared with the gadget (/etc/muse-asr/key)
  MUSE_ASR_MODEL     directory with model.int8.onnx and tokens.txt
  MUSE_TTS_DIR       directory holding the VITS voices' model directories (VOICES);
                     empty: no /tts
  MUSE_ASR_LISTEN    host:port (127.0.0.1:8102)
  MUSE_ASR_SOCKET    path of a Unix socket for local clients such as the TV media
                     server (/run/muse-asr/asr.sock); empty: none. See below.
  MUSE_ASR_SOCKET_GROUP  group the socket is given (e.g. tvserver); the process
                     must be a member. Empty: the process's own group.

The Unix socket speaks plain HTTP without the AES framing, because the clients
on this host have no key: whoever may open the socket (mode 0660, owner and
group) may transcribe, and nobody else can. POST /asr with a Content-Length
body of PCM16 LE mono at X-Sample-Rate (default 16000) answers 200 with
{"text": ...}; GET /health answers ok. TTS is not offered there.

Recognition has its own lock, separate from speech synthesis: a Muse reply being
read aloud (slower than real time on a small host) holds the TTS lock for each
sentence, and a TV's voice command must not wait behind it. The two run on
different sherpa-onnx models, so they may overlap; requests of the same kind
still go one at a time.

Transcripts and replies are never logged, only their lengths.
"""

import asyncio
import grp
import json
import logging
import os
import stat
import re
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as P  # noqa: E402

MAX_SECONDS = 20
TTS_TEXT_MAX = 1500          # bytes of reply text taken
TTS_SPEECH_MAX = 400         # characters read aloud; the rest stays on screen
TTS_RATE = 16000
ADPCM_FRAME = 2048           # bytes of ADPCM per frame: 256 ms
HEADER_LIMIT = 8192
READ_TIMEOUT = 30

log = logging.getLogger("muse_asr")


class HttpError(Exception):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status = status
        self.reason = reason


async def read_headers(reader):
    raw = await reader.readuntil(b"\r\n\r\n")
    if len(raw) > HEADER_LIMIT:
        raise HttpError(431, "headers too large")
    lines = raw.decode("latin-1").split("\r\n")
    method, path, _ = lines[0].split(" ", 2)
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    return method, path, headers


async def body_chunks(reader, headers):
    """Yields the request body as it arrives (chunked or Content-Length)."""
    if headers.get("transfer-encoding", "").lower() == "chunked":
        while True:
            size_line = await reader.readuntil(b"\r\n")
            size = int(size_line.split(b";")[0].strip(), 16)
            if size == 0:
                await reader.readuntil(b"\r\n")  # end of trailers
                return
            data = await reader.readexactly(size)
            await reader.readexactly(2)
            yield data
    else:
        left = int(headers.get("content-length", "0"))
        while left > 0:
            data = await reader.read(min(left, 65536))
            if not data:
                raise HttpError(400, "short body")
            left -= len(data)
            yield data


class Recognizer:
    def __init__(self, model_dir):
        import sherpa_onnx
        self.rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=os.path.join(model_dir, "model.int8.onnx"),
            tokens=os.path.join(model_dir, "tokens.txt"),
            num_threads=2,
            use_itn=True,
            language="auto",
        )

    def _decode(self, pcm, rate):
        import numpy as np
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        stream = self.rec.create_stream()
        stream.accept_waveform(rate, samples)
        self.rec.decode_stream(stream)
        return stream.result.text.strip()

    async def transcribe(self, pcm, rate):
        return await asyncio.get_running_loop().run_in_executor(None, self._decode, pcm, rate)


# ---- Reading a reply aloud ----

_SENTENCE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]*\n*|\n+")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_MARK = re.compile(r"(\*\*|__|`+|~~)")
_LINE_START = re.compile(r"^\s*(#+|[-*+>]|\d+[.)])\s+")
_EMOJI = re.compile("[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE0F\u200D\u2B00-\u2BFF]")


def speakable(segment):
    """What to say for one piece of Markdown-ish chat text."""
    t = _LINK.sub(r"\1", segment)
    t = _URL.sub("", t)
    t = _LINE_START.sub("", t)
    t = _MARK.sub("", t)
    t = _EMOJI.sub("", t)
    return " ".join(t.split())


# Where a clause ends: Chinese commas and colons, or ASCII ones before a space
# (so URLs, times and 3,000 stay whole).
_CLAUSE = re.compile(r"(?:[^，、：,:]|[,:](?!\s|$))+(?:[，、：]+|[,:]+(?=\s|$)|$)")
MIN_PIECE = 4     # characters; a shorter clause is said together with the next


def _words(text):
    return sum(ch.isalnum() for ch in text)


def segments(text):
    """Splits text into pieces to synthesize one at a time:
    [(what to say, UTF-8 byte offset where it ends)].

    Sentences are cut at their commas too, so each piece is short: the server
    (slower than real time on a weak CPU) synthesizes the next while the gadget
    plays this one. A clause under MIN_PIECE characters joins the next, except
    the reply's first, which goes alone so speech starts sooner."""
    out = []
    spoken = 0
    for m in _SENTENCE.finditer(text):
        start = m.start()
        for c in _CLAUSE.finditer(m.group(0)):
            end = m.start() + c.end()
            last = c.end() == len(m.group(0))
            say = speakable(text[start:end])
            if not last and _words(say) < (2 if not out else MIN_PIECE):
                continue                      # too short alone: keep gathering
            if _words(say) == 0:
                start = end
                continue
            if spoken + len(say) > TTS_SPEECH_MAX:
                return out
            spoken += len(say)
            out.append((say, len(text[:end].encode())))
            start = end
    return out


def resample(samples, rate):
    """float32 at `rate` -> int16 at 16 kHz, low-passed first so it doesn't alias."""
    import numpy as np
    x = np.asarray(samples, dtype=np.float32)
    if rate != TTS_RATE:
        cutoff = 0.45 * TTS_RATE / rate          # of the input rate
        n = np.arange(-50, 51)
        h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(len(n))
        x = np.convolve(x, (h / h.sum()).astype(np.float32), mode="same")
        t = np.arange(0, len(x) * TTS_RATE // rate) * (rate / TTS_RATE)
        x = np.interp(t, np.arange(len(x)), x).astype(np.float32)
    pcm = np.clip(x * 32767, -32768, 32767).astype(np.int16)
    return pcm[: len(pcm) // 2 * 2]


# The voices a gadget can pick, by the byte it sends: (model directory, speaker id).
VOICES = [("tts-ll", 0), ("tts-ll", 2), ("tts-fc", 0)]


def vits_config(d):
    import sherpa_onnx
    return sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=os.path.join(d, "model.onnx"),
                lexicon=os.path.join(d, "lexicon.txt"),
                tokens=os.path.join(d, "tokens.txt"),
                dict_dir=os.path.join(d, "dict"),
            ),
            num_threads=2,
        ),
        rule_fsts=",".join(os.path.join(d, f) for f in ("date.fst", "phone.fst", "number.fst")),
        max_num_sentences=1,
    )


class Speaker:
    """sherpa-onnx VITS voices under one directory (see VOICES). One model is
    kept loaded at a time: switching to the other takes a few seconds, once."""

    def __init__(self, root):
        self.root = root
        self.loaded = None
        self.tts = None
        self.load(VOICES[0][0])

    def load(self, name):
        if self.loaded != name:
            import sherpa_onnx
            self.tts = None
            t = time.monotonic()
            self.tts = sherpa_onnx.OfflineTts(vits_config(os.path.join(self.root, name)))
            self.loaded = name
            log.info("tts: loaded %s in %.1fs", name, time.monotonic() - t)

    def say(self, text, voice=0):
        name, sid = VOICES[voice if 0 <= voice < len(VOICES) else 0]
        self.load(name)
        audio = self.tts.generate(text, sid=sid, speed=1.0)
        return resample(audio.samples, audio.sample_rate)


MAX_LEAD = 3.0      # seconds of speech held back at most before streaming starts


class Pace:
    """How fast this host synthesizes, learned per voice: seconds of compute
    per second of speech, and seconds of speech per character."""

    def __init__(self):
        self.rtf = {}
        self.per_char = {}

    def lead(self, voice, chars):
        """How long to hold speech back so the rest keeps up once it starts."""
        speech = chars * self.per_char.get(voice, 0.25)
        return max(0.0, min(MAX_LEAD, speech * (self.rtf.get(voice, 1.2) - 1)))

    def learn(self, voice, chars, speech, compute):
        if speech <= 0 or chars <= 0:
            return
        for table, value in ((self.rtf, compute / speech), (self.per_char, speech / chars)):
            table[voice] = 0.7 * table.get(voice, value) + 0.3 * value


PACE = Pace()


async def speak(writer, key, hello, text, voice, speaker, lock, pace=PACE):
    """Streams the reply's speech: per piece a segment frame, then its audio.

    The host may synthesize slower than real time, so the first pieces are held
    back long enough (pace.lead) that, once playing, the rest keeps up."""
    loop = asyncio.get_running_loop()
    counter = P.SERVER_COUNTER
    adpcm = P.Adpcm()
    total = 0
    pieces = segments(text)
    t0 = time.monotonic()
    lead = pace.lead(voice, sum(len(say) for say, _ in pieces))
    held = []
    for say, end in pieces:
        t = time.monotonic()
        async with lock:
            pcm = await loop.run_in_executor(None, speaker.say, say, voice)
            data = await loop.run_in_executor(None, adpcm.encode, pcm)
        pace.learn(voice, len(say), len(pcm) / TTS_RATE, time.monotonic() - t)
        frames = [P.seal(key, hello, counter, P.T_SEGMENT, struct.pack("<IH", len(pcm), min(end, 0xFFFF)))]
        counter += 1
        for off in range(0, len(data), ADPCM_FRAME):
            frames.append(P.seal(key, hello, counter, P.T_AUDIO, data[off:off + ADPCM_FRAME]))
            counter += 1
        total += len(pcm)
        held += frames
        if time.monotonic() - t0 >= lead:
            writer.write(b"".join(held))
            held = []
            await writer.drain()
    writer.write(b"".join(held) + P.seal(key, hello, counter, P.T_END))
    await writer.drain()
    return total


async def read_text(key, reader, headers):
    """The /tts request body: the hello, then one frame of voice byte + text."""
    length = int(headers.get("content-length", "0"))
    if not P.HELLO_LEN < length <= TTS_TEXT_MAX + 64:
        raise P.ProtocolError("bad length")
    body = await reader.readexactly(length)
    hello = body[:P.HELLO_LEN]
    P.parse_hello(hello)
    frames = P.FrameReader()
    frames.feed(body[P.HELLO_LEN:])
    for ct in frames.frames():
        kind, data = P.open_frame(key, hello, 0, ct)
        if kind != P.T_TEXT or not data:
            break
        return hello, data[0], data[1:].decode("utf-8", "replace")
    raise P.ProtocolError("no text frame")


async def read_speech(key, chunks):
    """Opens the gadget's frames; returns (hello, rate, pcm)."""
    hello = b""
    reader = P.FrameReader()
    pcm = bytearray()
    counter = 0
    rate = 0
    async for data in chunks:
        if len(hello) < P.HELLO_LEN:
            take = P.HELLO_LEN - len(hello)
            hello += data[:take]
            data = data[take:]
            if len(hello) == P.HELLO_LEN:
                _, rate = P.parse_hello(hello)
        reader.feed(data)
        for ct in reader.frames():
            kind, payload = P.open_frame(key, hello, counter, ct)
            counter += 1
            if kind == P.T_AUDIO:
                pcm += payload
                if len(pcm) > MAX_SECONDS * rate * 2:
                    raise P.ProtocolError("too long")
            elif kind == P.T_END:
                return hello, rate, bytes(pcm)
            else:
                raise P.ProtocolError("unexpected frame")
    raise P.ProtocolError("no end frame")


async def respond(writer, status, body, ctype="application/octet-stream"):
    reason = {200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found",
              405: "Method Not Allowed", 500: "Internal Server Error"}.get(status, "Error")
    head = (f"HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n")
    writer.write(head.encode() + body)
    await writer.drain()


def make_handler(key, recognizer, speaker=None, lock=None, asr_lock=None):
    """`lock` serialises TTS, `asr_lock` serialises recognition (shared with the
    local handler, so the two entrances take turns on the one model)."""
    lock = lock or asyncio.Lock()
    asr_lock = asr_lock or asyncio.Lock()

    async def handle(reader, writer):
        t0 = time.monotonic()
        try:
            method, path, headers = await asyncio.wait_for(read_headers(reader), 10)
            if method == "GET" and path.rstrip("/").endswith("/health"):
                await respond(writer, 200, b"ok\n", "text/plain")
                return
            if method != "POST":
                raise HttpError(405, "POST only")
            if path.rstrip("/").endswith("/tts"):
                if not speaker:
                    raise HttpError(404, "no speech")
                hello, voice, text = await asyncio.wait_for(read_text(key, reader, headers), 10)
                writer.write(b"HTTP/1.0 200 OK\r\nContent-Type: application/octet-stream\r\n"
                             b"Connection: close\r\n\r\n")
                samples = await speak(writer, key, hello, text, voice, speaker, lock)
                log.info("tts: voice %d, %d bytes of text, %.2fs of speech in %.0f ms",
                         voice, len(text.encode()), samples / TTS_RATE, (time.monotonic() - t0) * 1000)
                return
            hello, rate, pcm = await asyncio.wait_for(
                read_speech(key, body_chunks(reader, headers)), READ_TIMEOUT)
            t1 = time.monotonic()
            async with asr_lock:
                text = await recognizer.transcribe(pcm, rate) if pcm else ""
            t2 = time.monotonic()
            reply = json.dumps({"text": text}, ensure_ascii=False).encode()
            await respond(writer, 200, P.seal(key, hello, P.SERVER_COUNTER, P.T_RESULT, reply))
            log.info("ok: %.2fs of %d Hz audio, upload %.0f ms, asr %.0f ms, %d chars",
                     len(pcm) / 2 / rate, rate, (t1 - t0) * 1000, (t2 - t1) * 1000, len(text))
        except P.ProtocolError as e:
            log.warning("rejected: %s", e)
            await respond(writer, 403, b"forbidden\n", "text/plain")
        except HttpError as e:
            await respond(writer, e.status, (e.reason + "\n").encode(), "text/plain")
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError, ValueError) as e:
            log.warning("dropped: %s", type(e).__name__)
        except Exception:
            log.exception("failed")
            try:
                await respond(writer, 500, b"error\n", "text/plain")
            except Exception:
                pass
        finally:
            try:
                writer.close()
            except Exception:
                pass
    return handle


LOCAL_RATES = (8000, 16000, 24000)


def make_local_handler(recognizer, asr_lock=None):
    """Plain POST /asr for clients on this host, over the Unix socket (see the module doc)."""
    asr_lock = asr_lock or asyncio.Lock()

    async def handle(reader, writer):
        t0 = time.monotonic()
        try:
            method, path, headers = await asyncio.wait_for(read_headers(reader), 10)
            route = path.split("?", 1)[0].rstrip("/")
            if method == "GET" and route.endswith("/health"):
                await respond(writer, 200, b"ok\n", "text/plain")
                return
            if method != "POST":
                raise HttpError(405, "POST only")
            if not route.endswith("/asr"):
                raise HttpError(404, "no such path")
            try:
                rate = int(headers.get("x-sample-rate", "16000"))
                length = int(headers.get("content-length", ""))
            except ValueError:
                raise HttpError(400, "bad rate or length") from None
            if rate not in LOCAL_RATES:
                raise HttpError(400, "bad rate")
            if not 0 <= length <= MAX_SECONDS * rate * 2 or length % 2:
                raise HttpError(400, "bad length")
            pcm = await asyncio.wait_for(reader.readexactly(length), READ_TIMEOUT)
            t1 = time.monotonic()
            async with asr_lock:
                text = await recognizer.transcribe(pcm, rate) if pcm else ""
            t2 = time.monotonic()
            await respond(writer, 200, json.dumps({"text": text}, ensure_ascii=False).encode(),
                          "application/json")
            log.info("local ok: %.2fs of %d Hz audio, upload %.0f ms, asr %.0f ms, %d chars",
                     len(pcm) / 2 / rate, rate, (t1 - t0) * 1000, (t2 - t1) * 1000, len(text))
        except HttpError as e:
            await respond(writer, e.status, (e.reason + "\n").encode(), "text/plain")
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError, ValueError) as e:
            log.warning("local dropped: %s", type(e).__name__)
        except Exception:
            log.exception("local failed")
            try:
                await respond(writer, 500, b"error\n", "text/plain")
            except Exception:
                pass
        finally:
            try:
                writer.close()
            except Exception:
                pass
    return handle


async def start_local(handler, path, group=""):
    """Listens on the Unix socket `path`, mode 0660, in `group` if given.

    A socket left by an earlier run is replaced; any other file there is not
    touched (refuses to start instead)."""
    try:
        if not stat.S_ISSOCK(os.lstat(path).st_mode):
            raise SystemExit(f"{path}: exists and is not a socket")
        os.unlink(path)
    except FileNotFoundError:
        pass
    old = os.umask(0o117)               # created 0660, never wider for a moment
    try:
        server = await asyncio.start_unix_server(handler, path)
    finally:
        os.umask(old)
    if group:
        os.chown(path, -1, grp.getgrnam(group).gr_gid)
    os.chmod(path, 0o660)
    return server


def load_key(path):
    with open(path) as f:
        key = bytes.fromhex(f.read().strip())
    if len(key) != 32:
        raise SystemExit(f"{path}: need 32 bytes of hex")
    return key


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    key = load_key(os.environ.get("MUSE_ASR_KEY_FILE", "/etc/muse-asr/key"))
    recognizer = Recognizer(os.environ.get("MUSE_ASR_MODEL", "/opt/muse-asr/model"))
    tts_dir = os.environ.get("MUSE_TTS_DIR", "")
    speaker = Speaker(tts_dir) if tts_dir else None
    host, port = os.environ.get("MUSE_ASR_LISTEN", "127.0.0.1:8102").rsplit(":", 1)
    asr_lock = asyncio.Lock()
    server = await asyncio.start_server(make_handler(key, recognizer, speaker, asr_lock=asr_lock),
                                        host, int(port))
    log.info("listening on %s:%s", host, port)
    servers = [server]
    path = os.environ.get("MUSE_ASR_SOCKET", "")
    if path:
        servers.append(await start_local(make_local_handler(recognizer, asr_lock), path,
                                         os.environ.get("MUSE_ASR_SOCKET_GROUP", "")))
        log.info("listening on %s", path)
    await asyncio.gather(*(s.serve_forever() for s in servers))


if __name__ == "__main__":
    asyncio.run(main())
