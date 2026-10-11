#!/usr/bin/env python3
"""Speech-to-text for a Muse gadget that can't upload voice notes to Muse.

The AI Passport (ESP32-C3, no PSRAM) reaches Muse through a relay with a long
round trip, and can neither upload a voice note fast enough nor receive the
16 KB frame Muse echoes it back with. Instead it streams the mic here (a nearby
server), gets the transcript back, and sends Muse plain text.

Listens on plain HTTP behind nginx (POST, chunked body; see protocol.py).
SenseVoice-Small through sherpa-onnx does the recognition, on CPU.

  MUSE_ASR_KEY_FILE  hex AES-256 key shared with the gadget (/etc/muse-asr/key)
  MUSE_ASR_MODEL     directory with model.int8.onnx and tokens.txt
  MUSE_ASR_LISTEN    host:port (127.0.0.1:8102)

Transcripts are never logged, only their lengths.
"""

import asyncio
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as P  # noqa: E402

MAX_SECONDS = 20
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
        self.lock = asyncio.Lock()

    def _decode(self, pcm, rate):
        import numpy as np
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        stream = self.rec.create_stream()
        stream.accept_waveform(rate, samples)
        self.rec.decode_stream(stream)
        return stream.result.text.strip()

    async def transcribe(self, pcm, rate):
        async with self.lock:
            return await asyncio.get_running_loop().run_in_executor(None, self._decode, pcm, rate)


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


def make_handler(key, recognizer):
    async def handle(reader, writer):
        t0 = time.monotonic()
        try:
            method, path, headers = await asyncio.wait_for(read_headers(reader), 10)
            if method == "GET" and path.rstrip("/").endswith("/health"):
                await respond(writer, 200, b"ok\n", "text/plain")
                return
            if method != "POST":
                raise HttpError(405, "POST only")
            hello, rate, pcm = await asyncio.wait_for(
                read_speech(key, body_chunks(reader, headers)), READ_TIMEOUT)
            t1 = time.monotonic()
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
    host, port = os.environ.get("MUSE_ASR_LISTEN", "127.0.0.1:8102").rsplit(":", 1)
    server = await asyncio.start_server(make_handler(key, recognizer), host, int(port))
    log.info("listening on %s:%s", host, port)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
