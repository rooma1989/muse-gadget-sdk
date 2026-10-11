"""Host tests for muse_asr: framing, the server's request handling, rejection.

  python3 -m unittest tools/muse_asr/test_muse_asr.py
"""

import asyncio
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as P  # noqa: E402
import server  # noqa: E402

KEY = bytes(range(32))


class FakeRecognizer:
    def __init__(self):
        self.calls = []

    async def transcribe(self, pcm, rate):
        self.calls.append((len(pcm), rate))
        return "你好，世界"


def chunked(parts):
    out = b""
    for p in parts:
        out += b"%x\r\n" % len(p) + p + b"\r\n"
    return out + b"0\r\n\r\n"


def request(body, chunked_body=True):
    head = b"POST /asr HTTP/1.1\r\nHost: x\r\n"
    head += b"Transfer-Encoding: chunked\r\n\r\n" if chunked_body else b"Content-Length: %d\r\n\r\n" % len(body)
    return head + body


async def roundtrip(raw, recognizer):
    srv = await asyncio.start_server(server.make_handler(KEY, recognizer), "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    async with srv:
        r, w = await asyncio.open_connection("127.0.0.1", port)
        w.write(raw)
        await w.drain()
        resp = await r.read()
        w.close()
    head, _, body = resp.partition(b"\r\n\r\n")
    return int(head.split(b" ")[1]), body


def speech_frames(hello, pcm, key=KEY, end=True):
    frames = [hello]
    counter = 0
    for off in range(0, len(pcm), 3200):
        frames.append(P.seal(key, hello, counter, P.T_AUDIO, pcm[off:off + 3200]))
        counter += 1
    if end:
        frames.append(P.seal(key, hello, counter, P.T_END))
    return frames


class ProtocolTests(unittest.TestCase):
    def test_seal_open_roundtrip(self):
        hello = P.make_hello(b"\x01" * 8, 16000)
        frame = P.seal(KEY, hello, 7, P.T_AUDIO, b"abc")
        rd = P.FrameReader()
        rd.feed(frame[:5])
        self.assertEqual(list(rd.frames()), [])
        rd.feed(frame[5:])
        (ct,) = list(rd.frames())
        self.assertEqual(P.open_frame(KEY, hello, 7, ct), (P.T_AUDIO, b"abc"))

    def test_wrong_counter_or_key_fails(self):
        hello = P.make_hello(b"\x02" * 8, 16000)
        ct = P.seal(KEY, hello, 0, P.T_END)[2:]
        with self.assertRaises(P.ProtocolError):
            P.open_frame(KEY, hello, 1, ct)
        with self.assertRaises(P.ProtocolError):
            P.open_frame(bytes(32), hello, 0, ct)

    def test_hello_is_authenticated(self):
        hello = P.make_hello(b"\x03" * 8, 16000)
        ct = P.seal(KEY, hello, 0, P.T_END)[2:]
        forged = P.make_hello(b"\x03" * 8, 8000)
        with self.assertRaises(P.ProtocolError):
            P.open_frame(KEY, forged, 0, ct)

    def test_bad_hello(self):
        with self.assertRaises(P.ProtocolError):
            P.parse_hello(b"XXXX" + bytes(12))
        with self.assertRaises(P.ProtocolError):
            P.parse_hello(P.MAGIC + bytes(8) + struct.pack("<I", 44100))


class ServerTests(unittest.TestCase):
    def test_transcribes_chunked_upload(self):
        rec = FakeRecognizer()
        hello = P.make_hello(os.urandom(8), 16000)
        pcm = bytes(16000 * 2)  # one second
        # Split the stream at awkward places, as TCP would.
        stream = b"".join(speech_frames(hello, pcm))
        parts = [stream[i:i + 777] for i in range(0, len(stream), 777)]
        status, body = asyncio.run(roundtrip(request(chunked(parts)), rec))
        self.assertEqual(status, 200)
        n = struct.unpack("<H", body[:2])[0]
        kind, payload = P.open_frame(KEY, hello, P.SERVER_COUNTER, body[2:2 + n])
        self.assertEqual(kind, P.T_RESULT)
        self.assertEqual(payload.decode(), '{"text": "你好，世界"}')
        self.assertEqual(rec.calls, [(len(pcm), 16000)])

    def test_content_length_upload(self):
        rec = FakeRecognizer()
        hello = P.make_hello(os.urandom(8), 8000)
        status, _ = asyncio.run(roundtrip(request(b"".join(speech_frames(hello, bytes(1600))), False), rec))
        self.assertEqual(status, 200)
        self.assertEqual(rec.calls, [(1600, 8000)])

    def test_wrong_key_is_rejected_without_transcribing(self):
        rec = FakeRecognizer()
        hello = P.make_hello(os.urandom(8), 16000)
        frames = speech_frames(hello, bytes(3200), key=bytes(32))
        status, _ = asyncio.run(roundtrip(request(chunked(frames)), rec))
        self.assertEqual(status, 403)
        self.assertEqual(rec.calls, [])

    def test_missing_end_is_rejected(self):
        rec = FakeRecognizer()
        hello = P.make_hello(os.urandom(8), 16000)
        status, _ = asyncio.run(roundtrip(request(chunked(speech_frames(hello, bytes(3200), end=False))), rec))
        self.assertEqual(status, 403)
        self.assertEqual(rec.calls, [])

    def test_get_is_refused(self):
        status, _ = asyncio.run(roundtrip(b"GET /asr HTTP/1.1\r\nHost: x\r\n\r\n", FakeRecognizer()))
        self.assertEqual(status, 405)


if __name__ == "__main__":
    unittest.main()
