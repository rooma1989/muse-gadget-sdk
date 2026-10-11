"""Host tests for muse_asr: framing, the server's request handling, rejection.

  python3 -m unittest tools/muse_asr/test_muse_asr.py
"""

import asyncio
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
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


async def roundtrip(raw, recognizer, speaker=None):
    srv = await asyncio.start_server(server.make_handler(KEY, recognizer, speaker), "127.0.0.1", 0)
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


class FakeSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text, voice=0):
        import numpy as np
        self.said.append((voice, text))
        t = np.arange(1600)
        return (8000 * np.sin(t * 0.05)).astype(np.int16)


def tts_request(text, key=KEY, voice=0):
    hello = P.make_hello(os.urandom(8), 16000)
    body = hello + P.seal(key, hello, 0, P.T_TEXT, bytes([voice]) + text.encode())
    head = b"POST /tts HTTP/1.0\r\nHost: x\r\nContent-Length: %d\r\n\r\n" % len(body)
    return hello, head + body


def tts_frames(hello, body, key=KEY):
    rd = P.FrameReader()
    rd.feed(body)
    out = []
    for i, ct in enumerate(rd.frames()):
        out.append(P.open_frame(key, hello, P.SERVER_COUNTER + i, ct))
    return out


class TtsTests(unittest.TestCase):
    def test_segments_strip_markdown_and_track_offsets(self):
        text = "今天下午：\n\n- 认识了你，记下你叫**小鼠** 👋\n- 看了[天气](https://x.y)。好的!"
        segs = server.segments(text)
        self.assertEqual([s for s, _ in segs], ["今天下午：", "认识了你，", "记下你叫小鼠", "看了天气。", "好的!"])
        ends = [e for _, e in segs]
        self.assertEqual(ends[-1], len(text.encode()))
        self.assertEqual(sorted(ends), ends)
        # Each end falls right after the text it covers.
        self.assertTrue(text.encode()[:ends[3]].decode().endswith("。"))

    def test_sentences_split_at_commas_short_clauses_join(self):
        text = "好的，我来帮你查一下。北京今天晴，风小，最高二十三度。"
        segs = server.segments(text)
        # The reply's first clause goes alone; later short ones ("风小") join the next.
        self.assertEqual([s for s, _ in segs], ["好的，", "我来帮你查一下。", "北京今天晴，", "风小，最高二十三度。"])
        self.assertEqual(segs[0][1], len("好的，".encode()))
        self.assertEqual(segs[-1][1], len(text.encode()))

    def test_ascii_commas_and_colons_split_only_before_a_space(self):
        segs = server.segments("Note: see https://a.b/c:d, ok? 3,000 items.")
        self.assertEqual([s for s, _ in segs], ["Note:", "see ok?", "3,000 items."])

    def test_segments_skip_symbol_only_and_cap_length(self):
        self.assertEqual(server.segments("👋\n---\n"), [])
        long = "这是一句话。" * 200
        said = sum(len(s) for s, _ in server.segments(long))
        self.assertLessEqual(said, server.TTS_SPEECH_MAX)

    def test_resample_to_16k(self):
        import numpy as np
        pcm = server.resample(np.zeros(44100, dtype=np.float32), 44100)
        self.assertEqual(pcm.dtype, np.int16)
        self.assertAlmostEqual(len(pcm), 16000, delta=2)

    def test_tts_streams_segments_audio_and_end(self):
        spk = FakeSpeaker()
        hello, raw = tts_request("你好。今天怎么样？", voice=2)
        status, body = asyncio.run(roundtrip(raw, FakeRecognizer(), spk))
        self.assertEqual(status, 200)
        frames = tts_frames(hello, body)
        kinds = [k for k, _ in frames]
        self.assertEqual(kinds, [P.T_SEGMENT, P.T_AUDIO, P.T_SEGMENT, P.T_AUDIO, P.T_END])  # no comma: no split
        samples, end = struct.unpack("<IH", frames[0][1])
        self.assertEqual((samples, end), (1600, len("你好。".encode())))
        self.assertEqual(len(frames[1][1]), 800)
        self.assertEqual(spk.said, [(2, "你好。"), (2, "今天怎么样？")])
        # One ADPCM state runs across segments.
        dec = P.Adpcm()
        first = dec.decode(frames[1][1])
        want = [int(v) for v in FakeSpeaker().say("x")]
        # Past the first few dozen samples, while the step size ramps up.
        self.assertLess(max(abs(a - b) for a, b in zip(first[64:], want[64:])), 1200)

    def test_pace_holds_back_only_what_a_slow_host_needs(self):
        pace = server.Pace()
        pace.learn(0, 10, 2.5, 2.5)            # real time: nothing to hold back
        self.assertEqual(pace.lead(0, 40), 0.0)
        pace = server.Pace()
        pace.learn(1, 10, 2.0, 3.0)            # 1.5x slower than real time
        self.assertAlmostEqual(pace.lead(1, 4), 4 * 0.2 * 0.5)
        self.assertEqual(pace.lead(1, 400), server.MAX_LEAD)

    def test_tts_wrong_key_is_rejected(self):
        spk = FakeSpeaker()
        _, raw = tts_request("你好", key=bytes(32))
        status, _ = asyncio.run(roundtrip(raw, FakeRecognizer(), spk))
        self.assertEqual(status, 403)
        self.assertEqual(spk.said, [])


ADPCM_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../components/muse/muse_adpcm.c")
HARNESS = r"""
#include <stdio.h>
#include "muse_adpcm.h"
int main(int argc, char **argv) {
    static int16_t pcm[1 << 16]; static uint8_t enc[1 << 15];
    size_t n = fread(pcm, 2, 1 << 16, stdin) & ~(size_t)1;
    muse_adpcm_t s = {0};
    if (argv[1][0] == 'e') { muse_adpcm_encode_block(&s, pcm, n, enc); fwrite(enc, 1, n / 2, stdout); }
    else { size_t m = fread(enc, 1, sizeof enc, stdin); (void)m; }
    return 0;
}
"""


@unittest.skipUnless(shutil.which("cc"), "needs a C compiler")
class AdpcmMatchesDeviceTests(unittest.TestCase):
    def test_python_encoder_matches_muse_adpcm_c(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "h.c"), "w") as f:
            f.write(HARNESS)
        exe = os.path.join(d, "h")
        subprocess.run(["cc", "-O1", "-I", os.path.dirname(ADPCM_C), "-o", exe,
                        os.path.join(d, "h.c"), ADPCM_C], check=True)
        rnd = random.Random(7)
        samples = [int(12000 * __import__("math").sin(i * 0.031)) + rnd.randint(-3000, 3000) for i in range(8000)]
        samples += [32767, -32768] * 50 + [0] * 100
        raw = struct.pack("<%dh" % len(samples), *samples)
        c_enc = subprocess.run([exe, "e"], input=raw, capture_output=True, check=True).stdout
        self.assertEqual(P.Adpcm().encode(samples), c_enc)


if __name__ == "__main__":
    unittest.main()
