#!/usr/bin/env python3
"""Talks to muse_asr the way the gadget does.

  client.py KEY_FILE URL file.wav       transcribe (URL e.g. http://203.0.113.10/TV/asr)
  client.py KEY_FILE URL --say TEXT out.wav   speak (URL .../TV/tts), save 16 kHz WAV

Requests carry the URL's host as Host unless --host-header. A WAV to transcribe must be
16-bit mono; it goes up in 100 ms frames, paced like a live mic unless --fast.
"""

import argparse
import http.client
import json
import os
import struct
import sys
import time
import urllib.parse
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as P  # noqa: E402


def body(key, hello, pcm, rate, fast):
    yield hello
    step = rate // 10 * 2
    counter = 0
    for off in range(0, len(pcm), step):
        yield P.seal(key, hello, counter, P.T_AUDIO, pcm[off:off + step])
        counter += 1
        if not fast:
            time.sleep(0.1)
    yield P.seal(key, hello, counter, P.T_END)


def speak(key, a):
    hello = P.make_hello(os.urandom(8), 16000)
    body = hello + P.seal(key, hello, 0, P.T_TEXT, bytes([a.voice]) + a.say.encode())
    u = urllib.parse.urlparse(a.url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=60)
    conn._http_vsn, conn._http_vsn_str = 10, "HTTP/1.0"
    t0 = time.monotonic()
    conn.request("POST", u.path, body=body, headers={"Host": a.host_header or u.netloc})
    r = conn.getresponse()
    if r.status != 200:
        sys.exit(f"HTTP {r.status}")
    rd, dec, pcm, counter, first = P.FrameReader(), P.Adpcm(), [], P.SERVER_COUNTER, None
    play_end, stalled = None, 0.0     # a player that starts each piece as soon as it's here
    while True:
        data = r.read1(4096)
        if not data:
            break
        rd.feed(data)
        for ct in rd.frames():
            kind, payload = P.open_frame(key, hello, counter, ct)
            counter += 1
            if kind == P.T_SEGMENT:
                n, end = struct.unpack("<IH", payload)
                now = time.monotonic() - t0
                if play_end is not None and now > play_end:
                    stalled += now - play_end
                play_end = max(now, play_end or now) + n / 16000
                print(f"{now:5.2f}s segment: {n / 16000:.2f}s, text to byte {end}", file=sys.stderr)
            elif kind == P.T_AUDIO:
                first = first or time.monotonic()
                pcm += dec.decode(payload)
    with wave.open(a.wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<%dh" % len(pcm), *pcm))
    print(f"{len(pcm) / 16000:.2f}s of speech, first audio after {(first - t0) * 1000:.0f} ms, "
          f"all after {(time.monotonic() - t0) * 1000:.0f} ms, gaps between pieces {stalled:.2f}s",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key_file")
    ap.add_argument("url")
    ap.add_argument("wav")
    ap.add_argument("--say", help="text to speak; wav is then the output")
    ap.add_argument("--voice", type=int, default=0, help="voice number (server.VOICES)")
    ap.add_argument("--host-header", help="Host header (default: the URL's host)")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    key = bytes.fromhex(open(a.key_file).read().strip())
    if a.say:
        return speak(key, a)
    with wave.open(a.wav) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, "16-bit mono WAV"
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
    hello = P.make_hello(os.urandom(8), rate)
    u = urllib.parse.urlparse(a.url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=60)
    t0 = time.monotonic()
    conn.request("POST", u.path, body=body(key, hello, pcm, rate, a.fast),
                 headers={"Host": a.host_header or u.netloc, "Content-Type": "application/octet-stream"},
                 encode_chunked=True)
    r = conn.getresponse()
    data = r.read()
    t1 = time.monotonic()
    if r.status != 200:
        sys.exit(f"HTTP {r.status}: {data[:200]!r}")
    n = struct.unpack("<H", data[:2])[0]
    kind, payload = P.open_frame(key, hello, P.SERVER_COUNTER, data[2:2 + n])
    assert kind == P.T_RESULT
    print(json.loads(payload)["text"])
    print(f"({len(pcm) / 2 / rate:.2f}s audio, {(t1 - t0) * 1000:.0f} ms total)", file=sys.stderr)


if __name__ == "__main__":
    main()
