"""Nexmon CSI payload parsing and reconstruction of complex CSI (BCM4366c0, 80 MHz, 256 subcarriers).

UDP payload layout (public Nexmon CSI documentation), little endian:
    magic(2)=0x1111 | rssi(1, int8) | frame-control(1) | source MAC(6) | sequence(2) |
    core/spatial-stream(2) | chanspec(2) | chip version(2)   -> 18 header bytes
    followed by nfft 32-bit words, one per subcarrier.

For the BCM4366c0 each 32-bit word packs one complex subcarrier value as a block floating-point
number (bit 0 = LSB):
    bits  0.. 5 : 6-bit two's-complement exponent
    bits  6..16 : 11-bit magnitude of the imaginary part,  bit 17 : its sign
    bits 18..28 : 11-bit magnitude of the real part,       bit 29 : its sign
After the per-subcarrier exponents are applied, the whole packet is rescaled ("auto-scale") so that
the largest mantissa lands on bit `nbits`; this removes the per-packet receiver gain.
Everything here is an independent NumPy implementation written from that public description.
"""
from __future__ import annotations

import struct

import numpy as np

_HDR = struct.Struct("<HbB6sHHHH")  # magic, rssi, fctl, mac, seq, css, chanspec, chipver = 18 bytes


def unpack_packed_float(words, nman: int = 12, nexp: int = 6, nbits: int = 10) -> np.ndarray:
    """Vectorised unpack of Broadcom packed-float CSI words.

    words: integer array (n_packets, nfft) (or (nfft,)); returns complex64 array of the same shape.
    """
    w = np.asarray(words)
    if w.ndim == 1:
        w = w[None, :]
    H = w.astype(np.uint32).astype(np.int64)

    e_half = 1 << (nexp - 1)
    e_mask = (1 << nexp) - 1
    m_mask = (1 << (nman - 1)) - 1

    e = H & e_mask
    e = np.where(e >= e_half, e - (e_half << 1), e)               # signed exponent
    re_m = (H >> (nexp + nman)) & m_mask                           # real mantissa magnitude
    im_m = (H >> nexp) & m_mask                                    # imaginary mantissa magnitude
    re_neg = ((H >> (nexp + 2 * nman - 1)) & 1).astype(bool)
    im_neg = ((H >> (nexp + nman - 1)) & 1).astype(bool)

    lead = re_m | im_m
    nonzero = lead > 0
    msb = np.frexp(lead.astype(np.float64))[1].astype(np.int64) - 1   # index of leading 1 bit
    top = np.where(nonzero, e + msb, -e_half)
    maxbit = np.maximum(top.max(axis=1, keepdims=True), -e_half)      # per-packet reference exponent
    shift = (e + (nbits - maxbit)).astype(np.int32)                   # per-subcarrier binary shift
    flush = shift < -nman                                             # too small -> flushed to zero

    def _scale(mant: np.ndarray) -> np.ndarray:
        val = np.floor(np.ldexp(mant.astype(np.float64), shift))
        val[flush] = 0.0
        return val

    re = np.where(re_neg, -_scale(re_m), _scale(re_m))
    im = np.where(im_neg, -_scale(im_m), _scale(im_m))
    return (re + 1j * im).astype(np.complex64)


def parse_payloads(records, nfft: int = 256, header_bytes: int = 18, magic: int = 0x1111) -> dict:
    """Validate and unpack the UDP payloads of one capture, preserving capture order.

    records: [(timestamp, payload_bytes), ...]
    Returns dict with words (n, nfft) uint32 and per-packet seq / core-spatial-stream / rssi /
    chanspec / timestamp arrays, plus 'rejected' counters for malformed packets.
    """
    expected_len = header_bytes + 4 * nfft
    words, seq, css, rssi, chanspec, ts = [], [], [], [], [], []
    rejected = {"bad_length": 0, "bad_magic": 0}
    for t, payload in records:
        if len(payload) != expected_len:
            rejected["bad_length"] += 1
            continue
        m, rs, _fctl, _mac, sq, cs, chs, _chip = _HDR.unpack_from(payload, 0)
        if m != magic:
            rejected["bad_magic"] += 1
            continue
        words.append(np.frombuffer(payload, dtype="<u4", count=nfft, offset=header_bytes))
        seq.append(sq)
        css.append(cs)
        rssi.append(rs)
        chanspec.append(chs)
        ts.append(t)
    if words:
        words_arr = np.stack(words).astype(np.uint32)
    else:
        words_arr = np.zeros((0, nfft), dtype=np.uint32)
    return {
        "words": words_arr,
        "seq": np.asarray(seq, dtype=np.int64),
        "css": np.asarray(css, dtype=np.int64),
        "rssi": np.asarray(rssi, dtype=np.int64),
        "chanspec": np.asarray(chanspec, dtype=np.int64),
        "timestamp": np.asarray(ts, dtype=np.float64),
        "rejected": rejected,
    }
