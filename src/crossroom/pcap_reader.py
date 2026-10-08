"""Minimal reader for classic libpcap files that returns the UDP payloads in capture order.

The Nexmon CSI extractor sends each CSI observation as one UDP datagram (port 5500 by default),
so a PCAP is turned into an ordered list of (timestamp, udp_payload). pcapng is not supported
(the supplied captures are classic pcap, Ethernet link type).
"""
from __future__ import annotations

import struct
from pathlib import Path

# magic bytes -> (endianness, timestamp-fraction scale)
_PCAP_MAGIC = {
    b"\xd4\xc3\xb2\xa1": ("<", 1e-6),
    b"\xa1\xb2\xc3\xd4": (">", 1e-6),
    b"\x4d\x3c\xb2\xa1": ("<", 1e-9),
    b"\xa1\xb2\x3c\x4d": (">", 1e-9),
}


def _ip_offset(linktype: int, frame: bytes):
    """Offset of the IPv4 header inside a link-layer frame, or None if the frame is not IPv4."""
    if linktype == 1:  # Ethernet (optionally 802.1Q / 802.1ad tagged)
        if len(frame) < 14:
            return None
        ethertype = (frame[12] << 8) | frame[13]
        off = 14
        while ethertype in (0x8100, 0x88A8) and len(frame) >= off + 4:
            ethertype = (frame[off + 2] << 8) | frame[off + 3]
            off += 4
        return off if ethertype == 0x0800 else None
    if linktype == 113:  # Linux cooked capture
        if len(frame) < 16:
            return None
        return 16 if ((frame[14] << 8) | frame[15]) == 0x0800 else None
    if linktype in (12, 14, 101):  # raw IP
        return 0
    if linktype == 0:  # BSD loopback
        return 4
    return None


def read_pcap_udp(path, udp_port=None):
    """Return ([(timestamp_seconds, udp_payload_bytes), ...], stats) for one capture file."""
    data = Path(path).read_bytes()
    stats = {"n_records": 0, "n_non_ip": 0, "n_non_udp": 0, "n_other_port": 0, "n_truncated": 0}
    if len(data) < 24:
        raise ValueError(f"{path}: file too short to be a pcap")
    magic = data[:4]
    if magic not in _PCAP_MAGIC:
        raise ValueError(f"{path}: not a classic libpcap file (magic={magic.hex()}); pcapng is not supported")
    endian, ts_scale = _PCAP_MAGIC[magic]
    linktype = struct.unpack(endian + "I", data[20:24])[0]
    rec = struct.Struct(endian + "IIII")
    out = []
    off, n = 24, len(data)
    while off + 16 <= n:
        ts_s, ts_f, incl, _orig = rec.unpack_from(data, off)
        start, end = off + 16, off + 16 + incl
        if end > n:
            stats["n_truncated"] += 1
            break
        off = end
        stats["n_records"] += 1
        frame = data[start:end]
        l3 = _ip_offset(linktype, frame)
        if l3 is None or len(frame) < l3 + 20:
            stats["n_non_ip"] += 1
            continue
        ip = frame[l3:]
        if (ip[0] >> 4) != 4:
            stats["n_non_ip"] += 1
            continue
        ihl = (ip[0] & 0x0F) * 4
        if ip[9] != 17 or len(ip) < ihl + 8:
            stats["n_non_udp"] += 1
            continue
        sport, dport, ulen = struct.unpack_from(">HHH", ip, ihl)
        if udp_port is not None and udp_port not in (sport, dport):
            stats["n_other_port"] += 1
            continue
        payload = ip[ihl + 8: ihl + max(ulen, 8)]
        out.append((ts_s + ts_f * ts_scale, payload))
    return out, stats
