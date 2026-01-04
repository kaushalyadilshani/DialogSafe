from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.config import AppConfig
from src.domain import ProfanitySpan
from src.video_tools import (
    _build_ffmpeg_censor_and_mux_cmd,
    build_bleep_filter,
    build_mute_filter,
    probe_primary_audio_codec_and_bitrate,
)


def test_build_mute_filter_generates_volume_filters():
    spans = [
        ProfanitySpan(start=1.0, end=2.0, hits=[], max_confidence=0.9),
        ProfanitySpan(start=3.5, end=3.9, hits=[], max_confidence=0.8),
    ]
    af = build_mute_filter(spans)
    assert af is not None
    # Mute mode pads the END of each span by 150ms to cover trailing audio.
    assert "volume=enable='between(t,1.000,2.150)'" in af
    assert "volume=enable='between(t,3.500,4.050)'" in af


def test_build_bleep_filter_contains_expected_components():
    spans = [
        ProfanitySpan(start=1.0, end=2.0, hits=[], max_confidence=0.9),
    ]
    fc = build_bleep_filter(spans, sample_rate=16000)
    assert fc is not None
    assert "aevalsrc=0.5*sin(2*PI*1000*t)" in fc
    assert "adelay=" in fc
    assert "amix=inputs=" in fc


def test_build_ffmpeg_cmd_preserves_other_audio_streams_and_reencodes_only_primary():
    spans = [ProfanitySpan(start=1.0, end=2.0, hits=[], max_confidence=0.9)]
    config = AppConfig(input_path=Path("in.mp4"), output_path=Path("out.mp4"), mode="mute")

    cmd = _build_ffmpeg_censor_and_mux_cmd(
        input_video=Path("in.mp4"),
        output_video=Path("out.mp4"),
        spans=spans,
        config=config,
        primary_audio_codec="ac3",
        primary_audio_bit_rate=384000,
    )

    # Must use filter_complex so we can target only 0:a:0 and keep other streams via mapping.
    assert "-filter_complex" in cmd

    # Exact mapping per spec.
    map_args = []
    for i, token in enumerate(cmd):
        if token == "-map":
            map_args.append(cmd[i + 1])
    assert map_args == ["0:v:0", "[aout]", "0:a", "-0:a:0"]

    # Copy video + all (other) audio streams, but re-encode only the first output audio stream.
    assert ["-c:v", "copy"] == cmd[cmd.index("-c:v") : cmd.index("-c:v") + 2]
    assert ["-c:a", "copy"] == cmd[cmd.index("-c:a") : cmd.index("-c:a") + 2]
    assert ["-c:a:0", "ac3"] == cmd[cmd.index("-c:a:0") : cmd.index("-c:a:0") + 2]
    assert ["-b:a:0", "384000"] == cmd[cmd.index("-b:a:0") : cmd.index("-b:a:0") + 2]


def test_probe_primary_audio_codec_and_bitrate_parses_ffprobe_json(monkeypatch):
    def fake_run(cmd, stdout, stderr, text):  # noqa: ANN001
        assert cmd[:4] == ["ffprobe", "-v", "error", "-select_streams"]
        return SimpleNamespace(
            returncode=0,
            stdout='{"streams":[{"codec_name":"eac3","bit_rate":"768000"}]}',
            stderr="",
        )

    monkeypatch.setattr("src.video_tools.subprocess.run", fake_run)
    codec, bitrate = probe_primary_audio_codec_and_bitrate(Path("in.mp4"))
    assert codec == "eac3"
    assert bitrate == 768000
