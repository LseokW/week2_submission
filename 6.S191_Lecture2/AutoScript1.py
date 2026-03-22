import streamlit as st
import json
import os
import re
import tempfile
import yt_dlp

st.set_page_config(page_title="YT Script Extractor", page_icon="📄", layout="wide")

st.title("📄 YT Script Extractor")
st.caption("유튜브 영상 스크립트 추출기 — API 키 불필요")

# ── 사이드바 설정 ─────────────────────────────────────────────
st.sidebar.header("설정")
show_timestamps = st.sidebar.checkbox("타임스탬프 포함", value=True)
merge_lines = st.sidebar.checkbox("줄 합치기 (읽기 편하게)", value=True)

# ── 자막 추출 함수 ────────────────────────────────────────────
def fetch_subtitles(url: str) -> list[dict] | None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "sub")

        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
            "subtitlesformat": "json3",
            "outtmpl": out_path,
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # json3 파일 찾기
        for fname in [f"{out_path}.en.json3", f"{out_path}.en-orig.json3"]:
            if os.path.exists(fname):
                with open(fname, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return parse_json3(data)

        # fallback: vtt
        ydl_opts2 = {**ydl_opts, "subtitlesformat": "vtt"}
        with yt_dlp.YoutubeDL(ydl_opts2) as ydl:
            ydl.download([url])

        for fname in [f"{out_path}.en.vtt", f"{out_path}.en-orig.vtt"]:
            if os.path.exists(fname):
                with open(fname, "r", encoding="utf-8") as f:
                    content = f.read()
                return parse_vtt(content)

    return None


def parse_json3(data: dict) -> list[dict]:
    segments = []
    for event in data.get("events", []):
        if "segs" not in event:
            continue
        text = "".join(s.get("utf8", "") for s in event["segs"]).strip()
        if not text or text == "\n":
            continue
        start_ms = event.get("tStartMs", 0)
        segments.append({
            "start": ms_to_time(start_ms),
            "text": text.replace("\n", " ")
        })
    return segments


def parse_vtt(content: str) -> list[dict]:
    segments = []
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" in line:
            start = line.split("-->")[0].strip().split(".")[0]
            texts = []
            i += 1
            while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
                t = re.sub(r"<[^>]+>", "", lines[i]).strip()
                if t:
                    texts.append(t)
                i += 1
            if texts:
                segments.append({"start": start, "text": " ".join(texts)})
        else:
            i += 1
    return segments


def ms_to_time(ms: int) -> str:
    s = ms // 1000
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def segments_to_text(segments: list[dict], with_ts: bool, merged: bool) -> str:
    if merged:
        text = " ".join(seg["text"] for seg in segments)
        text = re.sub(r'([.!?])\s+', r'\1\n', text)
        if with_ts:
            lines = []
            for i, seg in enumerate(segments):
                if i % 10 == 0:
                    lines.append(f"\n[{seg['start']}]")
                lines.append(seg["text"])
            return " ".join(lines)
        return text
    else:
        if with_ts:
            return "\n".join(f"[{seg['start']}]  {seg['text']}" for seg in segments)
        else:
            return "\n".join(seg["text"] for seg in segments)


# ── 메인 UI ───────────────────────────────────────────────────
url = st.text_input(
    "유튜브 URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

if st.button("📥 스크립트 추출", type="primary", use_container_width=True):
    if not url.strip():
        st.error("URL을 입력해주세요.")
        st.stop()

    with st.spinner("자막 다운로드 중..."):
        segments = fetch_subtitles(url.strip())

    if not segments:
        st.error("자막을 찾을 수 없어요. 자동생성 자막도 없는 영상이거나 URL을 확인해보세요.")
        st.stop()

    st.success(f"✅ {len(segments)}개 세그먼트 추출 완료")

    script_text = segments_to_text(segments, show_timestamps, merge_lines)

    st.text_area("스크립트", script_text, height=500)

    st.download_button(
        "📥 txt로 저장",
        data=script_text,
        file_name="script.txt",
        mime="text/plain",
        use_container_width=True
    )

    word_count = sum(len(seg["text"].split()) for seg in segments)
    duration = segments[-1]["start"] if segments else "?"
    st.caption(f"단어 수: {word_count:,}개 · 영상 길이: ~{duration}")
