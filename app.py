import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from PIL import Image
import pytesseract
import json

st.set_page_config(page_title="LinguaFlow API", page_icon="🌐", layout="wide")

# ── Model map ──────────────────────────────────────────────────────────────
MODEL_MAP = {
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("gu", "en"): "Helsinki-NLP/opus-mt-gu-en",
    ("en", "gu"): "Helsinki-NLP/opus-mt-en-gu",
}

# ── Load MarianMT directly — no pipeline(), works on all transformers versions
@st.cache_resource(show_spinner="Loading model…")
def load_model(src: str, tgt: str):
    name = MODEL_MAP.get((src, tgt))
    if not name:
        return None, None
    tok = MarianTokenizer.from_pretrained(name)
    mdl = MarianMTModel.from_pretrained(name)
    return tok, mdl

def translate(text: str, src: str, tgt: str) -> str:
    if not text.strip() or src == tgt:
        return text

    tok, mdl = load_model(src, tgt)
    if tok and mdl:
        batch = tok([text], return_tensors="pt",
                    padding=True, truncation=True, max_length=512)
        out = mdl.generate(**batch)
        return tok.decode(out[0], skip_special_tokens=True)

    # Bridge via English for unsupported pairs (fr↔hi, fr↔gu, etc.)
    tok1, mdl1 = load_model(src, "en")
    tok2, mdl2 = load_model("en", tgt)
    if tok1 and mdl1 and tok2 and mdl2:
        b1  = tok1([text], return_tensors="pt", padding=True,
                   truncation=True, max_length=512)
        en  = tok1.decode(mdl1.generate(**b1)[0], skip_special_tokens=True)
        b2  = tok2([en],   return_tensors="pt", padding=True,
                   truncation=True, max_length=512)
        return tok2.decode(mdl2.generate(**b2)[0], skip_special_tokens=True)

    return f"[No model for {src}→{tgt}]"

# ══════════════════════════════════════════════════════════════════════════
# API MODE — called by Vercel frontend
# GET /?api=1&text=bonjour&src=fr&tgt=en
# ══════════════════════════════════════════════════════════════════════════
params = st.query_params

if params.get("api") == "1":
    text = params.get("text", "")
    src  = params.get("src", "fr")
    tgt  = params.get("tgt", "en")
    if text:
        try:
            result = translate(text, src, tgt)
            st.json({"translation": result, "src": src, "tgt": tgt})
        except Exception as e:
            st.json({"error": str(e)})
    else:
        st.json({"error": "No text provided"})
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# NORMAL STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════
st.markdown("## 🌐 LinguaFlow — Neural Translation Backend")
st.caption("MarianMT · Helsinki-NLP · transformers 5.x · Python 3.14 compatible")
st.divider()

LANGS = {
    "French 🇫🇷":   "fr",
    "English 🇬🇧":  "en",
    "Hindi 🇮🇳":    "hi",
    "Gujarati 🇮🇳": "gu",
}

tab1, tab2, tab3 = st.tabs(["📝 Text", "📷 Image OCR", "🔌 API Docs"])

# ── Tab 1 : Text ────────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        src_l = st.selectbox("From", list(LANGS.keys()), index=0)
        src   = LANGS[src_l]
        user_text = st.text_area(
            "Input",
            placeholder="e.g. je me sens affreusement mal",
            height=180, label_visibility="collapsed"
        )
    with c2:
        tgt_l = st.selectbox("To", list(LANGS.keys()), index=1)
        tgt   = LANGS[tgt_l]
        out_box = st.empty()

    if st.button("Translate ⚡", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some text first.")
        elif src == tgt:
            st.warning("Source and target languages are the same.")
        else:
            with st.spinner("Translating…"):
                try:
                    res = translate(user_text.strip(), src, tgt)
                    out_box.text_area("Result", value=res, height=180,
                                      label_visibility="collapsed")
                    st.success("✓ Done!")
                except Exception as e:
                    st.error(f"Error: {e}")

            with st.expander("↩ Back-translation check"):
                with st.spinner("Back-translating…"):
                    try:
                        back = translate(res, tgt, src)
                        st.info(f"**Back-translated:** {back}")
                    except Exception as e:
                        st.error(str(e))

# ── Tab 2 : Image OCR ───────────────────────────────────────────────────────
with tab2:
    st.caption("Upload image containing text — menus, signs, documents")
    uploaded = st.file_uploader("Upload", type=["jpg","jpeg","png","webp"])
    oc1, oc2 = st.columns(2)
    with oc1:
        ocr_src = st.selectbox("Language in image", list(LANGS.keys()), key="os")
    with oc2:
        ocr_tgt = st.selectbox("Translate to", list(LANGS.keys()), index=1, key="ot")

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_column_width=True)
        if st.button("🔍 Scan & Translate", type="primary"):
            with st.spinner("Running OCR…"):
                try:
                    raw = pytesseract.image_to_string(img).strip()
                except Exception as e:
                    raw = ""
                    st.error(f"Tesseract error: {e}")
            if raw:
                st.code(raw, language=None)
                with st.spinner("Translating…"):
                    try:
                        out = translate(raw, LANGS[ocr_src], LANGS[ocr_tgt])
                        st.success(out)
                    except Exception as e:
                        st.error(str(e))
            else:
                st.error("No text detected — try a clearer image.")

# ── Tab 3 : API Docs ────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
### 🔌 API Usage

```
GET https://YOUR-APP.streamlit.app/?api=1&text=bonjour&src=fr&tgt=en
```

| Param | Values | Example |
|-------|--------|---------|
| `api` | `1` | enables API mode |
| `text` | any string | `bonjour` |
| `src` | `fr` `en` `hi` `gu` | `fr` |
| `tgt` | `fr` `en` `hi` `gu` | `en` |

**Response:**
```json
{ "translation": "Hello", "src": "fr", "tgt": "en" }
```

**Supported pairs:**

| Pair | Method |
|------|--------|
| French ↔ English | Direct |
| Hindi ↔ English | Direct |
| Gujarati ↔ English | Direct |
| French ↔ Hindi/Gujarati | Bridged via English |
    """)