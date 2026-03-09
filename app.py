import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from PIL import Image
import pytesseract
import json

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LinguaFlow API",
    page_icon="🌐",
    layout="wide"
)

# ── Model map ──────────────────────────────────────────────────────────────
MODEL_MAP = {
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("gu", "en"): "Helsinki-NLP/opus-mt-gu-en",
    ("en", "gu"): "Helsinki-NLP/opus-mt-en-gu",
}

# ── Load model directly (no pipeline — avoids task name issues) ────────────
@st.cache_resource(show_spinner="Loading translation model…")
def load_model(src: str, tgt: str):
    model_name = MODEL_MAP.get((src, tgt))
    if not model_name:
        return None, None
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model     = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# ── Translate function ─────────────────────────────────────────────────────
def translate(text: str, src: str, tgt: str) -> str:
    if not text.strip():
        return ""
    if src == tgt:
        return text

    tokenizer, model = load_model(src, tgt)
    if tokenizer and model:
        inputs  = tokenizer(text, return_tensors="pt",
                            padding=True, truncation=True, max_length=512)
        outputs = model.generate(**inputs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Bridge via English
    tok_en, mod_en = load_model(src, "en")
    tok_tg, mod_tg = load_model("en", tgt)
    if tok_en and mod_en and tok_tg and mod_tg:
        inp1 = tok_en(text, return_tensors="pt",
                      padding=True, truncation=True, max_length=512)
        out1 = mod_en.generate(**inp1)
        en   = tok_en.decode(out1[0], skip_special_tokens=True)
        inp2 = tok_tg(en, return_tensors="pt",
                      padding=True, truncation=True, max_length=512)
        out2 = mod_tg.generate(**inp2)
        return tok_tg.decode(out2[0], skip_special_tokens=True)

    return f"[No model available for {src}→{tgt}]"

# ══════════════════════════════════════════════════════════════════════════
# API MODE  ?api=1&text=bonjour&src=fr&tgt=en
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
# NORMAL UI
# ══════════════════════════════════════════════════════════════════════════
st.markdown("## 🌐 LinguaFlow — Neural Translation Backend")
st.caption("MarianMT · Helsinki-NLP · PyTorch · Free · No API key")
st.divider()

tab1, tab2, tab3 = st.tabs(["📝 Text", "📷 Image OCR", "🔌 API Docs"])

LANGS = {
    "French 🇫🇷":   "fr",
    "English 🇬🇧":  "en",
    "Hindi 🇮🇳":    "hi",
    "Gujarati 🇮🇳": "gu",
}

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        src_label = st.selectbox("From", list(LANGS.keys()), index=0)
        src = LANGS[src_label]
        user_text = st.text_area("Input",
            placeholder="e.g. je me sens affreusement mal",
            height=180, label_visibility="collapsed")
    with c2:
        tgt_label = st.selectbox("To", list(LANGS.keys()), index=1)
        tgt = LANGS[tgt_label]
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
                    st.error(f"Translation error: {e}")
            with st.expander("↩ Back-translation check"):
                with st.spinner("Back-translating…"):
                    try:
                        back = translate(res, tgt, src)
                        st.info(f"**Back-translated:** {back}")
                    except Exception as e:
                        st.error(str(e))

with tab2:
    st.caption("Upload an image containing text — menus, signs, documents")
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png","webp"])
    oc1, oc2 = st.columns(2)
    with oc1:
        ocr_src_l = st.selectbox("Language in image", list(LANGS.keys()), key="os")
    with oc2:
        ocr_tgt_l = st.selectbox("Translate to", list(LANGS.keys()), index=1, key="ot")
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
                        out = translate(raw, LANGS[ocr_src_l], LANGS[ocr_tgt_l])
                        st.success(out)
                    except Exception as e:
                        st.error(str(e))
            else:
                st.error("No text detected. Try a clearer image.")

with tab3:
    st.markdown("""
### 🔌 API Usage

```
GET https://YOUR-APP.streamlit.app/?api=1&text=bonjour&src=fr&tgt=en
```

#### Parameters
| Param | Values | Example |
|-------|--------|---------|
| `api` | `1` | enables API mode |
| `text` | any string | `bonjour` |
| `src` | `fr` `en` `hi` `gu` | `fr` |
| `tgt` | `fr` `en` `hi` `gu` | `en` |

#### Response
```json
{ "translation": "Hello", "src": "fr", "tgt": "en" }
```
    """)