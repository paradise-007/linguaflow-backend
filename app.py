import streamlit as st
from transformers import pipeline
from PIL import Image
import pytesseract
import json
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LinguaFlow API",
    page_icon="🌐",
    layout="wide"
)

# ── CORS helper — returns JSON so the Vercel frontend can call this ─────────
# Streamlit doesn't support true REST, so we use st.query_params to act
# like a lightweight API endpoint when called with ?api=1
params = st.query_params

# ── Model loader ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading translation model…")
def load_translator(src: str, tgt: str):
    model_map = {
        ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
        ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
        ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
        ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
        ("gu", "en"): "Helsinki-NLP/opus-mt-gu-en",
        ("en", "gu"): "Helsinki-NLP/opus-mt-en-gu",
        ("fr", "hi"): None,   # bridged
        ("hi", "fr"): None,   # bridged
        ("fr", "gu"): None,   # bridged
        ("gu", "fr"): None,   # bridged
    }
    name = model_map.get((src, tgt))
    if name:
        return pipeline("translation", model=name, framework="pt")
    return None

def translate(text: str, src: str, tgt: str) -> str:
    if src == tgt:
        return text

    # Direct pair
    pipe = load_translator(src, tgt)
    if pipe:
        return pipe(text, max_length=512)[0]["translation_text"]

    # Bridge via English
    to_en   = load_translator(src, "en")
    from_en = load_translator("en", tgt)
    if to_en and from_en:
        en_text = to_en(text, max_length=512)[0]["translation_text"]
        return from_en(en_text, max_length=512)[0]["translation_text"]

    return f"[No model for {src}→{tgt}]"

# ══════════════════════════════════════════════════════════════════════════
# API MODE — called by Vercel frontend via ?api=1&text=...&src=fr&tgt=en
# ══════════════════════════════════════════════════════════════════════════
if params.get("api") == "1":
    text = params.get("text", "")
    src  = params.get("src", "fr")
    tgt  = params.get("tgt", "en")

    if text:
        result = translate(text, src, tgt)
        # Render as raw JSON in a code block Vercel can scrape
        st.markdown("```json")
        st.markdown(json.dumps({"translation": result, "src": src, "tgt": tgt}))
        st.markdown("```")
    else:
        st.markdown(json.dumps({"error": "No text provided"}))
    st.stop()   # Don't render the rest of the UI

# ══════════════════════════════════════════════════════════════════════════
# NORMAL STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════
st.markdown("## 🌐 LinguaFlow — Neural Translation Backend")
st.caption("Helsinki-NLP Opus-MT · PyTorch · Free · No API key")
st.divider()

tab1, tab2, tab3 = st.tabs(["📝 Text", "📷 Image OCR", "🔌 API Docs"])

LANGS = {
    "French 🇫🇷":   "fr",
    "English 🇬🇧":  "en",
    "Hindi 🇮🇳":    "hi",
    "Gujarati 🇮🇳": "gu",
}

# ── Tab 1 : Text ────────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        src_label = st.selectbox("From", list(LANGS.keys()), index=0)
        src = LANGS[src_label]
        user_text = st.text_area("Input", placeholder="Type here…", height=180,
                                 label_visibility="collapsed")
    with c2:
        tgt_label = st.selectbox("To", list(LANGS.keys()), index=1)
        tgt = LANGS[tgt_label]
        out_box = st.empty()

    if st.button("Translate ⚡", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Enter some text first.")
        elif src == tgt:
            st.warning("Source and target are the same.")
        else:
            with st.spinner("Translating…"):
                res = translate(user_text.strip(), src, tgt)
            out_box.text_area("Result", value=res, height=180,
                              label_visibility="collapsed")
            st.success("✓ Done!")
            with st.expander("↩ Back-translation check"):
                with st.spinner():
                    back = translate(res, tgt, src)
                st.info(back)

# ── Tab 2 : Image OCR ───────────────────────────────────────────────────────
with tab2:
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png","webp"])
    oc1, oc2 = st.columns(2)
    with oc1:
        ocr_src_l = st.selectbox("Language in image", list(LANGS.keys()), key="os")
    with oc2:
        ocr_tgt_l = st.selectbox("Translate to",      list(LANGS.keys()), index=1, key="ot")

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
                    out = translate(raw, LANGS[ocr_src_l], LANGS[ocr_tgt_l])
                st.success(out)
            else:
                st.error("No text detected. Try a clearer image.")

# ── Tab 3 : API Docs ────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
### 🔌 Using LinguaFlow as an API

Your Vercel frontend calls this Streamlit app with query parameters:

```
GET https://YOUR-APP.streamlit.app/?api=1&text=bonjour&src=fr&tgt=en
```

#### Parameters
| Param | Values | Example |
|-------|--------|---------|
| `api` | `1` | enables API mode |
| `text` | any string | `bonjour` |
| `src`  | `fr` `en` `hi` `gu` | `fr` |
| `tgt`  | `fr` `en` `hi` `gu` | `en` |

#### Response
```json
{ "translation": "Hello", "src": "fr", "tgt": "en" }
```

#### Supported Pairs
| From | To | Method |
|------|----|--------|
| French | English | Direct |
| English | French | Direct |
| Hindi | English | Direct |
| English | Hindi | Direct |
| Gujarati | English | Direct |
| English | Gujarati | Direct |
| French ↔ Hindi | — | Bridged via English |
| French ↔ Gujarati | — | Bridged via English |

#### Example JavaScript (for your Vercel app)
```javascript
async function translateViaStreamlit(text, src, tgt) {
  const base = "https://YOUR-APP.streamlit.app/";
  const url  = `${base}?api=1&text=${encodeURIComponent(text)}&src=${src}&tgt=${tgt}`;
  const res  = await fetch(url);
  const html = await res.text();
  const match = html.match(/\\{"translation".*?\\}/);
  if (match) return JSON.parse(match[0]).translation;
  return null;
}
```
    """)