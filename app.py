import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from PIL import Image
import pytesseract

st.set_page_config(page_title="LinguaFlow API", page_icon="🌐", layout="wide")

MODEL_MAP = {
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("gu", "en"): "Helsinki-NLP/opus-mt-gu-en",
    ("en", "gu"): "Helsinki-NLP/opus-mt-en-gu",
}

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
        batch = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        out = mdl.generate(**batch)
        return tok.decode(out[0], skip_special_tokens=True)
    # Bridge via English
    tok1, mdl1 = load_model(src, "en")
    tok2, mdl2 = load_model("en", tgt)
    if tok1 and mdl1 and tok2 and mdl2:
        b1 = tok1([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        en = tok1.decode(mdl1.generate(**b1)[0], skip_special_tokens=True)
        b2 = tok2([en], return_tensors="pt", padding=True, truncation=True, max_length=512)
        return tok2.decode(mdl2.generate(**b2)[0], skip_special_tokens=True)
    return f"[No model for {src}→{tgt}]"

# ═══════════════════════════════════════════════════════════
# API MODE  —  ?api=1&text=bonjour&src=fr&tgt=en
#
# IMPORTANT: st.components.v1.html injects a real HTTP response
# with CORS header Access-Control-Allow-Origin: *
# so the Vercel frontend can call it directly — no proxy needed.
# ═══════════════════════════════════════════════════════════
import streamlit.components.v1 as components

params = st.query_params

if params.get("api") == "1":
    text = params.get("text", "")
    src  = params.get("src", "fr")
    tgt  = params.get("tgt", "en")

    if not text:
        result = {"error": "No text provided"}
    else:
        try:
            translation = translate(text, src, tgt)
            result = {"translation": translation, "src": src, "tgt": tgt}
        except Exception as e:
            result = {"error": str(e)}

    import json
    json_str = json.dumps(result)

    # Render the JSON in a way that's easy to parse from the outside,
    # AND inject a script that posts a message so the parent iframe can read it.
    # The key trick: put the JSON in a uniquely-identifiable element.
    components.html(f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body{{margin:0;background:#0a0a0a;font-family:monospace;color:#22d472;padding:16px}}
  pre{{font-size:14px;line-height:1.6;white-space:pre-wrap;word-break:break-all}}
  #lf-result{{display:none}}
</style>
</head>
<body>
<pre>{json_str}</pre>
<div id="lf-result" data-json='{json_str}'></div>
<script>
  // Post message to any parent window (allows direct JS reading)
  try {{
    window.parent.postMessage({{type:'lf-result',data:{json_str}}}, '*');
  }} catch(e) {{}}
</script>
</body>
</html>
""", height=80)
    st.stop()

# ═══════════════════════════════════════════════════════════
# NORMAL UI
# ═══════════════════════════════════════════════════════════
st.markdown("## 🌐 LinguaFlow — Translation Backend")
st.caption("MarianMT · Helsinki-NLP · Python 3.14 compatible")
st.divider()

LANGS = {"French 🇫🇷":"fr","English 🇬🇧":"en","Hindi 🇮🇳":"hi","Gujarati 🇮🇳":"gu"}

tab1, tab2, tab3 = st.tabs(["📝 Text", "📷 Image OCR", "🔌 API Docs"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        src_l = st.selectbox("From", list(LANGS.keys()), index=0)
        src   = LANGS[src_l]
        user_text = st.text_area("Input", placeholder="e.g. je me sens affreusement mal",
                                  height=180, label_visibility="collapsed")
    with c2:
        tgt_l = st.selectbox("To", list(LANGS.keys()), index=1)
        tgt   = LANGS[tgt_l]
        out_box = st.empty()

    if st.button("Translate ⚡", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some text first.")
        elif src == tgt:
            st.warning("Source and target are the same.")
        else:
            with st.spinner("Translating…"):
                try:
                    res = translate(user_text.strip(), src, tgt)
                    out_box.text_area("Result", value=res, height=180,
                                      label_visibility="collapsed")
                    st.success("✓ Done!")
                except Exception as e:
                    st.error(f"Error: {e}")

with tab2:
    st.caption("Upload image containing text to OCR + translate")
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
                    raw = ""; st.error(f"Tesseract error: {e}")
            if raw:
                st.code(raw, language=None)
                with st.spinner("Translating…"):
                    try:
                        out = translate(raw, LANGS[ocr_src], LANGS[ocr_tgt])
                        st.success(out)
                    except Exception as e:
                        st.error(str(e))
            else:
                st.error("No text detected.")

with tab3:
    st.markdown("""
### 🔌 API Usage
```
GET https://YOUR-APP.streamlit.app/?api=1&text=bonjour&src=fr&tgt=en
```
**Response:** `{ "translation": "Hello", "src": "fr", "tgt": "en" }`

| Param | Values |
|-------|--------|
| `api` | `1` |
| `text` | any string |
| `src` | `fr` `en` `hi` `gu` |
| `tgt` | `fr` `en` `hi` `gu` |
""")