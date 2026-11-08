import streamlit as st

from shorts_clipper import DEFAULT_MODEL, run

st.set_page_config(page_title="Shorts Clipper", page_icon="✂️")
st.title("✂️ YouTube → Shorts")
st.caption("Paste a link + your Gemini API key. Only use videos you own or have permission to reuse.")

url = st.text_input("YouTube link")
key = st.text_input("Gemini API key", type="password")
col1, col2, col3 = st.columns(3)
n = col1.number_input("Max clips", 1, 10, 5)
fit = col2.selectbox("Framing", ["crop", "blur"])
captions = col3.checkbox("Burn captions", value=False)
model = st.text_input("Gemini model", DEFAULT_MODEL)

if st.button("Make Shorts", type="primary", disabled=not (url and key)):
    with st.status("Working...", expanded=True) as status:
        try:
            results = run(url, key, n=int(n), fit=fit, captions=captions,
                          model=model, log=st.write)
            status.update(label="Done", state="complete")
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(str(e))
            st.stop()

    for r in results:
        st.divider()
        st.subheader(f"{r['title']}  ·  score {r['score']}/10")
        st.video(r["file"])
        st.markdown(f"**Hook:** {r['hook']}  \n**Why:** {r['reason']}  \n"
                    f"**Tags:** {' '.join('#' + t.lstrip('#') for t in r['hashtags'])}  \n"
                    f"**Source range:** {r['start_s']}s–{r['end_s']}s")
