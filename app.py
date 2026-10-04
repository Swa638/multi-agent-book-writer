import streamlit as st
from agents import CURATED_SOURCES, run_pipeline, safe_error_message

st.set_page_config(page_title="Multi-Agent Book Writer", page_icon="📚")
st.title("📚 Multi-Agent Book Writer")
st.caption("AI Engineer Take-Home Assignment")

with st.expander("Curated Research Sources", expanded=False):
    st.caption(
        "Researcher fetches these public pages directly. Google Search grounding is not used."
    )
    for source in CURATED_SOURCES:
        st.markdown(f"[{source['title']}]({source['url']})")
        st.caption(source["focus"])

topic = st.text_input(
    "Book topic",
    "Pay Me on UPI: How Digital Payments Changed Small Business in India"
)

if st.button("Run Multi-Agent System", type="primary", disabled=not topic.strip()):
    try:
        with st.status("Preparing agents...", expanded=True) as run_status:
            def update_status(agent, message):
                run_status.update(label=f"{agent}: {message}", state="running")
                run_status.write(f"**{agent}**: {message}")

            result = run_pipeline(topic.strip(), progress_callback=update_status)
            run_status.update(label="Book generation complete", state="complete")

        st.success("Book generated successfully!")
        st.subheader("Final Book")
        st.markdown(result["book"])

        st.subheader("Agent Trace")
        for item in result["trace"]:
            st.write(f"**{item['agent']}** — {item['summary']}")

        st.subheader("Sources by chapter")
        for source in result["sources"]:
            st.markdown(
                f"**Chapter {source['chapter']} [{source['number']}] "
                f"{source['title']}**"
            )
            st.link_button(
                "Open source",
                source["url"],
                key=f"source-{source['chapter']}-{source['number']}",
            )

        st.download_button(
            "Download Markdown",
            result["book"],
            "multi_agent_book.md",
            "text/markdown",
        )
    except Exception as error:
        st.error("The book could not be generated.")
        with st.expander("Debug/Error Details", expanded=True):
            exception_type = getattr(error, "original_type", type(error).__name__)
            st.write(f"Exception type: `{exception_type}`")
            st.code(safe_error_message(error))
