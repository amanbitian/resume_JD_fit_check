from candidate_fit.context_audit import ChatTemplate


def test_chat_template_renders_exact_messages():
    template = ChatTemplate("<system>", "</system><user>", "</user><assistant>")
    assert template.render("RULES", "INPUT") == (
        "<system>RULES</system><user>INPUT</user><assistant>"
    )
