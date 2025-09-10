from hiva.agents.semantics import rewrite_prompt

def test_rewrite_prompt_add_remove():
    p = "You are helpful."
    g = {"suggestions":{"add_phrase":"Be explicit.", "remove_phrase":"unrelated"}}
    p2 = rewrite_prompt(p, g)
    assert "Be explicit." in p2
    g2 = {"suggestions":{"remove_phrase":"helpful"}}
    p3 = rewrite_prompt(p2, g2)
    assert "helpful" not in p3
