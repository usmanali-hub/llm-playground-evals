"""Loads versioned prompt files from prompts/ and renders both versions of the same prompt side by side.

Shows why versioning in files (rather than editing a string in code) is useful: v1 and v2 of "summarize" coexist,
so a caller can pin v1 for reproducibility or pick up v2's stricter tone/length wording deliberately, and the
reason v2 exists is written down in its own frontmatter instead of lost to a diff.
"""

from pathlib import Path

from llm_playground import PromptLibrary

library = PromptLibrary.load(Path(__file__).parent.parent / "prompts")
print("known prompts:", library.names())

for name in library.names():
    for version in library.versions(name):
        prompt = library.get(name, version)
        print(f"\n{name} v{version} — {prompt.description}")
        print(f"  declared variables: {prompt.variables}")

text = "The library is open from 9am to 8pm on weekdays, and 10am to 5pm on weekends."
print("\n--- summarize v1 ---")
print(library.render("summarize", version=1, text=text))
print("--- summarize v2 (latest) ---")
print(library.render("summarize", tone="casual", max_words="15", text=text))
