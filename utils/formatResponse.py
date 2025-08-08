import re

def format_response_as_points(text: str) -> str:
    """
    Converts a long raw text into a clean, pointwise format with headings and bullets.
    """
    # Split by numbered or titled sections
    sections = re.split(r"(?=\d+\.\s|Conclusion:)", text.strip())

    formatted = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Add markdown-like formatting for headings
        if re.match(r"^\d+\.\s", section):
            heading, *rest = section.split(":", 1)
            if rest:
                formatted.append(f"**{heading.strip()}**: {rest[0].strip()}")
            else:
                formatted.append(f"**{heading.strip()}**")
        elif section.lower().startswith("conclusion"):
            formatted.append(f"\n**Conclusion**: {section[len('Conclusion:'):].strip()}")
        else:
            formatted.append(section)

    return "\n\n".join(formatted)
