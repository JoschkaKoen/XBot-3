"""
Disney style — Pixar-style 3D animated aesthetic.

Lifted verbatim from nodes/generate_image.py (the `if image_style == "disney"`
branch) plus the Disney Midjourney suffix and grok_video's Disney motion
guidance.
"""

from .base import PromptContext


midjourney_suffix = (
    ", Pixar 3D animation style, expressive character design, "
    "strong silhouettes, cinematic directional lighting, rich saturated colours, "
    "8K render, polished and characterful"
)


motion_guidance = (
    "The image is a 3D CGI / Pixar-Disney animated scene. "
    "Motion should feel like a held frame from an animated feature: "
    "slightly exaggerated but smooth character motion (a slow blink, "
    "a gentle head tilt, a soft smile forming), with lush environmental "
    "animation (leaves drifting, light rays shifting, dust motes floating). "
    "Keep the storybook warmth — nothing jarring."
)


_DISNEY_AESTHETIC = (
    "Style: polished 3D CGI animation in the style of Pixar and Walt Disney. "
    "Stylised shapes with clear, confident silhouettes. "
    "Characters have expressive eyes and readable facial features — personality-driven, not overly saccharine. "
    "Colour palette: rich, harmonious tones — warm ambers, deep blues, forest greens, and saturated accents "
    "grounded by neutral mid-tones. "
    "Lighting: cinematic directional light with strong contrast, rim highlights, and atmospheric depth, "
    "as if lit for a Pixar feature film. "
    "Background: a purposeful environment with painterly detail, soft depth of field, and clear visual hierarchy. "
    "Everything feels polished, characterful, and cinematic. "
    "The image should look like a still from a Pixar or Disney animated feature."
)


def build_image_prompt(
    *, funny: bool, is_zit: bool, ctx: PromptContext
) -> tuple[str, str]:
    if funny and ctx.example_de:
        tweet_context = f"Full tweet:\n{ctx.full_tweet}\n\n" if ctx.full_tweet else ""
        if is_zit:
            img_req = (
                f"A {ctx.source_language} learning tweet contains a joke. "
                f"Write a rich, detailed image description for a Disney/Pixar-style 3D animated scene "
                f"in a {ctx.aspect_hint} that shows the punchline clearly and with visual wit. "
                "Use full natural-language sentences, not comma-separated tags. "
                "Describe the setting, characters, their expressions and body language, the lighting, "
                "and the colour palette in concrete detail — aim for 100–200 words.\n\n"
                f"{tweet_context}"
                f"{ctx.source_language} sentence: \"{ctx.example_de}\"\n"
                f"{ctx.target_language} sentence: \"{ctx.example_en}\"\n\n"
                "Step 1 — Identify the punchline: find the ironic twist, absurd contrast, or subverted expectation.\n"
                "Step 2 — Stage it visually: describe expressive body language and facial expressions that land the joke — "
                "the comedy should be immediately readable from the image alone.\n"
                "Step 3 — Describe the environment: background, lighting direction (rim, key, fill), colour temperature.\n"
                "Step 4 — Keep it clean: ONE main character, ONE clear joke, uncluttered focused background.\n"
                "Step 5 — Keep it family-friendly: warm, uplifting, never dark or unsettling.\n\n"
                f"{_DISNEY_AESTHETIC}"
                f"{ctx.rules}"
            )
            system_prompt = (
                "You are an expert Disney/Pixar 3D animation image description writer for Z-Image-Turbo, "
                "a model that excels at detailed natural-language scene descriptions. "
                "Write rich, flowing prose that covers subject, environment, lighting, colour palette, and mood — "
                "never use comma-separated tags. "
                "Every description should feel like a scene memo from a Pixar director: "
                "specific, visual, and full of personality. "
                "Output only the image description."
            )
        else:
            img_req = (
                f"A {ctx.source_language} learning tweet contains a joke. "
                "Create an image generation prompt for a Disney/Pixar-style 3D animated scene "
                "that shows the punchline of the joke clearly and with visual wit.\n\n"
                f"{tweet_context}"
                f"{ctx.source_language} sentence: \"{ctx.example_de}\"\n"
                f"{ctx.target_language} sentence: \"{ctx.example_en}\"\n\n"
                "Step 1 — Identify the punchline: find the ironic twist, absurd contrast, or subverted expectation.\n"
                "Step 2 — Stage it visually: use expressive body language and facial expressions to land the joke — "
                "the comedy should be immediately readable from the image alone.\n"
                "Step 3 — Make it cinematic and polished: deliberate lighting, strong composition, rich colours. "
                "Think of a memorable frame from a Pixar feature — that level of craft and visual storytelling.\n"
                "Step 4 — Keep it clean: ONE main character, ONE clear joke, uncluttered focused background.\n"
                "Step 5 — Keep it family-friendly: warm, uplifting, never dark or unsettling.\n\n"
                f"{_DISNEY_AESTHETIC}"
                f"{ctx.rules}"
            )
            system_prompt = (
                "You are an expert Disney/Pixar 3D animation prompt engineer. "
                "You write image prompts that produce polished, expressive, and funny animated stills. "
                "Every prompt you write feels like a frame from a Pixar feature: "
                "strong character silhouettes, expressive faces, cinematic lighting, rich cohesive colours. "
                "Humour is conveyed through clear visual storytelling and expressive performance, never saccharine excess. "
                "Never mention photography, cameras, lenses, or film. "
                "No parameter flags. No double hyphens. Output only the image description."
            )
    else:
        if is_zit:
            img_req = (
                f"Write a rich, detailed image description for a Disney/Pixar-style 3D animated scene "
                f"in a {ctx.aspect_hint}. "
                "Use full natural-language sentences, not comma-separated tags. "
                "Describe the setting, the main character(s), their pose and expression, "
                "the lighting (direction, quality, colour temperature), "
                "and the colour palette in concrete detail — aim for 100–200 words.\n\n"
                f"Scene to illustrate: \"{ctx.example_en}\"\n\n"
                "Design a visually compelling, characterful scene that brings this sentence to life. "
                "Characters should have expressive features and strong readable silhouettes. "
                "The scene should look like a cinematic still from a Pixar or Disney animated feature — "
                "polished, purposeful, and full of personality without being saccharine.\n\n"
                f"{_DISNEY_AESTHETIC}"
                "No text visible in the image."
                f"{ctx.rules}"
            )
            system_prompt = (
                "You are an expert Disney/Pixar 3D animation image description writer for Z-Image-Turbo, "
                "a model that excels at detailed natural-language scene descriptions. "
                "Write rich, flowing prose that covers subject, environment, lighting, colour palette, and mood — "
                "never use comma-separated tags. Aim for the detail of a Pixar art director's scene brief. "
                "Output only the image description."
            )
        else:
            img_req = (
                "Create an image generation prompt for a Disney/Pixar-style 3D animated scene.\n\n"
                f"Sentence: \"{ctx.example_en}\"\n\n"
                "Design a visually compelling, characterful scene that brings this sentence to life. "
                "Characters should have expressive features and strong readable silhouettes. "
                "The scene should look like a cinematic still from a Pixar or Disney animated feature — "
                "polished, purposeful, and full of personality without being saccharine.\n\n"
                f"{_DISNEY_AESTHETIC}"
                "No text in the image."
                f"{ctx.rules}"
            )
            system_prompt = (
                "You are an expert Disney/Pixar 3D animation prompt engineer. "
                "You write image prompts that produce cinematic, expressive, Pixar-quality stills. "
                "Strong character design, deliberate lighting, rich cohesive colour palette — "
                "every element should feel polished, purposeful, and full of personality. "
                "Avoid over-sweetening: aim for charming and engaging, not saccharine. "
                "Never mention photography, cameras, lenses, or film. "
                "No parameter flags. No double hyphens. Output only the image description."
            )

    return img_req, system_prompt
