"""
Photographic style — realistic editorial photography aesthetic.

Lifted verbatim from nodes/generate_image.py (the `else` branch of the old
`if image_style == "disney"` block) plus the photographic Midjourney suffix
and grok_video's photographic motion guidance.
"""

from .base import PromptContext


midjourney_suffix = (
    ", shot on Canon EOS R5, 35mm lens, natural lighting, "
    "RAW photo, ultra realistic, 8k UHD, "
    "positive joyful atmosphere, warm and welcoming, bright uplifting mood, "
    "subjects with natural warm smiles, positive facial expressions"
)


motion_guidance = (
    "The image is photorealistic / editorial photography. "
    "Motion should feel like a locked-off cinema camera that barely moves: "
    "a very slow push-in or static shot with shallow depth-of-field rack. "
    "Subject motion is restrained and lifelike (breathing, a micro-expression, "
    "weight shift). Environmental motion is naturalistic (wind in foliage, "
    "steam rising, light flicker through clouds)."
)


_IMMERSIVE = (
    "Frame the shot so the viewer feels placed directly inside the scene: "
    "The composition should feel lived-in and immediate, as if the viewer just walked into the moment. "
)
_CLEAN_AESTHETIC = (
    "Composition: ONE clear subject, uncluttered frame, minimal background elements. "
    "The joke or mood must be immediately readable at a glance — never crowd the scene. "
)
_AESTHETIC = (
    "Aesthetics: make this image genuinely beautiful — not just technically correct. "
    "Think carefully about: harmonious colour palette (warm, vibrant, or richly contrasted), "
    "flattering and dramatic natural light (golden hour, soft side-light, or crisp morning sun), "
    "shallow depth of field to isolate the subject against a beautifully blurred background, "
    "and a composition that would stop someone mid-scroll. "
    "The image should look like a professional editorial photo that people want to share for its looks alone. "
)


def build_image_prompt(
    *, funny: bool, is_zit: bool, ctx: PromptContext
) -> tuple[str, str]:
    if funny and ctx.example_de:
        tweet_context = f"Full tweet:\n{ctx.full_tweet}\n\n" if ctx.full_tweet else ""
        if is_zit:
            img_req = (
                f"A {ctx.source_language} learning tweet contains a joke. "
                f"Write a rich, detailed photographic scene description for Z-Image-Turbo in a {ctx.aspect_hint} "
                "that is BOTH visually stunning AND makes the punchline instantly obvious. "
                "Use flowing natural-language sentences — not comma tags. "
                "Cover: the subject and their expression/body language, the environment, "
                "the lighting (direction, quality, colour temperature), the colour palette, "
                "and the camera framing. Aim for 100–200 words.\n\n"
                f"{tweet_context}"
                f"{ctx.source_language} sentence: \"{ctx.example_de}\"\n"
                f"{ctx.target_language} sentence: \"{ctx.example_en}\"\n\n"
                "Step 1 — Identify the punchline: the ironic twist, subverted expectation, or absurd contrast.\n"
                "Step 2 — Stage it: describe a real scene where expressions or body language land the joke — "
                "the comedy must read from the image alone.\n"
                "Step 3 — Make it beautiful: golden-hour or cinematic light, rich colours, "
                "shallow depth of field — a composition worth sharing.\n"
                "Step 4 — Keep it photorealistic: no CGI, no animation, no cartoon. ONE subject, uncluttered frame.\n\n"
                f"{_IMMERSIVE}"
                f"{_CLEAN_AESTHETIC}"
                f"{_AESTHETIC}"
                f"{ctx.rules}"
            )
            system_prompt = (
                "You are an expert photographic scene description writer for Z-Image-Turbo. "
                "This model responds best to detailed natural-language prose describing subjects, "
                "environments, lighting, and colour — never comma-separated tag lists. "
                "Your descriptions are both visually stunning and instantly funny: "
                "a clear visual punchline combined with cinematic photographic beauty. "
                "ALL output must be photorealistic. Never describe CGI, animation, or cartoon styles. "
                "Output only the scene description."
            )
        else:
            img_req = (
                f"A {ctx.source_language} learning tweet contains a joke. Your job is to create an image generation prompt that is "
                "BOTH visually stunning AND makes the punchline of the joke instantly obvious.\n\n"
                f"{tweet_context}"
                f"{ctx.source_language} sentence: \"{ctx.example_de}\"\n"
                f"{ctx.target_language} sentence: \"{ctx.example_en}\"\n\n"
                "Step 1 — Identify the punchline: find the ironic twist, the subverted expectation, or the absurd contrast.\n"
                "Step 2 — Stage it visually: design a real photographic scene that shows the punchline, "
                "body language, or clever composition. The comedy must land from the image alone.\n"
                "Step 3 — Make it beautiful: apply deliberate aesthetic choices — golden-hour light, rich colours, "
                "shallow depth of field, a composition worth sharing for its looks alone. "
                "Beauty and humour must coexist: a stunning photograph that is also funny.\n"
                "Step 4 — Keep it clean and readable: ONE subject, ONE joke, uncluttered frame.\n"
                "Step 5 — Keep it positive and photorealistic: warm, light-hearted, family-friendly. "
                "The output MUST be a photorealistic photograph — never animation, illustration, or cartoon.\n\n"
                f"{_IMMERSIVE}"
                f"{_CLEAN_AESTHETIC}"
                f"{_AESTHETIC}"
                "Photorealistic photography ONLY — no CGI, no illustration, no cartoon, no animation."
                f"{ctx.rules}"
            )
            system_prompt = (
                "You are an expert photographic image prompt engineer who creates images that are both visually stunning "
                "and instantly funny. Your prompts always combine: (1) a clear visual punchline readable from the photo alone, "
                "and (2) deliberately beautiful photographic aesthetics — perfect light, rich colours, shallow depth of field, "
                "editorial composition. "
                "ALL output must be photorealistic photography. NEVER describe CGI, illustration, animation, or Pixar/Disney style — "
                "even for unusual or humorous subjects. If a scene seems absurd, make it work as a clever, well-composed photograph. "
                "Always include specific camera model, lens, and lighting descriptors "
                "(e.g. 'shot on Sony A7IV, 50mm f/1.4, golden hour backlight'). "
                "Never use words like 'painting', 'illustration', 'artistic', 'rendered', 'digital art', 'animated', 'cartoon'. "
                "No parameter flags. No double hyphens. Output only the description."
            )
    else:
        if is_zit:
            img_req = (
                f"Write a rich, detailed photographic scene description for Z-Image-Turbo in a {ctx.aspect_hint}. "
                "Use flowing natural-language sentences — not comma tags. "
                "Describe: the main subject (identity, pose, expression), the environment and background, "
                "the lighting (direction, quality, colour temperature — e.g. warm golden-hour backlight, "
                "soft diffused daylight, cinematic key light), the colour palette, "
                "and the overall mood. Aim for 100–200 words.\n\n"
                f"Scene to illustrate: \"{ctx.example_en}\"\n\n"
                f"{_IMMERSIVE}"
                f"{_CLEAN_AESTHETIC}"
                f"{_AESTHETIC}"
                "Photorealistic scene — no text visible in the image."
                f"{ctx.rules}"
            )
            system_prompt = (
                "You are an expert photographic scene description writer for Z-Image-Turbo, "
                "a model that excels at detailed natural-language descriptions. "
                "Write rich, flowing prose that covers subject, environment, lighting direction and quality, "
                "colour palette, and mood — never comma-separated tags. "
                "Every description should be a complete cinematic scene brief: "
                "specific enough that a photographer could recreate the exact shot. "
                "Output only the scene description."
            )
        else:
            img_req = (
                f"Generate an image generation prompt for a photorealistic, aesthetically stunning {ctx.aspect_hint} photograph.\n\n"
                f"Sentence: \"{ctx.example_en}\"\n\n"
                f"{_IMMERSIVE}"
                f"{_CLEAN_AESTHETIC}"
                f"{_AESTHETIC}"
                "No text in the image."
                f"{ctx.rules}"
            )
            system_prompt = (
                "You are an expert image generation prompt engineer who creates images that look like professional "
                "editorial photography. Every prompt you write is deliberately beautiful: perfect light, "
                "rich harmonious colours, shallow depth of field, and a composition people want to share. "
                "ONE clear subject, uncluttered frame — every element serves the main subject. "
                "Always include specific camera model, lens, and lighting descriptors (e.g. 'shot on Sony A7IV, 50mm f/1.4, golden hour'). "
                "Never use words like 'painting', 'illustration', 'artistic', 'rendered', 'digital art'. "
                "No parameter flags. No double hyphens. Output only the description."
            )

    return img_req, system_prompt
