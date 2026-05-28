"""
Correction processing — transforms doctor corrections into RAG-ready text.
"""


def process_correction(request) -> str:
    """Convert a CorrectionRequest into human-readable text for RAG ingestion."""
    parts = []

    parts.append(
        f"CLINICAL CORRECTION: A medical professional reviewed a blood cell scan "
        f"(ID: {request.scan_id}). The automated CNN model predicted "
        f"'{request.original_prediction}', but the doctor corrected this to "
        f"'{request.corrected_prediction}'."
    )

    if request.corrected_species:
        species_info = {
            "P. falciparum": "Plasmodium falciparum — the most dangerous species, causes majority of malaria deaths. Requires immediate treatment with ACT.",
            "P. vivax": "Plasmodium vivax — can cause relapsing malaria due to dormant liver stages (hypnozoites). Requires primaquine for radical cure.",
            "P. ovale": "Plasmodium ovale — similar to P. vivax with relapsing potential. Less common.",
            "P. malariae": "Plasmodium malariae — causes chronic low-level infections. Can persist for decades if untreated.",
            "P. knowlesi": "Plasmodium knowlesi — zoonotic species from macaques. Can cause rapid and severe disease.",
        }
        detail = species_info.get(request.corrected_species, "")
        parts.append(f"Species identified: {request.corrected_species}. {detail}")

    if request.parasitemia_level:
        level_info = {
            "low": "Low parasitemia (<1% of red blood cells infected). Can often be treated with oral medication.",
            "moderate": "Moderate parasitemia (1-5% of red blood cells infected). Close monitoring required.",
            "high": "High parasitemia (>5% of red blood cells infected). This is a medical emergency requiring parenteral treatment.",
        }
        detail = level_info.get(request.parasitemia_level, "")
        parts.append(f"Parasitemia level: {request.parasitemia_level}. {detail}")

    if request.doctor_notes:
        parts.append(f"Doctor's clinical notes: {request.doctor_notes}")

    parts.append(
        "This correction has been recorded to improve future AI-assisted diagnoses."
    )

    return " ".join(parts)
