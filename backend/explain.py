
import os
import json
import math

from dotenv import load_dotenv
from google import genai


load_dotenv()

GEMINI_MODEL = "gemini-3.5-flash-lite"



def validate_shap_features(features):

    if not isinstance(features, list):
        return []

    valid_features = []

    for item in features:

        if not isinstance(item, dict):
            continue

        feature = item.get("feature")
        shap_value = item.get("shap_value")
        feature_value = item.get("feature_value")

        if not isinstance(feature, str) or not feature.strip():
            continue

        try:
            shap_value = float(shap_value)
            feature_value = float(feature_value)
        except (TypeError, ValueError):
            continue

        if not math.isfinite(shap_value):
            continue

        if not math.isfinite(feature_value):
            continue

        valid_features.append({
            "feature": feature,
            "feature_value": feature_value,
            "shap_value": shap_value
        })

    return valid_features




def prepare_shap_evidence(
    shap_data=None,
    top_features=None,
    top_n=10
):
    """
    Prepare SHAP evidence using a fallback strategy.

    Priority:
    1. Full SHAP data.
    2. Saved top features.
    3. No explanation available.
    """

    if isinstance(shap_data, str):
        try:
            shap_data = json.loads(shap_data)
        except (json.JSONDecodeError, TypeError):
            shap_data = None

    if isinstance(shap_data, dict):

        base_value = shap_data.get("base_value")

        full_features = validate_shap_features(
            shap_data.get("shap_values")
        )

        if full_features:

            full_features.sort(
                key=lambda item: abs(item["shap_value"]),
                reverse=True
            )

            return {
                "base_value": base_value,
                "top_shap_features": full_features[:top_n],
                "source": "full_shap_values"
            }

    fallback_features = validate_shap_features(top_features)

    if fallback_features:

        fallback_features.sort(
            key=lambda item: abs(item["shap_value"]),
            reverse=True
        )

        return {
            "base_value": None,
            "top_shap_features": fallback_features[:top_n],
            "source": "saved_top_features"
        }

    return None



def generate_gemini_explanation(
    fraud_probability: float,
    decision_threshold: float,
    predicted_class: int,
    shap_data=None,
    top_features=None
) -> str:
    """
    Generate an analyst-friendly explanation using Gemini
    and validated SHAP evidence.
    """

    # Validate SHAP evidence before calling Gemini
    shap_evidence = prepare_shap_evidence(
        shap_data=shap_data,
        top_features=top_features
    )

    if shap_evidence is None:
        raise ValueError(
            "SHAP explanation unavailable: "
            "no valid feature contributions were found."
        )

    # Validate prediction information
    if not math.isfinite(float(fraud_probability)):
        raise ValueError("Invalid fraud probability.")

    if not math.isfinite(float(decision_threshold)):
        raise ValueError("Invalid decision threshold.")

    if predicted_class not in (0, 1):
        raise ValueError("Predicted class must be 0 or 1.")

    # Prepare evidence for Gemini
    evidence = {
        "fraud_probability": fraud_probability,
        "decision_threshold": decision_threshold,
        "predicted_class": predicted_class,
        "shap_evidence": shap_evidence
    }

    prompt = f"""
You are an AI assistant supporting fraud analysts.

Your task is to explain an existing machine learning
prediction using only the supplied evidence.

PREDICTION EVIDENCE:
{json.dumps(evidence, indent=2, default=str)}

INSTRUCTIONS:

1. Explain the model's fraud probability and classification.
2. Identify the strongest SHAP contributions.
3. A positive SHAP value pushes the model output
   toward fraud, while a negative value pushes it
   away from fraud.
4. Explain the contributions using their actual
   numerical values and directions.
5. SHAP values are raw model-output contributions,
   not percentages.
6. V1 through V28 are anonymized PCA components.
   Never invent or assume their real-world meanings.
7. Do not claim that a predicted transaction is
   confirmed fraud.
8. Do not invent transaction, customer, merchant,
   location, or behavioural information.
9. If only some SHAP contributions are available,
   explain only those contributions.
10. Keep the explanation professional and concise,
    approximately 100-150 words.

Return only the explanation.
"""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Gemini API key is not configured."
        )

    client = genai.Client(api_key=api_key)

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        if not response.text or not response.text.strip():
            raise RuntimeError(
                "Gemini returned an empty explanation."
            )

        return response.text.strip()

    finally:
        client.close()
