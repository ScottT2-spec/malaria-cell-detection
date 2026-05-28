"""
K2 Think V2 API Client
Handles all LLM calls — clinical analysis, chat, report generation.
Includes retry logic, confidence-aware prompting, and robust JSON parsing.
"""

import json
import asyncio
import httpx
import re
import logging

logger = logging.getLogger("malariaai.k2")


class K2ThinkClient:
    """Client for MBZUAI K2 Think V2 API (OpenAI-compatible)."""

    def __init__(self, api_key: str, api_url: str, model: str):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.timeout = 120.0  # K2 Think is a reasoning model — needs more time
        self.max_retries = 3

    async def _call(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        Make a chat completion request to K2 Think V2 with retry logic.
        Retries on 429 (rate limit), 500, 502, 503, 504 with exponential backoff.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        last_error = None

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.api_url, headers=headers, json=payload
                    )

                    # Rate limit — respect Retry-After header
                    if response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                        logger.warning(f"Rate limited, waiting {retry_after}s (attempt {attempt + 1})")
                        await asyncio.sleep(retry_after)
                        continue

                    # Server errors — retry with backoff
                    if response.status_code in (500, 502, 503, 504):
                        wait = 2 ** attempt
                        logger.warning(f"Server error {response.status_code}, retrying in {wait}s")
                        await asyncio.sleep(wait)
                        continue

                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"]

            except httpx.TimeoutException:
                last_error = "Request timed out"
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
                await asyncio.sleep(2 ** attempt)
            except httpx.ConnectError:
                last_error = "Connection failed"
                logger.warning(f"Connection failed on attempt {attempt + 1}/{self.max_retries}")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                last_error = str(e)
                logger.error(f"Unexpected error: {e}")
                break

        raise RuntimeError(f"K2 Think API failed after {self.max_retries} attempts: {last_error}")

    def _parse_json_response(self, raw: str, defaults: dict) -> dict:
        """
        Robust JSON parser for K2 Think V2 (a reasoning model).

        K2 Think outputs chain-of-thought reasoning BEFORE the actual JSON.
        We need to find and extract the JSON object from within the reasoning text.
        Handles: reasoning preamble, markdown fences, partial JSON, nested objects.
        """
        cleaned = raw.strip()

        # Strategy 1: Direct parse (unlikely with reasoning model, but try)
        try:
            return {**defaults, **json.loads(cleaned)}
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find JSON inside markdown code fences
        fence_match = re.search(r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```', cleaned, re.DOTALL)
        if fence_match:
            try:
                return {**defaults, **json.loads(fence_match.group(1))}
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find the LAST complete JSON object in the text
        # (K2 Think reasons first, then outputs JSON at the end)
        # Use bracket matching for nested objects
        json_objects = []
        i = 0
        while i < len(cleaned):
            if cleaned[i] == '{':
                depth = 0
                start = i
                in_string = False
                escape = False
                for j in range(i, len(cleaned)):
                    c = cleaned[j]
                    if escape:
                        escape = False
                        continue
                    if c == '\\' and in_string:
                        escape = True
                        continue
                    if c == '"' and not escape:
                        in_string = not in_string
                        continue
                    if not in_string:
                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0:
                                json_objects.append(cleaned[start:j + 1])
                                i = j
                                break
                else:
                    break
            i += 1

        # Try each JSON object found (prefer the last/largest one)
        for candidate in reversed(json_objects):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and len(parsed) >= 2:
                    return {**defaults, **parsed}
            except json.JSONDecodeError:
                continue

        # Strategy 4: Extract key fields from the reasoning text itself
        logger.warning("Could not parse JSON from K2 response, extracting from reasoning text")
        result = dict(defaults)

        # Try to pull out quoted values for known keys
        for key in defaults:
            pattern = rf'"{key}"\s*:\s*"((?:[^"\\]|\\.){{10,}})"'
            match = re.search(pattern, cleaned, re.DOTALL)
            if match:
                result[key] = match.group(1).replace('\\"', '"').replace('\\n', '\n')

        # If we got nothing useful, use the last paragraph as clinical summary
        if result.get("clinical_summary") == defaults.get("clinical_summary"):
            # Find the most informative paragraph in the reasoning
            paragraphs = [p.strip() for p in cleaned.split('\n\n') if len(p.strip()) > 50]
            if paragraphs:
                result["clinical_summary"] = paragraphs[-1][:600]

        return result

    # ── Confidence-Aware Analysis ────────────────────────────────

    def _confidence_context(self, prediction: str, confidence: float) -> str:
        """
        Generate confidence-aware instructions for K2 Think.
        Different confidence levels require fundamentally different clinical responses.
        """
        if prediction == "parasitized":
            if confidence >= 0.95:
                return """HIGH CONFIDENCE POSITIVE (≥95%): The CNN is highly certain parasites are present.
Focus on: species-specific treatment protocols, severity assessment, immediate next steps.
Flag if parasitemia appears high — this could be a medical emergency."""
            elif confidence >= 0.80:
                return """MODERATE-HIGH CONFIDENCE POSITIVE (80-95%): Likely positive but confirmation is important.
Focus on: recommended confirmatory tests, presumptive treatment if in endemic area, follow-up protocol."""
            elif confidence >= 0.60:
                return """LOW-MODERATE CONFIDENCE POSITIVE (60-80%): Borderline result — could be a true positive or a false positive.
Focus on: STRONG recommendation for expert review, possible artifacts or image quality issues,
recommend thick blood smear + RDT for confirmation. DO NOT recommend treatment based solely on this result."""
            else:
                return """VERY LOW CONFIDENCE (<60%): The model is barely above chance. This result is NOT reliable.
Focus on: the unreliability of this result, mandatory expert review, image quality assessment,
recommend collecting a new sample. This should NOT be used for any clinical decision."""

        else:  # uninfected
            if confidence >= 0.95:
                return """HIGH CONFIDENCE NEGATIVE (≥95%): The CNN is highly certain no parasites are present.
Focus on: reassurance with caveats, remind that a single negative doesn't rule out malaria
if symptoms are present, recommend re-testing in 12-24h if febrile."""
            elif confidence >= 0.80:
                return """MODERATE-HIGH CONFIDENCE NEGATIVE (80-95%): Likely negative but not definitive.
Focus on: recommend repeat testing especially if symptomatic, consider RDT as complementary test."""
            elif confidence >= 0.60:
                return """LOW-MODERATE CONFIDENCE NEGATIVE (60-80%): Borderline result — could be a false negative.
Focus on: HIGH RISK of missed infection, strongly recommend repeat microscopy + RDT,
treat presumptively if clinical suspicion is high (WHO recommends treating suspected cases in endemic areas)."""
            else:
                return """VERY LOW CONFIDENCE (<60%): Unreliable result.
Focus on: this result should be disregarded, mandatory re-examination, treat based on clinical judgment."""

    async def clinical_analysis(
        self,
        prediction: str,
        confidence: float,
        infected_prob: float,
        healthy_prob: float,
        rag_context: list[dict],
        corrections_context: list[dict],
    ) -> dict:
        """
        Generate clinical analysis from CNN results + RAG context.
        Uses confidence-aware prompting for medically appropriate responses.
        """

        # Build context from RAG results
        guidelines_text = "\n\n".join(
            [f"[Source: {c['source']}] (relevance: {c.get('score', 'N/A')})\n{c['text']}"
             for c in rag_context]
        ) if rag_context else "No specific guidelines retrieved."

        corrections_text = "\n\n".join(
            [f"[Correction by {c.get('doctor_id', 'Doctor')}]: {c['text']}"
             for c in corrections_context]
        ) if corrections_context else "No prior corrections for similar cases."

        # Confidence-aware clinical context
        confidence_guidance = self._confidence_context(prediction, confidence)

        system_prompt = f"""You are MalariaAI Clinical Assistant, an expert medical AI specializing in malaria diagnosis and treatment. You work alongside a CNN-based blood cell detection system (95.43% accuracy, trained on 27,558 NIH images) to provide clinical decision support.

Your responsibilities:
1. Interpret CNN results with appropriate confidence calibration
2. Provide evidence-based treatment guidance citing WHO/CDC protocols
3. Assess severity and flag emergencies
4. Educate healthcare workers and patients
5. Incorporate learnings from prior doctor corrections

CONFIDENCE CALIBRATION:
{confidence_guidance}

MEDICAL GUIDELINES (Retrieved via RAG):
{guidelines_text}

DOCTOR CORRECTIONS ON SIMILAR CASES:
{corrections_text}

CRITICAL RULES:
- You are a screening SUPPORT tool — never claim to make a diagnosis
- Always recommend professional confirmation
- For severe/emergency indicators, use urgent language
- Cite specific WHO/CDC guidelines when recommending treatment
- If confidence is below 70%, emphasize the need for confirmatory testing"""

        user_prompt = f"""A blood cell microscopy image was analyzed by our CNN model:

**CNN Detection Results:**
- Prediction: {prediction.upper()}
- Confidence: {confidence:.1%}
- Parasitized probability: {infected_prob:.1%}
- Uninfected probability: {healthy_prob:.1%}

Generate a clinical analysis. Respond in this exact JSON format:

{{
  "clinical_summary": "2-3 sentence clinical interpretation calibrated to the confidence level",
  "severity_assessment": "none|low|moderate|high|critical — with explanation",
  "treatment_guidance": "Evidence-based treatment recommendations from WHO/CDC. Include specific drug names, dosages, and duration if parasitized. Include confidence caveats.",
  "patient_education": "Clear, simple explanation for the patient about what results mean and what to do next",
  "guidelines_cited": ["Specific guidelines or sources referenced"],
  "follow_up_questions": ["3-5 clinically relevant follow-up questions the healthcare worker should ask/consider"]
}}

Return ONLY valid JSON."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        defaults = {
            "clinical_summary": "Analysis could not be completed. Please consult a healthcare professional.",
            "severity_assessment": "unknown — unable to assess, seek professional evaluation",
            "treatment_guidance": "Please consult a healthcare professional for treatment guidance.",
            "patient_education": "Your blood sample has been analyzed. Please follow up with your doctor for a complete evaluation.",
            "guidelines_cited": ["WHO Guidelines for Malaria (2023)"],
            "follow_up_questions": [
                "What symptoms is the patient experiencing?",
                "When did symptoms begin?",
                "Has the patient traveled to a malaria-endemic area?",
            ],
        }

        try:
            raw = await self._call(messages, temperature=0.15, max_tokens=4096)
            result = self._parse_json_response(raw, defaults)
        except RuntimeError as e:
            logger.error(f"K2 Think API error during analysis: {e}")
            result = dict(defaults)
            result["clinical_summary"] = (
                f"AI analysis temporarily unavailable. CNN detected: {prediction} "
                f"({confidence:.0%} confidence). Please interpret with clinical judgment."
            )

        return result

    async def chat(
        self,
        message: str,
        conversation_history: list[dict],
        rag_context: list[dict],
    ) -> dict:
        """Interactive Q&A about malaria, diagnosis, treatment."""

        context_text = "\n\n".join(
            [f"[{c['source']}]: {c['text']}" for c in rag_context]
        ) if rag_context else ""

        system_prompt = f"""You are MalariaAI Assistant — a knowledgeable, approachable medical AI helping healthcare workers and patients understand malaria diagnosis, treatment, and prevention.

Use the following retrieved medical knowledge to inform your answers:
{context_text}

Guidelines:
- Be accurate and cite sources when possible
- Use clear language accessible to community health workers
- Always recommend professional medical consultation for treatment decisions
- If unsure, say so honestly — never guess on medical questions
- For drug dosages, always cite the source guideline"""

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (bounded)
        for msg in conversation_history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": message})

        try:
            answer = await self._call(messages, temperature=0.3, max_tokens=4096)
        except RuntimeError as e:
            logger.error(f"K2 Think API error during chat: {e}")
            answer = "I'm temporarily unable to process your question. Please try again shortly."

        # Extract source citations from the answer
        sources = []
        for ctx in rag_context:
            source_lower = ctx["source"].lower()
            if source_lower in answer.lower() or any(
                word in answer.lower()
                for word in source_lower.split()
                if len(word) > 4
            ):
                sources.append(ctx["source"])

        return {"answer": answer, "sources": list(set(sources))}
