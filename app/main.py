"""
Project 1: AI-Powered Text Analysis Agent
==========================================
A multi-step AI agent built with LangGraph, served via FastAPI.
Demonstrates stateful graph orchestration, conditional routing,
tool use, and production API design.

Core Stack: LangGraph · FastAPI · Python
Author: Candace Grant · Threshold ML Solutions, LLC
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Literal, Annotated
from datetime import datetime, timezone
from enum import Enum
import time
import os
import re
import math
import json
import asyncio
import operator

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LangGraph-style State Machine (no external LLM dependency)
#
# This implements the LangGraph pattern from scratch so the project
# runs without API keys. In production, swap the analysis functions
# for actual LLM calls via langchain + langgraph.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ── Graph State Definition ────────────────────────────────────────
class AgentState:
    """
    Typed state container for the LangGraph agent.

    In production LangGraph, this would be:
        class AgentState(TypedDict):
            text: str
            intent: str
            ...

    We use a class here to keep it dependency-free while preserving
    the exact same pattern.
    """
    def __init__(self, text: str, threshold: float = 0.5):
        self.text = text
        self.threshold = threshold
        # Populated by graph nodes
        self.intent: str = ""
        self.text_features: dict = {}
        self.spam_analysis: dict = {}
        self.sentiment_analysis: dict = {}
        self.entity_analysis: dict = {}
        self.risk_assessment: dict = {}
        self.final_report: dict = {}
        self.route_taken: list[str] = []
        self.processing_times: dict = {}
        self.errors: list[str] = []


# ── Node Functions (each is a node in the LangGraph) ──────────────

def classify_intent_node(state: AgentState) -> AgentState:
    """
    NODE 1: Intent Classification
    Determines what type of analysis the text needs.
    Routes to different downstream nodes based on intent.

    LangGraph equivalent:
        graph.add_node("classify_intent", classify_intent_node)
    """
    start = time.perf_counter()
    text_lower = state.text.lower()

    # Intent signals
    spam_signals = sum(1 for w in ["free", "winner", "click", "prize", "urgent",
                                    "offer", "subscribe", "discount", "act now",
                                    "limited time", "congratulations", "buy now",
                                    "earn money", "no cost", "risk free", "claim",
                                    "100%", "double your", "million", "cash bonus",
                                    "credit card", "apply now", "order now"]
                       if w in text_lower)

    business_signals = sum(1 for w in ["meeting", "schedule", "project", "deadline",
                                        "report", "quarterly", "review", "team",
                                        "agenda", "budget", "proposal", "strategy",
                                        "attached", "follow up", "regarding"]
                           if w in text_lower)

    personal_signals = sum(1 for w in ["love", "miss", "family", "friend", "birthday",
                                        "dinner", "weekend", "vacation", "holiday",
                                        "thank", "sorry", "hope", "feel", "happy",
                                        "sad", "excited", "worried"]
                           if w in text_lower)

    # Determine primary intent
    scores = {
        "spam_check": spam_signals * 2.0,
        "business_analysis": business_signals * 1.5,
        "personal_message": personal_signals * 1.5,
    }
    state.intent = max(scores, key=scores.get) if max(scores.values()) > 0 else "general_analysis"

    state.route_taken.append("classify_intent")
    state.processing_times["classify_intent"] = round((time.perf_counter() - start) * 1000, 2)
    return state


def extract_features_node(state: AgentState) -> AgentState:
    """
    NODE 2: Feature Extraction
    Extracts NLP features from the text for downstream analysis.

    LangGraph equivalent:
        graph.add_node("extract_features", extract_features_node)
    """
    start = time.perf_counter()
    text = state.text
    text_lower = text.lower()
    words = text_lower.split()

    # Character-level features
    char_count = len(text)
    upper_count = sum(1 for c in text if c.isupper())
    digit_count = sum(1 for c in text if c.isdigit())
    special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())

    # Word-level features
    word_count = len(words)
    unique_words = len(set(words))
    avg_word_length = sum(len(w) for w in words) / max(word_count, 1)

    # Punctuation features
    excl_count = text.count("!")
    question_count = text.count("?")
    ellipsis_count = text.count("...")

    # URL and entity detection
    url_pattern = r'https?://\S+|www\.\S+|\S+\.com\S*'
    urls_found = re.findall(url_pattern, text_lower)
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails_found = re.findall(email_pattern, text)
    money_pattern = r'\$[\d,]+\.?\d*|\d+\s*(?:dollars|usd|eur|gbp)'
    money_refs = re.findall(money_pattern, text_lower)

    state.text_features = {
        "char_count": char_count,
        "word_count": word_count,
        "unique_word_count": unique_words,
        "vocabulary_richness": round(unique_words / max(word_count, 1), 3),
        "avg_word_length": round(avg_word_length, 2),
        "caps_ratio": round(upper_count / max(char_count, 1), 3),
        "digit_ratio": round(digit_count / max(char_count, 1), 3),
        "special_char_ratio": round(special_count / max(char_count, 1), 3),
        "exclamation_count": excl_count,
        "question_count": question_count,
        "ellipsis_count": ellipsis_count,
        "urls_found": urls_found,
        "url_count": len(urls_found),
        "emails_found": emails_found,
        "email_count": len(emails_found),
        "money_references": money_refs,
        "money_ref_count": len(money_refs),
        "has_all_caps_words": any(w.isupper() and len(w) > 2 for w in text.split()),
    }

    state.route_taken.append("extract_features")
    state.processing_times["extract_features"] = round((time.perf_counter() - start) * 1000, 2)
    return state


def spam_analysis_node(state: AgentState) -> AgentState:
    """
    NODE 3a: Spam Classification (conditional — only runs if intent is spam_check)

    LangGraph equivalent:
        graph.add_conditional_edges("classify_intent", route_by_intent,
            {"spam_check": "spam_analysis", ...})
    """
    start = time.perf_counter()
    text_lower = state.text.lower()
    features = state.text_features

    # Weighted spam scoring model
    spam_score = 0.0
    reasons = []

    # Spam keyword analysis
    spam_keywords = {
        "high": (["free", "winner", "congratulations", "claim", "prize",
                  "million", "100%", "guaranteed"], 0.15),
        "medium": (["click", "subscribe", "offer", "discount", "limited time",
                    "act now", "urgent", "buy now", "apply now"], 0.10),
        "low": (["deal", "save", "exclusive", "special", "bonus"], 0.05),
    }

    for severity, (keywords, weight) in spam_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                spam_score += weight
                reasons.append(f"{severity}_keyword:{kw}")

    # Feature-based scoring
    if features["caps_ratio"] > 0.3:
        spam_score += 0.15
        reasons.append("excessive_capitalization")
    if features["exclamation_count"] > 2:
        spam_score += 0.1
        reasons.append("excessive_exclamation")
    if features["url_count"] > 0 and features["word_count"] < 15:
        spam_score += 0.12
        reasons.append("short_message_with_urls")
    if features["money_ref_count"] > 0:
        spam_score += 0.08
        reasons.append("money_references")
    if features["vocabulary_richness"] < 0.5 and features["word_count"] > 5:
        spam_score += 0.05
        reasons.append("low_vocabulary_richness")

    # Normalize via sigmoid
    spam_probability = 1 / (1 + math.exp(-3 * (spam_score - 0.3)))
    spam_probability = round(spam_probability, 4)

    prediction = "spam" if spam_probability >= state.threshold else "ham"

    state.spam_analysis = {
        "prediction": prediction,
        "spam_probability": spam_probability,
        "confidence": round(spam_probability if prediction == "spam" else 1 - spam_probability, 4),
        "raw_score": round(spam_score, 4),
        "threshold_used": state.threshold,
        "risk_factors": reasons,
        "risk_factor_count": len(reasons),
    }

    state.route_taken.append("spam_analysis")
    state.processing_times["spam_analysis"] = round((time.perf_counter() - start) * 1000, 2)
    return state


def sentiment_analysis_node(state: AgentState) -> AgentState:
    """
    NODE 3b: Sentiment Analysis (runs for business/personal/general intents)

    LangGraph equivalent:
        graph.add_node("sentiment_analysis", sentiment_analysis_node)
    """
    start = time.perf_counter()
    text_lower = state.text.lower()

    positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic",
                      "happy", "pleased", "thank", "appreciate", "love", "enjoy",
                      "success", "achievement", "congratulate", "outstanding", "brilliant",
                      "delighted", "thrilled", "excited", "perfect", "impressive"]
    negative_words = ["bad", "terrible", "awful", "horrible", "disappointing", "poor",
                      "angry", "frustrated", "worried", "concern", "fail", "problem",
                      "issue", "delay", "unfortunately", "regret", "sorry", "difficult",
                      "struggling", "missed", "wrong", "error", "complaint"]
    intensifiers = ["very", "extremely", "incredibly", "absolutely", "totally", "really"]

    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    intensity = sum(1 for w in intensifiers if w in text_lower)

    # Weighted sentiment score [-1, 1]
    raw_score = (pos_count - neg_count) / max(pos_count + neg_count, 1)
    # Intensifiers amplify the sentiment
    if intensity > 0:
        raw_score *= (1 + intensity * 0.15)
    raw_score = max(-1, min(1, raw_score))

    if raw_score > 0.2:
        label = "positive"
    elif raw_score < -0.2:
        label = "negative"
    else:
        label = "neutral"

    state.sentiment_analysis = {
        "sentiment": label,
        "score": round(raw_score, 3),
        "positive_signals": pos_count,
        "negative_signals": neg_count,
        "intensity_modifiers": intensity,
        "confidence": round(abs(raw_score), 3),
    }

    state.route_taken.append("sentiment_analysis")
    state.processing_times["sentiment_analysis"] = round((time.perf_counter() - start) * 1000, 2)
    return state


def entity_extraction_node(state: AgentState) -> AgentState:
    """
    NODE 3c: Named Entity Extraction (runs for business intents)

    LangGraph equivalent:
        graph.add_node("entity_extraction", entity_extraction_node)
    """
    start = time.perf_counter()
    text = state.text

    # Simple pattern-based NER (in production: spaCy or LLM-based)
    date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:,? \d{4})?)\b'
    time_pattern = r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b|\b(?:noon|midnight)\b'
    percentage_pattern = r'\d+(?:\.\d+)?%'

    # Capitalized multi-word phrases (potential names/orgs)
    name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'

    entities = {
        "dates": re.findall(date_pattern, text),
        "times": re.findall(time_pattern, text),
        "percentages": re.findall(percentage_pattern, text),
        "potential_names": re.findall(name_pattern, text),
        "urls": state.text_features.get("urls_found", []),
        "emails": state.text_features.get("emails_found", []),
        "money": state.text_features.get("money_references", []),
    }

    total_entities = sum(len(v) for v in entities.values())

    state.entity_analysis = {
        "entities": entities,
        "total_entities_found": total_entities,
        "entity_density": round(total_entities / max(state.text_features.get("word_count", 1), 1), 3),
    }

    state.route_taken.append("entity_extraction")
    state.processing_times["entity_extraction"] = round((time.perf_counter() - start) * 1000, 2)
    return state


def risk_assessment_node(state: AgentState) -> AgentState:
    """
    NODE 4: Risk Assessment (aggregation node — always runs)
    Combines outputs from previous nodes into a risk profile.

    LangGraph equivalent:
        graph.add_node("risk_assessment", risk_assessment_node)
    """
    start = time.perf_counter()

    risk_score = 0.0
    risk_flags = []

    # From spam analysis
    if state.spam_analysis:
        if state.spam_analysis.get("prediction") == "spam":
            risk_score += 0.4
            risk_flags.append("classified_as_spam")
        risk_score += state.spam_analysis.get("spam_probability", 0) * 0.2

    # From sentiment
    if state.sentiment_analysis:
        if state.sentiment_analysis.get("sentiment") == "negative":
            risk_score += 0.1
            risk_flags.append("negative_sentiment")

    # From features
    features = state.text_features
    if features.get("url_count", 0) > 2:
        risk_score += 0.1
        risk_flags.append("multiple_urls")
    if features.get("caps_ratio", 0) > 0.4:
        risk_score += 0.05
        risk_flags.append("high_capitalization")

    risk_level = "low" if risk_score < 0.3 else "medium" if risk_score < 0.6 else "high"

    state.risk_assessment = {
        "risk_level": risk_level,
        "risk_score": round(min(risk_score, 1.0), 3),
        "risk_flags": risk_flags,
        "recommendation": {
            "low": "Message appears safe. No action needed.",
            "medium": "Exercise caution. Review content before acting on any links or offers.",
            "high": "High risk detected. Do not click links or share personal information.",
        }[risk_level],
    }

    state.route_taken.append("risk_assessment")
    state.processing_times["risk_assessment"] = round((time.perf_counter() - start) * 1000, 2)
    return state


def compile_report_node(state: AgentState) -> AgentState:
    """
    NODE 5: Final Report Compilation (terminal node)
    Assembles all analyses into a structured report.

    LangGraph equivalent:
        graph.add_node("compile_report", compile_report_node)
        graph.add_edge("compile_report", END)
    """
    start = time.perf_counter()

    total_time = sum(state.processing_times.values())

    state.final_report = {
        "input_text": state.text,
        "detected_intent": state.intent,
        "text_features": state.text_features,
        "spam_analysis": state.spam_analysis or None,
        "sentiment_analysis": state.sentiment_analysis or None,
        "entity_analysis": state.entity_analysis or None,
        "risk_assessment": state.risk_assessment,
        "graph_execution": {
            "nodes_executed": state.route_taken + ["compile_report"],
            "total_nodes": len(state.route_taken) + 1,
            "node_timings_ms": state.processing_times,
            "total_processing_time_ms": round(total_time, 2),
        },
        "errors": state.errors if state.errors else None,
        "metadata": {
            "model_version": "1.0.0",
            "engine": "LangGraph Agent (rule-based demo)",
            "author": "Candace Grant — Threshold ML Solutions, LLC",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    state.route_taken.append("compile_report")
    state.processing_times["compile_report"] = round((time.perf_counter() - start) * 1000, 2)
    return state


# ── Conditional Router (LangGraph conditional_edges) ──────────────

def route_by_intent(state: AgentState) -> list:
    """
    Conditional edge function — determines which analysis nodes to run.

    LangGraph equivalent:
        graph.add_conditional_edges(
            "classify_intent",
            route_by_intent,
            {"spam_check": "spam_analysis", "business": "sentiment_analysis", ...}
        )
    """
    if state.intent == "spam_check":
        return ["spam_analysis", "sentiment_analysis"]
    elif state.intent == "business_analysis":
        return ["sentiment_analysis", "entity_extraction"]
    elif state.intent == "personal_message":
        return ["sentiment_analysis"]
    else:  # general_analysis
        return ["spam_analysis", "sentiment_analysis", "entity_extraction"]


# ── Graph Executor ────────────────────────────────────────────────

def run_agent_graph(text: str, threshold: float = 0.5) -> dict:
    """
    Execute the full LangGraph agent pipeline.

    Graph topology:
        START → classify_intent → [conditional routing] → risk_assessment → compile_report → END

    Conditional routes:
        spam_check      → spam_analysis + sentiment_analysis
        business        → sentiment_analysis + entity_extraction
        personal        → sentiment_analysis
        general         → spam_analysis + sentiment_analysis + entity_extraction
    """
    state = AgentState(text=text, threshold=threshold)

    # Node 1: Intent classification
    state = classify_intent_node(state)

    # Node 2: Feature extraction (always runs)
    state = extract_features_node(state)

    # Node 3: Conditional routing
    routes = route_by_intent(state)

    node_map = {
        "spam_analysis": spam_analysis_node,
        "sentiment_analysis": sentiment_analysis_node,
        "entity_extraction": entity_extraction_node,
    }

    for route in routes:
        if route in node_map:
            try:
                state = node_map[route](state)
            except Exception as e:
                state.errors.append(f"{route}: {str(e)}")

    # Node 4: Risk assessment (always runs — aggregation)
    state = risk_assessment_node(state)

    # Node 5: Compile report (terminal node)
    state = compile_report_node(state)

    return state.final_report


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = FastAPI(
    title="Threshold ML — Text Analysis Agent API",
    description=(
        "A multi-step AI agent built with the LangGraph pattern, served via FastAPI. "
        "Features conditional routing, stateful graph execution, NLP feature engineering, "
        "spam classification, sentiment analysis, entity extraction, and risk assessment."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── In-Memory Monitoring ─────────────────────────────────────────
prediction_log: list[dict] = []
start_time = datetime.now(timezone.utc)


# ── Pydantic Request/Response Models ─────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze.")
    threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0, description="Spam classification threshold.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "Congratulations! You've won a free iPhone. Click here!", "threshold": 0.5},
                {"text": "Hi team, the Q3 report is attached for your review. Let's discuss at 3pm.", "threshold": 0.5},
            ]
        }
    }

class BatchAnalyzeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=20, description="Texts to analyze (max 20).")
    threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)

class QuickClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)

class QuickClassifyResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    spam_probability: float
    intent: str
    risk_level: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    total_analyses: int
    model_loaded: bool
    version: str
    graph_nodes: list[str]

class GraphInfoResponse(BaseModel):
    graph_name: str
    description: str
    nodes: list[dict]
    edges: list[dict]
    conditional_routes: dict
    author: str


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    tpl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates", "dashboard.html")
    with open(tpl) as f:
        return HTMLResponse(content=f.read())


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check with graph topology info."""
    uptime = (datetime.now(timezone.utc) - start_time).total_seconds()
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(uptime, 1),
        total_analyses=len(prediction_log),
        model_loaded=True,
        version="1.0.0",
        graph_nodes=[
            "classify_intent", "extract_features", "spam_analysis",
            "sentiment_analysis", "entity_extraction", "risk_assessment", "compile_report"
        ],
    )


@app.get("/graph/info", response_model=GraphInfoResponse, tags=["Graph"])
async def graph_info():
    """
    Get the LangGraph topology — nodes, edges, and conditional routing logic.
    This is the architectural blueprint of the agent.
    """
    return GraphInfoResponse(
        graph_name="TextAnalysisAgent",
        description="Multi-step text analysis agent with conditional routing based on intent classification.",
        nodes=[
            {"name": "classify_intent", "type": "router", "description": "Determines analysis path based on text intent"},
            {"name": "extract_features", "type": "processor", "description": "NLP feature engineering — 18 features extracted"},
            {"name": "spam_analysis", "type": "analyzer", "description": "Weighted spam scoring with sigmoid normalization"},
            {"name": "sentiment_analysis", "type": "analyzer", "description": "Lexicon-based sentiment with intensity modifiers"},
            {"name": "entity_extraction", "type": "analyzer", "description": "Pattern-based NER for dates, names, money, URLs"},
            {"name": "risk_assessment", "type": "aggregator", "description": "Cross-node risk scoring and recommendations"},
            {"name": "compile_report", "type": "terminal", "description": "Assembles final structured report"},
        ],
        edges=[
            {"from": "START", "to": "classify_intent", "type": "direct"},
            {"from": "classify_intent", "to": "extract_features", "type": "direct"},
            {"from": "extract_features", "to": "[conditional]", "type": "conditional"},
            {"from": "[analysis_nodes]", "to": "risk_assessment", "type": "direct"},
            {"from": "risk_assessment", "to": "compile_report", "type": "direct"},
            {"from": "compile_report", "to": "END", "type": "direct"},
        ],
        conditional_routes={
            "spam_check": ["spam_analysis", "sentiment_analysis"],
            "business_analysis": ["sentiment_analysis", "entity_extraction"],
            "personal_message": ["sentiment_analysis"],
            "general_analysis": ["spam_analysis", "sentiment_analysis", "entity_extraction"],
        },
        author="Candace Grant — Threshold ML Solutions, LLC",
    )


@app.post("/analyze", tags=["Analysis"])
async def analyze_text(request: AnalyzeRequest):
    """
    Full agent analysis — runs the complete LangGraph pipeline.
    Returns detailed results from all executed nodes including
    graph execution trace.
    """
    report = run_agent_graph(request.text, request.threshold)

    prediction_log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "intent": report.get("detected_intent", ""),
        "risk_level": report.get("risk_assessment", {}).get("risk_level", ""),
        "spam_prediction": report.get("spam_analysis", {}).get("prediction", "n/a") if report.get("spam_analysis") else "n/a",
        "sentiment": report.get("sentiment_analysis", {}).get("sentiment", "n/a") if report.get("sentiment_analysis") else "n/a",
        "nodes_executed": len(report.get("graph_execution", {}).get("nodes_executed", [])),
        "total_time_ms": report.get("graph_execution", {}).get("total_processing_time_ms", 0),
    })

    return report


@app.post("/classify", response_model=QuickClassifyResponse, tags=["Analysis"])
async def quick_classify(request: QuickClassifyRequest):
    """
    Quick classification — runs full pipeline but returns a simplified response.
    Ideal for high-throughput integrations.
    """
    report = run_agent_graph(request.text, request.threshold)

    spam = report.get("spam_analysis") or {}
    risk = report.get("risk_assessment") or {}

    prediction_log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "intent": report.get("detected_intent", ""),
        "risk_level": risk.get("risk_level", ""),
        "spam_prediction": spam.get("prediction", "ham"),
        "sentiment": report.get("sentiment_analysis", {}).get("sentiment", "n/a") if report.get("sentiment_analysis") else "n/a",
        "nodes_executed": len(report.get("graph_execution", {}).get("nodes_executed", [])),
        "total_time_ms": report.get("graph_execution", {}).get("total_processing_time_ms", 0),
    })

    return QuickClassifyResponse(
        text=request.text,
        prediction=spam.get("prediction", "ham"),
        confidence=spam.get("confidence", 0.0),
        spam_probability=spam.get("spam_probability", 0.0),
        intent=report.get("detected_intent", "general"),
        risk_level=risk.get("risk_level", "low"),
        processing_time_ms=report.get("graph_execution", {}).get("total_processing_time_ms", 0),
    )


@app.post("/analyze/batch", tags=["Analysis"])
async def analyze_batch(request: BatchAnalyzeRequest):
    """Batch analysis — run the full agent on up to 20 messages."""
    results = []
    for text in request.texts:
        report = run_agent_graph(text, request.threshold)
        results.append(report)

        prediction_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "intent": report.get("detected_intent", ""),
            "risk_level": report.get("risk_assessment", {}).get("risk_level", ""),
            "spam_prediction": report.get("spam_analysis", {}).get("prediction", "n/a") if report.get("spam_analysis") else "n/a",
            "sentiment": report.get("sentiment_analysis", {}).get("sentiment", "n/a") if report.get("sentiment_analysis") else "n/a",
            "nodes_executed": len(report.get("graph_execution", {}).get("nodes_executed", [])),
            "total_time_ms": report.get("graph_execution", {}).get("total_processing_time_ms", 0),
        })

    return {
        "results": results,
        "total_processed": len(results),
        "summary": {
            "intents": {r["detected_intent"]: sum(1 for x in results if x["detected_intent"] == r["detected_intent"]) for r in results},
            "risk_levels": {r["risk_assessment"]["risk_level"]: sum(1 for x in results if x["risk_assessment"]["risk_level"] == r["risk_assessment"]["risk_level"]) for r in results},
        },
    }


@app.post("/analyze/stream", tags=["Analysis"])
async def analyze_stream(request: AnalyzeRequest):
    """
    Streaming analysis — streams each graph node's output as it completes.
    Demonstrates FastAPI StreamingResponse for real-time agent transparency.
    """
    async def generate():
        state = AgentState(text=request.text, threshold=request.threshold)

        steps = [
            ("classify_intent", classify_intent_node),
            ("extract_features", extract_features_node),
        ]

        # Run initial nodes
        for name, fn in steps:
            state = fn(state)
            yield json.dumps({"node": name, "status": "complete", "time_ms": state.processing_times[name]}) + "\n"
            await asyncio.sleep(0.05)

        # Conditional routing
        routes = route_by_intent(state)
        yield json.dumps({"node": "router", "status": "routing", "routes": routes, "intent": state.intent}) + "\n"
        await asyncio.sleep(0.05)

        node_map = {
            "spam_analysis": spam_analysis_node,
            "sentiment_analysis": sentiment_analysis_node,
            "entity_extraction": entity_extraction_node,
        }

        for route in routes:
            if route in node_map:
                state = node_map[route](state)
                yield json.dumps({"node": route, "status": "complete", "time_ms": state.processing_times[route]}) + "\n"
                await asyncio.sleep(0.05)

        # Aggregation + final
        state = risk_assessment_node(state)
        yield json.dumps({"node": "risk_assessment", "status": "complete", "time_ms": state.processing_times["risk_assessment"]}) + "\n"
        await asyncio.sleep(0.05)

        state = compile_report_node(state)
        yield json.dumps({"node": "compile_report", "status": "complete", "report": state.final_report}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/history", tags=["Monitoring"])
async def analysis_history(limit: int = 50):
    """Recent analysis history."""
    return {
        "total_analyses": len(prediction_log),
        "recent": prediction_log[-limit:][::-1],
    }


@app.get("/stats", tags=["Monitoring"])
async def analysis_stats():
    """Aggregated analysis statistics."""
    if not prediction_log:
        return {"total_analyses": 0}

    intents = {}
    risk_levels = {}
    spam_count = 0
    total_time = 0.0

    for p in prediction_log:
        intent = p.get("intent", "unknown")
        intents[intent] = intents.get(intent, 0) + 1
        rl = p.get("risk_level", "unknown")
        risk_levels[rl] = risk_levels.get(rl, 0) + 1
        if p.get("spam_prediction") == "spam":
            spam_count += 1
        total_time += p.get("total_time_ms", 0)

    return {
        "total_analyses": len(prediction_log),
        "spam_detected": spam_count,
        "spam_rate": round(spam_count / len(prediction_log), 4),
        "intent_distribution": intents,
        "risk_distribution": risk_levels,
        "avg_processing_time_ms": round(total_time / len(prediction_log), 2),
    }