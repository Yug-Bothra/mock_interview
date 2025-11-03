"""
Vercel-Compatible Interview Assistant Backend
File: api/main.py
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from mangum import Mangum
import os
import json
import time

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

# Initialize FastAPI
app = FastAPI(title="Interview Assistant API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELS
# ============================================================================

class TranscriptRequest(BaseModel):
    audio_base64: str
    language: str = "en"
    stream_type: str = "candidate"

class QuestionProcessRequest(BaseModel):
    transcript: str
    settings: Dict[str, Any] = {}
    persona: Optional[Dict[str, Any]] = None
    custom_style_prompt: Optional[str] = None

# ============================================================================
# CONSTANTS
# ============================================================================

RESPONSE_STYLES = {
    "concise": {
        "name": "Concise Professional",
        "prompt": "You are a concise interview assistant. Provide brief, professional answers in 2-3 sentences."
    },
    "detailed": {
        "name": "Detailed Professional",
        "prompt": "You are a detailed interview assistant. Provide comprehensive answers with clear explanations and examples."
    },
    "storytelling": {
        "name": "Storytelling",
        "prompt": "You are an engaging interview assistant. Use STAR format when appropriate."
    },
    "technical": {
        "name": "Technical Expert",
        "prompt": "You are a technical interview expert. Provide in-depth technical answers with examples."
    }
}

QUESTION_DETECTION_PROMPT = """You are an intelligent interview assistant.

Your task:
1. Analyze the transcript
2. Extract the EXACT question (keep original wording)
3. If question detected, return:
   QUESTION: [extracted question]
   ANSWER: [your answer]
4. If casual conversation, respond: "SKIP"

Guidelines:
- DO NOT rephrase the question
- Keep ALL technical terms
- Remove only conversational preamble
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def transcribe_audio_deepgram(audio_base64: str, language: str = "en") -> Dict[str, Any]:
    """Transcribe using Deepgram REST API"""
    if not DEEPGRAM_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPGRAM_API_KEY not configured")
    
    import base64
    import httpx
    
    try:
        audio_bytes = base64.b64decode(audio_base64)
        
        url = "https://api.deepgram.com/v1/listen"
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav"
        }
        params = {
            "model": "nova-2",
            "language": language,
            "punctuate": "true",
            "smart_format": "true"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, params=params, content=audio_bytes)
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Deepgram error")
            
            result = response.json()
            transcript = ""
            confidence = 0.0
            
            if result.get("results", {}).get("channels", []):
                alternatives = result["results"]["channels"][0].get("alternatives", [])
                if alternatives:
                    transcript = alternatives[0].get("transcript", "")
                    confidence = alternatives[0].get("confidence", 0.0)
            
            return {"transcript": transcript, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_question_with_ai(
    transcript: str,
    settings: Dict[str, Any],
    persona_data: Optional[Dict] = None,
    custom_style_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Process question with OpenAI"""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    if not transcript or len(transcript.strip()) < 10:
        return {"has_question": False, "question": None, "answer": None}
    
    start_time = time.time()
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Build prompt
        response_style_id = settings.get("selectedResponseStyleId", "concise")
        if custom_style_prompt:
            style_prompt = custom_style_prompt
        else:
            style_config = RESPONSE_STYLES.get(response_style_id, RESPONSE_STYLES["concise"])
            style_prompt = style_config["prompt"]
        
        system_prompt = QUESTION_DETECTION_PROMPT + "\n\n" + style_prompt
        
        # Add persona context
        if persona_data:
            system_prompt += f"""

CANDIDATE CONTEXT:
- Position: {persona_data.get('position', 'N/A')}
- Company: {persona_data.get('company_name', 'N/A')}
"""
            if persona_data.get('company_description'):
                system_prompt += f"- Company Description: {persona_data.get('company_description')}\n"
            if persona_data.get('job_description'):
                system_prompt += f"- Job Description: {persona_data.get('job_description')}\n"
            if persona_data.get('resume_text'):
                system_prompt += f"\nRESUME:\n{persona_data.get('resume_text')}\n"
        
        # Add language preference
        prog_lang = settings.get("programmingLanguage", "Python")
        system_prompt += f"\n\nUse {prog_lang} for code examples."
        
        # Add custom instructions
        if settings.get("interviewInstructions"):
            system_prompt += f"\n\nINSTRUCTIONS:\n{settings['interviewInstructions']}"
        
        model = settings.get("defaultModel", "gpt-4o-mini")
        
        # Call OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transcript: {transcript}"}
            ],
            temperature=0.5,
            max_tokens=400,
            timeout=20
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Check for skip
        if "SKIP" in result_text.upper():
            return {
                "has_question": False,
                "question": None,
                "answer": None,
                "processing_time": time.time() - start_time
            }
        
        # Parse response
        question = None
        answer = None
        
        if "QUESTION:" in result_text and "ANSWER:" in result_text:
            parts = result_text.split("ANSWER:", 1)
            question = parts[0].replace("QUESTION:", "").strip()
            answer = parts[1].strip() if len(parts) > 1 else ""
        else:
            question = transcript
            answer = result_text
        
        return {
            "has_question": True,
            "question": question,
            "answer": answer,
            "processing_time": time.time() - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ROUTES
# ============================================================================

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Interview Assistant API",
        "version": "2.0.0",
        "platform": "Vercel"
    }

@app.get("/api")
async def api_root():
    return {
        "status": "running",
        "endpoints": {
            "health": "/health",
            "transcribe": "POST /api/transcribe",
            "process": "POST /api/process-question",
            "combined": "POST /api/transcribe-and-answer"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "openai": bool(OPENAI_API_KEY),
        "deepgram": bool(DEEPGRAM_API_KEY),
        "timestamp": time.time()
    }

@app.post("/api/transcribe")
async def transcribe(request: TranscriptRequest):
    """Transcribe audio"""
    try:
        result = await transcribe_audio_deepgram(request.audio_base64, request.language)
        return {
            "transcript": result["transcript"],
            "confidence": result["confidence"],
            "stream_type": request.stream_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-question")
async def process_question(request: QuestionProcessRequest):
    """Process question and generate answer"""
    try:
        result = await process_question_with_ai(
            request.transcript,
            request.settings,
            request.persona,
            request.custom_style_prompt
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcribe-and-answer")
async def transcribe_and_answer(request: Request):
    """Combined: transcribe + answer"""
    try:
        data = await request.json()
        
        # Transcribe
        audio_base64 = data.get("audio_base64")
        if not audio_base64:
            raise HTTPException(status_code=400, detail="audio_base64 required")
        
        transcription = await transcribe_audio_deepgram(
            audio_base64,
            data.get("language", "en")
        )
        
        transcript = transcription["transcript"]
        if not transcript:
            return {
                "success": False,
                "message": "No transcript",
                "transcript": "",
                "answer": None
            }
        
        # Process
        result = await process_question_with_ai(
            transcript,
            data.get("settings", {}),
            data.get("persona")
        )
        
        return {
            "success": True,
            "transcript": transcript,
            "confidence": transcription["confidence"],
            "has_question": result["has_question"],
            "question": result.get("question"),
            "answer": result.get("answer"),
            "processing_time": result.get("processing_time", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/status")
async def models_status():
    return {
        "default_provider": "gpt-4o-mini",
        "available_providers": {"gpt-4o-mini": True, "gpt-4o": True},
        "configured": bool(OPENAI_API_KEY)
    }

@app.get("/api/response-styles")
async def response_styles():
    return {
        "styles": {
            sid: {"name": cfg["name"], "description": cfg["prompt"][:100]}
            for sid, cfg in RESPONSE_STYLES.items()
        },
        "default": "concise"
    }

# ============================================================================
# VERCEL HANDLER - MUST BE AT MODULE LEVEL
# ============================================================================

handler = Mangum(app, lifespan="off")