"""
Vercel-Compatible Interview Assistant Backend
✅ FastAPI REST API - No WebSockets
✅ All functionality converted to POST endpoints
✅ Vercel Serverless compatible
"""

import os
import json
import time
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
from mangum import Mangum

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

app = FastAPI(title="Interview Assistant API - Vercel REST", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_MODEL = "gpt-4o-mini"

# ============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================================

class TranscriptRequest(BaseModel):
    """Request body for transcription endpoint"""
    audio_base64: str
    language: str = "en"
    stream_type: str = "candidate"  # "candidate" or "interviewer"

class TranscriptResponse(BaseModel):
    """Response from transcription endpoint"""
    transcript: str
    confidence: float = 0.0
    stream_type: str

class QuestionProcessRequest(BaseModel):
    """Request body for question processing"""
    transcript: str
    settings: Dict[str, Any] = {}
    persona: Optional[Dict[str, Any]] = None
    custom_style_prompt: Optional[str] = None

class QuestionProcessResponse(BaseModel):
    """Response from question processing"""
    has_question: bool
    question: Optional[str] = None
    answer: Optional[str] = None
    processing_time: float = 0.0

class BatchTranscriptRequest(BaseModel):
    """Process multiple transcripts at once"""
    transcripts: List[str]
    settings: Dict[str, Any] = {}
    persona: Optional[Dict[str, Any]] = None

# ============================================================================
# RESPONSE STYLES & PROMPTS
# ============================================================================

RESPONSE_STYLES = {
    "concise": {
        "name": "Concise Professional",
        "prompt": """You are a concise interview assistant. Provide brief, professional answers in 2-3 sentences.
Focus on the core information without elaboration. Be direct and efficient."""
    },
    "detailed": {
        "name": "Detailed Professional",
        "prompt": """You are a detailed interview assistant. Provide comprehensive answers with:
- Clear explanation of the concept
- Relevant examples from experience
- Practical insights
Keep responses around 150 words, professional and well-structured."""
    },
    "storytelling": {
        "name": "Storytelling",
        "prompt": """You are an engaging interview assistant using storytelling techniques.
Structure answers using STAR format when appropriate:
- Situation: Set the context
- Task: Describe the challenge
- Action: Explain what you did
- Result: Share the outcome
Make responses compelling and memorable while remaining professional."""
    },
    "technical": {
        "name": "Technical Expert",
        "prompt": """You are a technical interview expert. Provide in-depth technical answers:
- Explain concepts clearly with proper terminology
- Include code examples when relevant
- Discuss trade-offs and best practices
Be thorough but avoid unnecessary jargon."""
    }
}

QUESTION_DETECTION_PROMPT = """You are an intelligent interview assistant that processes conversation transcripts.

Your task:
1. Analyze the incoming transcript text
2. Extract the EXACT question being asked (keep original wording)
3. If a question is detected, return it in this EXACT format:
   QUESTION: [extracted question]
   ANSWER: [your answer]
4. If it's casual conversation or greetings, respond with: "SKIP"

Guidelines:
- DO NOT rephrase the question - extract it EXACTLY as asked
- Remove only conversational preamble
- Keep ALL technical terms and original phrasing
"""

# ============================================================================
# DEEPGRAM TRANSCRIPTION (REST API)
# ============================================================================

async def transcribe_audio_deepgram(audio_base64: str, language: str = "en") -> Dict[str, Any]:
    """
    Transcribe audio using Deepgram REST API
    Docs: https://developers.deepgram.com/docs/getting-started-with-pre-recorded-audio
    """
    if not DEEPGRAM_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPGRAM_API_KEY not configured")
    
    import base64
    import httpx
    
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_base64)
        
        # Deepgram REST API endpoint
        url = "https://api.deepgram.com/v1/listen"
        
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav"  # or audio/mp3, audio/webm, etc.
        }
        
        params = {
            "model": "nova-2",
            "language": language,
            "punctuate": "true",
            "smart_format": "true",
            "filler_words": "false"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers=headers,
                params=params,
                content=audio_bytes
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Deepgram API error: {response.text}"
                )
            
            result = response.json()
            
            # Extract transcript
            transcript = ""
            confidence = 0.0
            
            if result.get("results", {}).get("channels", []):
                alternatives = result["results"]["channels"][0].get("alternatives", [])
                if alternatives:
                    transcript = alternatives[0].get("transcript", "")
                    confidence = alternatives[0].get("confidence", 0.0)
            
            return {
                "transcript": transcript,
                "confidence": confidence,
                "full_response": result
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

# ============================================================================
# AI QUESTION PROCESSING
# ============================================================================

async def process_question_with_ai(
    transcript: str,
    settings: Dict[str, Any],
    persona_data: Optional[Dict] = None,
    custom_style_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Process transcript and generate answer using OpenAI"""
    
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    if not transcript or len(transcript.strip()) < 10:
        return {
            "has_question": False,
            "question": None,
            "answer": None,
            "message": "Transcript too short"
        }
    
    start_time = time.time()
    
    try:
        # Build system prompt
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
                system_prompt += f"\nCANDIDATE RESUME:\n{persona_data.get('resume_text')}\n"
        
        # Add programming language preference
        prog_lang = settings.get("programmingLanguage", "Python")
        system_prompt += f"\n\nWhen providing code examples, use {prog_lang}."
        
        # Add custom instructions
        if settings.get("interviewInstructions"):
            system_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{settings['interviewInstructions']}"
        
        model = settings.get("defaultModel", DEFAULT_MODEL)
        
        # Call OpenAI
        response = openai.chat.completions.create(
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
        
        # Check if it's a skip
        if result_text.upper() == "SKIP" or "SKIP" in result_text.upper():
            return {
                "has_question": False,
                "question": None,
                "answer": None,
                "message": "Not a question",
                "processing_time": time.time() - start_time
            }
        
        # Parse question and answer
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
        raise HTTPException(status_code=500, detail=f"AI processing error: {str(e)}")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "status": "running",
        "service": "Interview Assistant API - REST",
        "platform": "Vercel Serverless",
        "version": "2.0.0",
        "architecture": "REST API (No WebSockets)",
        "endpoints": {
            "health": "/health",
            "transcribe": "POST /api/transcribe",
            "process_question": "POST /api/process-question",
            "batch_process": "POST /api/batch-process",
            "models": "/api/models/status",
            "styles": "/api/response-styles"
        },
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "platform": "Vercel Serverless",
        "timestamp": time.time(),
        "services": {
            "openai": "configured" if OPENAI_API_KEY else "missing",
            "deepgram": "configured" if DEEPGRAM_API_KEY else "missing"
        },
        "architecture": "REST API",
        "websockets": "not_supported"
    }

@app.post("/api/transcribe")
async def transcribe_audio(request: TranscriptRequest):
    """
    Transcribe audio using Deepgram REST API
    
    Body:
    {
        "audio_base64": "base64_encoded_audio_data",
        "language": "en",
        "stream_type": "candidate"
    }
    """
    try:
        result = await transcribe_audio_deepgram(
            request.audio_base64,
            request.language
        )
        
        return TranscriptResponse(
            transcript=result["transcript"],
            confidence=result["confidence"],
            stream_type=request.stream_type
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-question")
async def process_question(request: QuestionProcessRequest):
    """
    Process interview question and generate answer
    
    Body:
    {
        "transcript": "What is your experience with Python?",
        "settings": {
            "selectedResponseStyleId": "concise",
            "programmingLanguage": "Python",
            "defaultModel": "gpt-4o-mini"
        },
        "persona": {
            "position": "Software Engineer",
            "company_name": "Tech Corp",
            "resume_text": "..."
        }
    }
    """
    try:
        result = await process_question_with_ai(
            request.transcript,
            request.settings,
            request.persona,
            request.custom_style_prompt
        )
        
        return QuestionProcessResponse(**result)
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch-process")
async def batch_process_questions(request: BatchTranscriptRequest):
    """
    Process multiple questions at once
    
    Body:
    {
        "transcripts": [
            "What is your experience?",
            "Tell me about a challenge you faced"
        ],
        "settings": {...},
        "persona": {...}
    }
    """
    try:
        results = []
        
        for transcript in request.transcripts:
            result = await process_question_with_ai(
                transcript,
                request.settings,
                request.persona
            )
            results.append(result)
        
        return {
            "total": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcribe-and-answer")
async def transcribe_and_answer(
    audio_base64: str,
    language: str = "en",
    settings: Dict[str, Any] = {},
    persona: Optional[Dict[str, Any]] = None
):
    """
    One-shot endpoint: Transcribe audio AND generate answer
    
    Body:
    {
        "audio_base64": "...",
        "language": "en",
        "settings": {...},
        "persona": {...}
    }
    """
    try:
        # Step 1: Transcribe
        transcription = await transcribe_audio_deepgram(audio_base64, language)
        transcript = transcription["transcript"]
        
        if not transcript:
            return {
                "success": False,
                "message": "No transcript generated",
                "transcript": "",
                "answer": None
            }
        
        # Step 2: Process question
        result = await process_question_with_ai(
            transcript,
            settings,
            persona
        )
        
        return {
            "success": True,
            "transcript": transcript,
            "confidence": transcription["confidence"],
            "has_question": result["has_question"],
            "question": result.get("question"),
            "answer": result.get("answer"),
            "processing_time": result.get("processing_time")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/status")
async def get_model_status():
    """Get available AI models"""
    return {
        "default_provider": DEFAULT_MODEL,
        "available_providers": {
            "gpt-4o-mini": True,
            "gpt-4o": True
        },
        "configured": OPENAI_API_KEY is not None
    }

@app.get("/api/response-styles")
async def get_response_styles():
    """Get available response styles"""
    return {
        "styles": {
            style_id: {
                "name": config["name"],
                "description": config["prompt"][:100] + "..."
            }
            for style_id, config in RESPONSE_STYLES.items()
        },
        "default": "concise"
    }

# ============================================================================
# VERCEL HANDLER
# ============================================================================

handler = Mangum(app)