"""
Vercel-Compatible Interview Assistant Backend - COMPLETE VERSION
✅ Real-time transcription with proper WAV format
✅ Automatic Q&A generation
✅ Resume processing with AI summaries
✅ Optimized for Vercel serverless
"""

import os
import sys
import json
import time
import io
import base64
from typing import Optional, Dict, Any, List

# Print Python version for debugging
print(f"Python version: {sys.version}", flush=True)

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    print("✅ FastAPI imports successful", flush=True)
except ImportError as e:
    print(f"❌ FastAPI import error: {e}", flush=True)
    raise

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ dotenv loaded", flush=True)
except ImportError:
    print("⚠️ python-dotenv not available, using environment variables directly", flush=True)

try:
    import openai
    print("✅ OpenAI imported", flush=True)
except ImportError as e:
    print(f"❌ OpenAI import error: {e}", flush=True)
    raise

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

print(f"🔑 OPENAI_API_KEY: {'✅ Set' if OPENAI_API_KEY else '❌ Missing'}", flush=True)
print(f"🔑 DEEPGRAM_API_KEY: {'✅ Set' if DEEPGRAM_API_KEY else '❌ Missing'}", flush=True)
print(f"🔑 SUPABASE_URL: {'✅ Set' if SUPABASE_URL else '❌ Missing'}", flush=True)

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Lazy load Supabase
supabase_client = None

def get_supabase_client():
    global supabase_client
    if supabase_client is None and SUPABASE_URL and SUPABASE_KEY:
        try:
            from supabase import create_client
            supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            print("✅ Supabase client initialized", flush=True)
        except Exception as e:
            print(f"❌ Supabase initialization error: {e}", flush=True)
    return supabase_client

# FastAPI app
app = FastAPI(
    title="Interview Assistant API",
    version="3.0.0",
    description="Vercel-compatible interview assistant with real-time features"
)

print("✅ FastAPI app created", flush=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("✅ CORS middleware added", flush=True)

DEFAULT_MODEL = "gpt-4o-mini"

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class TranscriptRequest(BaseModel):
    audio_base64: str
    language: str = "en"
    stream_type: str = "candidate"

class TranscriptResponse(BaseModel):
    transcript: str
    confidence: float = 0.0
    stream_type: str
    is_final: bool = True

class QuestionProcessRequest(BaseModel):
    transcript: str
    settings: Dict[str, Any] = {}
    persona: Optional[Dict[str, Any]] = None
    custom_style_prompt: Optional[str] = None

class QuestionProcessResponse(BaseModel):
    has_question: bool
    question: Optional[str] = None
    answer: Optional[str] = None
    processing_time: float = 0.0

class ProcessResumeRequest(BaseModel):
    persona_id: str

class ProcessResumeResponse(BaseModel):
    success: bool
    message: str
    persona_id: str
    summary: Optional[str] = None
    processing_time: float = 0.0

# ============================================================================
# RESPONSE STYLES
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

QUESTION_DETECTION_PROMPT = """You are an intelligent interview assistant that processes conversation transcripts in real-time.

Your task:
1. Analyze the incoming transcript text
2. Extract the EXACT question being asked (remove ONLY the preamble, but keep the question wording exactly as stated)
3. If a question is detected, return it in this EXACT format:
   QUESTION: [extracted question - keep original wording]
   ANSWER: [your answer]
4. If it's just casual conversation, greetings (like "hi", "hello"), or incomplete thoughts, respond with exactly: "SKIP"

Guidelines for extracting questions:
- Remove conversational preamble ONLY
- DO NOT rephrase the question - extract it EXACTLY as asked
- Keep the question wording completely unchanged
- Extract from the first question word to the question mark
- Preserve ALL technical terms, context, and original phrasing

Response format:
- If question detected: 
  QUESTION: [exact question with original wording]
  ANSWER: [your detailed answer]
- If no question: SKIP

CRITICAL: Do NOT rephrase or rewrite the question. Extract it EXACTLY as spoken.
"""

# ============================================================================
# DEEPGRAM TRANSCRIPTION (REST API)
# ============================================================================

async def transcribe_audio_deepgram(audio_base64: str, language: str = "en") -> Dict[str, Any]:
    """Transcribe audio using Deepgram REST API with proper WAV format"""
    if not DEEPGRAM_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPGRAM_API_KEY not configured")
    
    try:
        import httpx
        print("✅ httpx imported for Deepgram", flush=True)
    except ImportError as e:
        print(f"❌ httpx import error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"httpx not available: {str(e)}")
    
    try:
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(audio_base64)
        
        # Deepgram endpoint
        url = "https://api.deepgram.com/v1/listen"
        
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav"  # WAV format
        }
        
        params = {
            "model": "nova-2",
            "language": language,
            "punctuate": "true",
            "smart_format": "true",
            "filler_words": "false",
            "profanity_filter": "false"
        }
        
        # Make request with longer timeout for larger audio
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers=headers,
                params=params,
                content=audio_bytes
            )
            
            if response.status_code != 200:
                error_detail = response.text
                print(f"❌ Deepgram error: {error_detail}", flush=True)
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Deepgram API error: {error_detail}"
                )
            
            result = response.json()
            
            # Extract transcript and confidence
            transcript = ""
            confidence = 0.0
            
            if result.get("results", {}).get("channels", []):
                alternatives = result["results"]["channels"][0].get("alternatives", [])
                if alternatives:
                    transcript = alternatives[0].get("transcript", "")
                    confidence = alternatives[0].get("confidence", 0.0)
            
            print(f"✅ Transcription successful: {transcript[:50]}... (confidence: {confidence:.2f})", flush=True)
            
            return {
                "transcript": transcript,
                "confidence": confidence,
                "full_response": result
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Transcription error: {e}", flush=True)
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
    
    # Minimum length check
    if not transcript or len(transcript.strip()) < 10:
        return {
            "has_question": False,
            "question": None,
            "answer": None,
            "message": "Transcript too short"
        }
    
    start_time = time.time()
    
    try:
        print(f"🤖 Processing question: {transcript[:100]}...")
        
        # Get response style
        response_style_id = settings.get("selectedResponseStyleId", "concise")
        
        if custom_style_prompt:
            style_prompt = custom_style_prompt
        else:
            style_config = RESPONSE_STYLES.get(response_style_id, RESPONSE_STYLES["concise"])
            style_prompt = style_config["prompt"]
        
        # Build system prompt
        system_prompt = QUESTION_DETECTION_PROMPT + "\n\n" + style_prompt
        
        # Add persona context if available
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
                system_prompt += "\nIMPORTANT: Use the resume information to provide accurate, personalized answers.\n"
        
        # Add programming language preference
        prog_lang = settings.get("programmingLanguage", "Python")
        system_prompt += f"\n\nWhen providing code examples, use {prog_lang}."
        
        # Add custom instructions
        if settings.get("interviewInstructions"):
            system_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{settings['interviewInstructions']}"
        
        # Get model
        model = settings.get("defaultModel", DEFAULT_MODEL)
        
        print(f"🤖 Calling OpenAI with model: {model}")
        
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
        
        print(f"🤖 OpenAI response: {result_text[:200]}...")
        
        # Check if it's a skip
        if result_text.upper() == "SKIP" or "SKIP" in result_text.upper():
            print("⏭️ Skipping - not a question")
            return {
                "has_question": False,
                "question": None,
                "answer": None,
                "message": "Not a question",
                "processing_time": time.time() - start_time
            }
        
        # Extract question and answer
        question = None
        answer = None
        
        if "QUESTION:" in result_text and "ANSWER:" in result_text:
            parts = result_text.split("ANSWER:", 1)
            question = parts[0].replace("QUESTION:", "").strip()
            answer = parts[1].strip() if len(parts) > 1 else ""
            print(f"✅ Extracted Q: {question[:50]}... A: {answer[:50]}...")
        else:
            # Fallback: use full response as answer
            question = transcript
            answer = result_text
            print(f"✅ Using full response - Q: {question[:50]}... A: {answer[:50]}...")
        
        return {
            "has_question": True,
            "question": question,
            "answer": answer,
            "processing_time": time.time() - start_time
        }
        
    except Exception as e:
        print(f"❌ AI processing error: {e}")
        raise HTTPException(status_code=500, detail=f"AI processing error: {str(e)}")

# ============================================================================
# RESUME PROCESSING
# ============================================================================

async def extract_text_from_pdf_url(pdf_url: str) -> str:
    """Download PDF from URL and extract text using PyPDF2"""
    try:
        import requests
        from PyPDF2 import PdfReader
        print("✅ PDF libraries imported", flush=True)
    except ImportError as e:
        print(f"❌ PDF import error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"PDF libraries not available: {str(e)}")
    
    try:
        print(f"📄 Downloading PDF from: {pdf_url[:100]}...", flush=True)
        
        response = requests.get(pdf_url, timeout=15)
        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download PDF: HTTP {response.status_code}"
            )
        
        print(f"✅ PDF downloaded, size: {len(response.content)} bytes", flush=True)
        
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        
        text = ""
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                print(f"✅ Extracted page {page_num}, length: {len(page_text)}", flush=True)
        
        extracted_text = text.strip()
        
        if len(extracted_text) < 50:
            print("❌ Extracted text too short", flush=True)
            return ""
        
        print(f"✅ Total extracted text length: {len(extracted_text)}", flush=True)
        return extracted_text
        
    except Exception as e:
        print(f"❌ PDF extraction error: {e}", flush=True)
        raise HTTPException(
            status_code=500,
            detail=f"PDF extraction error: {str(e)}"
        )

async def generate_resume_summary(text: str) -> str:
    """Use OpenAI to generate a concise resume summary"""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    if not text or len(text.strip()) < 50:
        return "No meaningful content found in resume."
    
    # Limit text length for API
    max_chars = 12000
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    
    prompt = f"""Summarize this resume in 3-4 sentences covering:
1. Current role/most recent position
2. Key technical skills and expertise areas
3. Notable achievements or years of experience
4. Educational background (if mentioned)

Resume text:
{text}"""
    
    try:
        print("🤖 Generating resume summary...")
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
            timeout=15
        )
        
        summary = response.choices[0].message.content.strip()
        print(f"✅ Summary generated: {summary[:100]}...")
        return summary
        
    except Exception as e:
        print(f"❌ OpenAI summarization error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI summarization error: {str(e)}"
        )

# ============================================================================
# API ENDPOINTS - CORE
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "running",
        "service": "Interview Assistant API - Complete Edition",
        "platform": "Vercel",
        "version": "3.0.0",
        "features": [
            "Real-time transcription (WAV)",
            "Automatic Q&A generation",
            "Resume processing with AI",
            "Multiple response styles"
        ],
        "endpoints": {
            "health": "/health",
            "transcribe": "POST /api/transcribe",
            "process_question": "POST /api/process-question",
            "process_resume": "POST /api/process-resume",
            "resume_status": "GET /api/resume-status/{persona_id}",
            "models": "/api/models/status",
            "styles": "/api/response-styles"
        }
    }

@app.get("/health")
async def health_check():
    """Health check"""
    supabase = get_supabase_client()
    return {
        "status": "healthy",
        "platform": "Vercel",
        "timestamp": time.time(),
        "services": {
            "openai": "configured" if OPENAI_API_KEY else "missing",
            "deepgram": "configured" if DEEPGRAM_API_KEY else "missing",
            "supabase": "configured" if supabase else "missing"
        }
    }

# ============================================================================
# API ENDPOINTS - TRANSCRIPTION
# ============================================================================

@app.post("/api/transcribe", response_model=TranscriptResponse)
async def transcribe_audio(request: TranscriptRequest):
    """Transcribe audio using Deepgram REST API"""
    try:
        print(f"📥 Received transcription request for {request.stream_type}")
        
        result = await transcribe_audio_deepgram(
            request.audio_base64,
            request.language
        )
        
        return TranscriptResponse(
            transcript=result["transcript"],
            confidence=result["confidence"],
            stream_type=request.stream_type,
            is_final=True
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"❌ Transcription endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-question", response_model=QuestionProcessResponse)
async def process_question(request: QuestionProcessRequest):
    """Process interview question and generate answer"""
    try:
        print(f"📝 Processing question request")
        
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
        print(f"❌ Question processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# API ENDPOINTS - RESUME PROCESSING
# ============================================================================

@app.post("/api/process-resume", response_model=ProcessResumeResponse)
async def process_single_resume(request: ProcessResumeRequest):
    """Process a single resume: extract text and generate AI summary"""
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    start_time = time.time()
    
    try:
        print(f"📄 Processing resume for persona: {request.persona_id}")
        
        # Get persona data
        response = supabase.table("personas").select(
            "id, user_id, company_name, position, resume_url, resume_text"
        ).eq("id", request.persona_id).single().execute()
        
        persona = response.data
        
        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")
        
        # Check if already processed
        if persona.get("resume_text") and len(str(persona.get("resume_text", "")).strip()) > 50:
            print("✅ Resume already processed")
            return ProcessResumeResponse(
                success=True,
                message="Resume already processed",
                persona_id=request.persona_id,
                summary=persona["resume_text"],
                processing_time=time.time() - start_time
            )
        
        if not persona.get("resume_url"):
            raise HTTPException(
                status_code=400,
                detail="No resume URL found for this persona"
            )
        
        # Extract text from PDF
        text = await extract_text_from_pdf_url(persona["resume_url"])
        
        if not text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF"
            )
        
        # Generate AI summary
        summary = await generate_resume_summary(text)
        
        # Update database
        supabase.table("personas").update({
            "resume_text": summary
        }).eq("id", request.persona_id).execute()
        
        print("✅ Resume processed and saved")
        
        return ProcessResumeResponse(
            success=True,
            message="Resume processed successfully",
            persona_id=request.persona_id,
            summary=summary,
            processing_time=time.time() - start_time
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"❌ Resume processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )

@app.get("/api/resume-status/{persona_id}")
async def get_resume_status(persona_id: str):
    """Check if a resume has been processed"""
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    try:
        response = supabase.table("personas").select(
            "id, resume_url, resume_text"
        ).eq("id", persona_id).single().execute()
        
        persona = response.data
        
        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")
        
        has_resume_url = bool(persona.get("resume_url"))
        has_summary = bool(persona.get("resume_text") and len(str(persona.get("resume_text", "")).strip()) > 50)
        
        return {
            "persona_id": persona_id,
            "has_resume_url": has_resume_url,
            "has_summary": has_summary,
            "needs_processing": has_resume_url and not has_summary,
            "summary_length": len(str(persona.get("resume_text", ""))) if has_summary else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# API ENDPOINTS - CONFIGURATION
# ============================================================================

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

from mangum import Mangum
handler = Mangum(app)