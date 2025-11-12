import React, { useState, useEffect, useRef } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "./Auth/AuthContext";
import {
  Mic,
  Download,
  Settings,
  ArrowLeft,
  Pause,
  Play,
  Volume2,
  X,
  AlertCircle,
  MicOff
} from "lucide-react";
import { jsPDF } from "jspdf";

// ============================================================================
// BACKEND URL CONFIGURATION - VERCEL
// ============================================================================

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 
  (window.location.hostname === 'localhost' 
    ? 'http://localhost:8000' 
    : 'https://mock-interview-ybuf.vercel.app');

console.log('🔗 Backend URL:', BACKEND_URL);

// ============================================================================
// WAV FILE CREATION UTILITY
// ============================================================================

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function createWavFile(pcmData, sampleRate = 16000) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * bitsPerSample / 8;
  const blockAlign = numChannels * bitsPerSample / 8;
  const dataSize = pcmData.length * 2; // 2 bytes per sample for 16-bit
  
  // Create WAV file buffer
  const wavBuffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(wavBuffer);
  
  // Write WAV header
  // "RIFF" chunk descriptor
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true); // File size - 8
  writeString(view, 8, 'WAVE');
  
  // "fmt " sub-chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true); // AudioFormat (1 for PCM)
  view.setUint16(22, numChannels, true); // NumChannels
  view.setUint32(24, sampleRate, true); // SampleRate
  view.setUint32(28, byteRate, true); // ByteRate
  view.setUint16(32, blockAlign, true); // BlockAlign
  view.setUint16(34, bitsPerSample, true); // BitsPerSample
  
  // "data" sub-chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true); // Subchunk2Size
  
  // Write PCM data
  const offset = 44;
  for (let i = 0; i < pcmData.length; i++) {
    view.setInt16(offset + i * 2, pcmData[i], true);
  }
  
  return new Uint8Array(wavBuffer);
}

// ============================================================================
// AUDIO BUFFER MANAGER (Client-side batching for HTTP API)
// ============================================================================

class AudioBufferManager {
  constructor(onFlush, streamType, bufferDuration = 2000) {
    this.buffer = [];
    this.streamType = streamType;
    this.onFlush = onFlush;
    this.bufferDuration = bufferDuration;
    this.timer = null;
    this.isProcessing = false;
    this.isPaused = false;
  }

  add(audioData) {
    if (this.isPaused) return;
    
    this.buffer.push(...audioData);
    
    if (!this.timer) {
      this.timer = setTimeout(() => this.flush(), this.bufferDuration);
    }
  }

  async flush() {
    if (this.isProcessing || this.buffer.length === 0 || this.isPaused) return;
    
    this.isProcessing = true;
    clearTimeout(this.timer);
    this.timer = null;

    const dataToSend = new Int16Array([...this.buffer]);
    this.buffer = [];

    try {
      await this.onFlush(dataToSend, this.streamType);
    } catch (error) {
      console.error('Flush error:', error);
    } finally {
      this.isProcessing = false;
    }
  }

  pause() {
    this.isPaused = true;
  }

  resume() {
    this.isPaused = false;
  }

  clear() {
    clearTimeout(this.timer);
    this.buffer = [];
    this.isProcessing = false;
  }
}

// ============================================================================
// API FUNCTIONS (HTTP POST instead of WebSocket)
// ============================================================================

async function transcribeAudio(pcmData, streamType, language = 'en') {
  try {
    // Convert PCM16 to WAV file with proper headers
    const wavData = createWavFile(pcmData, 16000);
    
    // Convert WAV to base64
    const base64Audio = btoa(String.fromCharCode(...wavData));

    console.log(`📤 Sending ${streamType} audio: ${wavData.length} bytes (${pcmData.length} samples)`);

    const response = await fetch(`${BACKEND_URL}/api/transcribe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        audio_base64: base64Audio,
        language: language,
        stream_type: streamType
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error('❌ Transcription error:', errorData);
      return null;
    }

    const data = await response.json();
    console.log('✅ Transcription received:', data.transcript?.substring(0, 50) + '...');
    return data;

  } catch (error) {
    console.error('❌ Transcription fetch error:', error);
    return null;
  }
}

async function processQuestion(transcript, settings, personaData) {
  try {
    const response = await fetch(`${BACKEND_URL}/api/process-question`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        transcript: transcript,
        settings: settings,
        persona: personaData
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error('❌ Question processing error:', errorData);
      return null;
    }

    const data = await response.json();
    console.log('✅ Q&A result:', data.has_question ? 'Question found' : 'No question');
    return data;

  } catch (error) {
    console.error('❌ Question processing fetch error:', error);
    return null;
  }
}

// ============================================================================
// STREAMING COMPONENTS
// ============================================================================

function StreamingText({ text, isComplete, className = "" }) {
  const [displayedWords, setDisplayedWords] = useState([]);
  const [currentWordIndex, setCurrentWordIndex] = useState(0);

  useEffect(() => {
    if (isComplete) {
      setDisplayedWords(text.split(' '));
      return;
    }

    const words = text.split(' ');
    if (currentWordIndex < words.length) {
      const timer = setTimeout(() => {
        setDisplayedWords(words.slice(0, currentWordIndex + 1));
        setCurrentWordIndex(currentWordIndex + 1);
      }, 80);

      return () => clearTimeout(timer);
    }
  }, [text, currentWordIndex, isComplete]);

  return (
    <p className={`whitespace-pre-wrap leading-relaxed ${className}`}>
      {displayedWords.join(' ')}
      {!isComplete && currentWordIndex < text.split(' ').length && (
        <span className="inline-block w-1 h-4 bg-blue-400 ml-1 animate-pulse"></span>
      )}
    </p>
  );
}

function StreamingAnswer({ text, isComplete }) {
  const [displayedText, setDisplayedText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (isComplete) {
      setDisplayedText(text);
      return;
    }

    if (currentIndex < text.length) {
      const timer = setTimeout(() => {
        setDisplayedText(text.slice(0, currentIndex + 1));
        setCurrentIndex(currentIndex + 1);
      }, 20);

      return () => clearTimeout(timer);
    }
  }, [text, currentIndex, isComplete]);

  return (
    <p className="text-gray-300 whitespace-pre-wrap leading-relaxed">
      {displayedText}
      {!isComplete && currentIndex < text.length && (
        <span className="inline-block w-1 h-4 bg-green-400 ml-1 animate-pulse"></span>
      )}
    </p>
  );
}

// ============================================================================
// QA LIST
// ============================================================================

function QAList({ qaList }) {
  return (
    <div className="space-y-4">
      {qaList.map((item, index) => {
        const questionNumber = index + 1;

        return (
          <div key={item.id} className="bg-gray-900 rounded-lg p-4 border border-gray-800 hover:border-gray-700 transition-colors">
            <div className="mb-3">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-semibold text-purple-400 bg-purple-900/30 px-2 py-1 rounded">
                  ❓ QUESTION #{questionNumber}
                </span>
              </div>
              <p className="text-gray-200 font-medium leading-relaxed">{item.question}</p>
            </div>

            <div className="border-t border-gray-800 pt-3 mt-3">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-semibold text-green-400 bg-green-900/30 px-2 py-1 rounded">💬 ANSWER</span>
              </div>
              <p className="text-gray-300 whitespace-pre-wrap leading-relaxed">
                {item.answer}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function InterviewAssist() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, loading } = useAuth();

  const [personaId] = useState(
    location.state?.personaId || localStorage.getItem("selectedPersona") || null
  );
  const [personaData] = useState(() => {
    if (location.state?.personaData) return location.state.personaData;
    const saved = localStorage.getItem("selectedPersonaData");
    try {
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });
  const [domain] = useState(
    location.state?.domain || localStorage.getItem("selectedDomain") || ""
  );
  const [settings] = useState(() => {
    if (location.state?.settings) return location.state.settings;
    const saved = localStorage.getItem("copilotSettings");
    try {
      return saved
        ? JSON.parse(saved)
        : {
          responseStyle: "professional",
          audioLanguage: "English",
          pauseInterval: 2.0,
          advancedQuestionDetection: true,
          messageDirection: "bottom",
          autoScroll: true,
          programmingLanguage: "Python",
          selectedResponseStyleId: "concise",
          defaultModel: "gpt-4o-mini"
        };
    } catch {
      return {
        responseStyle: "professional",
        audioLanguage: "English",
        pauseInterval: 2.0,
        advancedQuestionDetection: true,
        messageDirection: "bottom",
        autoScroll: true,
        programmingLanguage: "Python",
        selectedResponseStyleId: "concise",
        defaultModel: "gpt-4o-mini"
      };
    }
  });

  const [isRecording, setIsRecording] = useState(false);
  const [qaList, setQaList] = useState([]);
  const [candidateTranscript, setCandidateTranscript] = useState([]);
  const [interviewerTranscript, setInterviewerTranscript] = useState([]);
  const [activeView, setActiveView] = useState("interviewer");
  const [deepgramStatus, setDeepgramStatus] = useState("Ready");
  const [qaStatus, setQaStatus] = useState("Ready");
  const [showSettings, setShowSettings] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [showTabModal, setShowTabModal] = useState(false);
  const [tabAudioError, setTabAudioError] = useState("");

  const [currentQuestion, setCurrentQuestion] = useState("");
  const [currentAnswer, setCurrentAnswer] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isStreamingComplete, setIsStreamingComplete] = useState(false);

  const [currentCandidateParagraph, setCurrentCandidateParagraph] = useState("");
  const [currentInterviewerParagraph, setCurrentInterviewerParagraph] = useState("");
  
  const [currentCandidateInterim, setCurrentCandidateInterim] = useState("");
  const [currentInterviewerInterim, setCurrentInterviewerInterim] = useState("");

  const candidateStreamRef = useRef(null);
  const interviewerStreamRef = useRef(null);
  const candidateAudioContextRef = useRef(null);
  const interviewerAudioContextRef = useRef(null);
  const candidateProcessorRef = useRef(null);
  const interviewerProcessorRef = useRef(null);
  const candidateBufferManagerRef = useRef(null);
  const interviewerBufferManagerRef = useRef(null);
  const transcriptEndRef = useRef(null);
  const copilotEndRef = useRef(null);

  // ============================================================================
  // AUTH CHECK
  // ============================================================================
  useEffect(() => {
    if (!loading && !user) {
      navigate("/sign-in");
    }
  }, [user, loading, navigate]);

  // ============================================================================
  // AUTO-SCROLL
  // ============================================================================
  useEffect(() => {
    if (settings.autoScroll) {
      transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [candidateTranscript, interviewerTranscript, currentCandidateParagraph, currentInterviewerParagraph, currentCandidateInterim, currentInterviewerInterim, settings.autoScroll]);

  useEffect(() => {
    if (settings.autoScroll) {
      copilotEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [qaList, currentAnswer, settings.autoScroll]);

  // ============================================================================
  // TRANSCRIPT HANDLING (HTTP API)
  // ============================================================================

  const handleTranscriptFromAPI = (transcript, streamType) => {
    if (!transcript || !transcript.trim()) return;

    const timestamp = new Date().toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit"
    });

    const entry = {
      text: transcript,
      timestamp,
      id: Date.now() + Math.random()
    };

    if (streamType === 'candidate') {
      setCandidateTranscript(prev => [...prev, entry]);
      setCurrentCandidateParagraph('');
      console.log('📝 Candidate:', transcript.substring(0, 50) + '...');
    } else {
      setInterviewerTranscript(prev => [...prev, entry]);
      setCurrentInterviewerParagraph('');
      console.log('📝 Interviewer:', transcript.substring(0, 50) + '...');
      
      handleQuestionProcessing(transcript);
    }
  };

  const handleQuestionProcessing = async (transcript) => {
    try {
      setIsGenerating(true);
      
      const result = await processQuestion(transcript, settings, personaData);
      
      if (result && result.has_question && result.answer) {
        console.log('✅ Answer generated');
        
        setCurrentQuestion(result.question);
        setCurrentAnswer(result.answer);
        setIsStreamingComplete(false);

        const streamingDuration = result.answer.length * 20 + 500;

        setTimeout(() => {
          setIsStreamingComplete(true);
          
          addQA({
            question: result.question,
            answer: result.answer
          });

          setTimeout(() => {
            setCurrentQuestion("");
            setCurrentAnswer("");
            setIsStreamingComplete(false);
          }, 1000);
        }, streamingDuration);
      }

    } catch (error) {
      console.error('❌ Q&A processing error:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const flushAudioBuffer = async (audioData, streamType) => {
    const languageMap = {
      "English": "en",
      "Spanish": "es",
      "French": "fr",
      "German": "de",
      "Hindi": "hi",
      "Mandarin": "zh"
    };
    const language = languageMap[settings.audioLanguage] || "en";

    if (streamType === 'candidate') {
      setCurrentCandidateParagraph('Processing...');
    } else {
      setCurrentInterviewerParagraph('Processing...');
    }

    const result = await transcribeAudio(
      audioData, // Pass Int16Array directly
      streamType,
      language
    );

    if (result && result.transcript) {
      handleTranscriptFromAPI(result.transcript, streamType);
    } else {
      if (streamType === 'candidate') {
        setCurrentCandidateParagraph('');
      } else {
        setCurrentInterviewerParagraph('');
      }
    }
  };

  // ============================================================================
  // AUDIO CAPTURE
  // ============================================================================

  const startMicrophoneCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      candidateStreamRef.current = stream;

      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      });
      candidateAudioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      candidateProcessorRef.current = processor;

      candidateBufferManagerRef.current = new AudioBufferManager(
        flushAudioBuffer,
        'candidate',
        settings.pauseInterval * 1000 || 2000
      );

      processor.onaudioprocess = (e) => {
        if (isPaused) return;

        const inputData = e.inputBuffer.getChannelData(0);
        const pcm16 = new Int16Array(inputData.length);

        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        candidateBufferManagerRef.current?.add(pcm16);
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      console.log('✓ Microphone started');
    } catch (err) {
      console.error('❌ Microphone error:', err);
      setTabAudioError('Microphone denied');
      throw err;
    }
  };

  const startSystemAudioCapture = async () => {
    if (interviewerStreamRef.current) {
      console.warn('⚠️ Interviewer audio already capturing');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          width: { ideal: 1 },
          height: { ideal: 1 },
          frameRate: { ideal: 1 }
        },
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          suppressLocalAudioPlayback: false
        }
      });

      const videoTrack = stream.getVideoTracks()[0];
      if (videoTrack) {
        videoTrack.stop();
        stream.removeTrack(videoTrack);
      }

      const audioTracks = stream.getAudioTracks();
      if (audioTracks.length === 0) {
        throw new Error('No audio track. Enable "Share audio".');
      }

      interviewerStreamRef.current = stream;

      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      });
      interviewerAudioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      interviewerProcessorRef.current = processor;

      interviewerBufferManagerRef.current = new AudioBufferManager(
        flushAudioBuffer,
        'interviewer',
        settings.pauseInterval * 1000 || 2000
      );

      processor.onaudioprocess = (e) => {
        if (isPaused) return;

        const inputData = e.inputBuffer.getChannelData(0);
        const pcm16 = new Int16Array(inputData.length);

        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        interviewerBufferManagerRef.current?.add(pcm16);
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      console.log('✓ System audio started');
      setShowTabModal(false);
    } catch (err) {
      console.error('❌ System audio error:', err);

      if (err.name === 'NotAllowedError') {
        setTabAudioError('Screen share denied');
      } else if (err.message.includes('No audio track')) {
        setTabAudioError('No audio. Check "Share tab audio"');
        setShowTabModal(true);
      } else {
        setTabAudioError(`Failed: ${err.message}`);
      }
      throw err;
    }
  };

  const stopAllCapture = () => {
    if (candidateBufferManagerRef.current) {
      candidateBufferManagerRef.current.flush();
      candidateBufferManagerRef.current.clear();
      candidateBufferManagerRef.current = null;
    }

    if (interviewerBufferManagerRef.current) {
      interviewerBufferManagerRef.current.flush();
      interviewerBufferManagerRef.current.clear();
      interviewerBufferManagerRef.current = null;
    }

    if (candidateProcessorRef.current) {
      candidateProcessorRef.current.disconnect();
      candidateProcessorRef.current = null;
    }
    if (candidateAudioContextRef.current) {
      candidateAudioContextRef.current.close();
      candidateAudioContextRef.current = null;
    }
    if (candidateStreamRef.current) {
      candidateStreamRef.current.getTracks().forEach(track => track.stop());
      candidateStreamRef.current = null;
    }

    if (interviewerProcessorRef.current) {
      interviewerProcessorRef.current.disconnect();
      interviewerProcessorRef.current = null;
    }
    if (interviewerAudioContextRef.current) {
      interviewerAudioContextRef.current.close();
      interviewerAudioContextRef.current = null;
    }
    if (interviewerStreamRef.current) {
      interviewerStreamRef.current.getTracks().forEach(track => track.stop());
      interviewerStreamRef.current = null;
    }

    setCurrentCandidateParagraph('');
    setCurrentInterviewerParagraph('');
    setCurrentCandidateInterim('');
    setCurrentInterviewerInterim('');

    console.log('✓ All capture stopped');
  };

  // ============================================================================
  // RECORDING CONTROL
  // ============================================================================

  const addQA = (qa) => {
    setQaList((prev) => {
      const isDuplicate = prev.some(
        (item) => item.question.trim().toLowerCase() === qa.question.trim().toLowerCase()
      );
      if (isDuplicate) return prev;

      const newQA = { ...qa, id: Date.now() + Math.random() };
      return [...prev, newQA];
    });
  };

  const startRecording = async () => {
    try {
      setTabAudioError("");
      setDeepgramStatus("Starting...");
      setQaStatus("Starting...");

      await startMicrophoneCapture();
      setShowTabModal(true);

      setIsRecording(true);
      setDeepgramStatus("🎤 Recording (Select Tab)");
      setQaStatus("🤖 Q&A Active");
    } catch (err) {
      console.error('❌ Failed to start:', err);
      stopAllCapture();
      setIsRecording(false);
      setDeepgramStatus("Error");
      setQaStatus("Error");
    }
  };

  const stopRecording = () => {
    stopAllCapture();
    setIsRecording(false);
    setDeepgramStatus("Stopped");
    setQaStatus("Stopped");
  };

  const handleTabAudioSelection = async () => {
    try {
      await startSystemAudioCapture();
      setDeepgramStatus("🎤 Recording Active");
    } catch (err) {
      // Error handled in startSystemAudioCapture
    }
  };

  const generatePDF = () => {
    if (!qaList || qaList.length === 0) {
      alert("No Q&A data");
      return;
    }

    const doc = new jsPDF();
    let y = 10;

    doc.setFontSize(16);
    doc.text("Interview Q&A", 10, y);
    y += 10;

    if (personaData) {
      doc.setFontSize(10);
      doc.text(`${personaData.position} @ ${personaData.company_name}`, 10, y);
      y += 6;
    }

    if (domain) {
      doc.text(`Domain: ${domain}`, 10, y);
      y += 10;
    }

    doc.setFontSize(12);
    qaList.forEach((item, index) => {
      if (y > 260) {
        doc.addPage();
        y = 10;
      }

      doc.text(`Q${index + 1}: ${item.question}`, 10, y);
      y += 10;

      const lines = doc.splitTextToSize(`A: ${item.answer}`, 180);
      doc.text(lines, 10, y);
      y += lines.length * 7 + 5;
    });

    doc.save(`Interview_QnA_${new Date().toISOString().split('T')[0]}.pdf`);
  };

  // ============================================================================
  // PAUSE HANDLING
  // ============================================================================

  useEffect(() => {
    if (isPaused) {
      candidateBufferManagerRef.current?.pause();
      interviewerBufferManagerRef.current?.pause();
    } else {
      candidateBufferManagerRef.current?.resume();
      interviewerBufferManagerRef.current?.resume();
    }
  }, [isPaused]);

  // ============================================================================
  // CLEANUP
  // ============================================================================

  useEffect(() => {
    return () => {
      stopAllCapture();
    };
  }, []);

  // ============================================================================
  // LOADING & AUTH
  // ============================================================================

  if (loading) {
    return (
      <div className="h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-white text-lg">Loading...</div>
      </div>
    );
  }

  if (!user) return null;

  const currentTranscript = activeView === "interviewer" ? interviewerTranscript : candidateTranscript;
  const currentParagraph = activeView === "interviewer" ? currentInterviewerParagraph : currentCandidateParagraph;
  const currentInterim = activeView === "interviewer" ? currentInterviewerInterim : currentCandidateInterim;

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <div className="h-screen bg-gray-950 text-gray-100 flex flex-col">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate("/interview")}
              className="text-gray-400 hover:text-gray-200 transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
            <div>
              <h1 className="text-xl font-semibold text-white">
                {personaData ? `${personaData.position} @ ${personaData.company_name}` : "Interview Assistant"}
              </h1>
              <p className="text-xs text-gray-500 mt-0.5">
                Vercel Edition - HTTP API (WAV Fixed)
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold text-lg transition-all transform hover:scale-105 shadow-2xl ${
                isRecording
                  ? "bg-red-600 hover:bg-red-700 shadow-red-500/50"
                  : "bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 shadow-blue-500/50"
              }`}
            >
              {isRecording ? (
                <>
                  <MicOff className="w-5 h-5" />
                  Stop Recording
                </>
              ) : (
                <>
                  <Mic className="w-5 h-5" />
                  Start Recording
                </>
              )}
            </button>

            {isRecording && (
              <div className="flex items-center gap-2 text-red-400 animate-pulse">
                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                <span className="text-sm font-medium">Recording</span>
              </div>
            )}

            <button
              onClick={generatePDF}
              className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50"
              disabled={qaList.length === 0}
            >
              <Download className="w-5 h-5" />
            </button>

            <button
              onClick={() => setShowSettings(true)}
              className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
            >
              <Settings className="w-5 h-5" />
            </button>

            <button
              onClick={() => navigate("/interview")}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg font-medium transition-colors"
            >
              Exit
            </button>
          </div>
        </div>

        {personaData && (
          <div className="flex items-center gap-6 text-sm text-gray-400">
            <span><strong className="text-purple-400">Position:</strong> {personaData.position}</span>
            <span><strong className="text-purple-400">Company:</strong> {personaData.company_name}</span>
            {domain && <span><strong className="text-pink-400">Domain:</strong> {domain}</span>}
          </div>
        )}
      </header>

      {/* Tab Modal */}
      {showTabModal && isRecording && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 backdrop-blur-sm">
          <div className="bg-gray-900 rounded-xl p-6 w-full max-w-lg border border-gray-800 shadow-2xl">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xl font-semibold text-white">📡 Enable Interviewer Audio</h3>
              <button
                onClick={() => {
                  setShowTabModal(false);
                  if (!interviewerStreamRef.current) {
                    stopRecording();
                  }
                }}
                className="text-gray-400 hover:text-gray-200"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4">
              <div className="bg-yellow-900/20 border border-yellow-700/50 rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                  <div className="text-sm text-yellow-200">
                    <p className="font-semibold mb-2">Enable "Share tab audio"</p>
                    <ol className="list-decimal list-inside space-y-1 text-xs">
                      <li>Click "Select Tab Audio"</li>
                      <li>Choose interview tab (Zoom/Meet)</li>
                      <li><strong className="text-yellow-300">Check "Share tab audio"</strong></li>
                      <li>Click "Share"</li>
                    </ol>
                  </div>
                </div>
              </div>

              {tabAudioError && (
                <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-3">
                  <p className="text-red-300 text-sm">{tabAudioError}</p>
                </div>
              )}

              <button
                onClick={handleTabAudioSelection}
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white py-3 rounded-lg font-medium"
              >
                Select Tab Audio
              </button>

              <p className="text-xs text-gray-400 text-center">
                Microphone already active. This captures interviewer voice.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* LEFT: Deepgram Dual-Stream Transcription */}
        <div className="w-1/2 border-r border-gray-800 flex flex-col">
          <div className="bg-gray-900 border-b border-gray-800 flex items-center">
            <button
              onClick={() => setActiveView("interviewer")}
              className={`flex items-center gap-2 px-6 py-3 text-sm font-medium transition-all border-b-2 ${
                activeView === "interviewer"
                  ? "border-green-500 text-green-400 bg-gray-800"
                  : "border-transparent text-gray-400 hover:text-gray-300"
              }`}
            >
              <Volume2 className="w-4 h-4" />
              Interviewer
              {interviewerTranscript.length > 0 && (
                <span className="ml-1 text-xs bg-green-900/30 text-green-400 px-2 py-0.5 rounded-full">
                  {interviewerTranscript.length}
                </span>
              )}
            </button>

            <button
              onClick={() => setActiveView("candidate")}
              className={`flex items-center gap-2 px-6 py-3 text-sm font-medium transition-all border-b-2 ${
                activeView === "candidate"
                  ? "border-blue-500 text-blue-400 bg-gray-800"
                  : "border-transparent text-gray-400 hover:text-gray-300"
              }`}
            >
              <Mic className="w-4 h-4" />
              Candidate (You)
              {candidateTranscript.length > 0 && (
                <span className="ml-1 text-xs bg-blue-900/30 text-blue-400 px-2 py-0.5 rounded-full">
                  {candidateTranscript.length}
                </span>
              )}
            </button>

            <div className="ml-auto px-4 py-3">
              <button
                onClick={() => setIsPaused(!isPaused)}
                className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
                disabled={!isRecording}
              >
                {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
            {currentTranscript.length === 0 && !currentParagraph && !currentInterim ? (
              <div className="h-full flex items-center justify-center">
                <div className="text-center">
                  <div className="mb-4">
                    {activeView === "interviewer" ? (
                      <Volume2 className="w-16 h-16 mx-auto text-gray-700" />
                    ) : (
                      <Mic className="w-16 h-16 mx-auto text-gray-700" />
                    )}
                  </div>
                  <p className="text-gray-500 mb-2 font-medium">
                    {activeView === "interviewer" ? "Interviewer speech" : "Your speech"}
                  </p>
                  <p className="text-gray-600 text-sm">Click "Start Recording" to begin</p>
                </div>
              </div>
            ) : (
              <>
                {currentTranscript.map((entry) => (
                  <div
                    key={entry.id}
                    className="bg-gray-900 rounded-lg p-4 border border-gray-800 hover:border-gray-700 transition-colors"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <span
                        className={`text-xs font-semibold px-2 py-1 rounded ${
                          activeView === "interviewer"
                            ? "text-green-400 bg-green-900/30"
                            : "text-blue-400 bg-blue-900/30"
                        }`}
                      >
                        {activeView === "interviewer" ? "Interviewer" : "You"}
                      </span>
                      <span className="text-xs text-gray-500">{entry.timestamp}</span>
                    </div>
                    <p className="text-gray-200 leading-relaxed">{entry.text}</p>
                  </div>
                ))}

                {currentParagraph && (
                  <div
                    className={`bg-gray-900/50 rounded-lg p-4 border-2 ${
                      activeView === "interviewer" ? "border-green-500/30" : "border-blue-500/30"
                    } animate-pulse`}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <span
                        className={`text-xs font-semibold flex items-center gap-2 px-2 py-1 rounded ${
                          activeView === "interviewer"
                            ? "text-green-300 bg-green-900/20"
                            : "text-blue-300 bg-blue-900/20"
                        }`}
                      >
                        {activeView === "interviewer" ? "Interviewer" : "You"}
                        <span className="text-xs">processing...</span>
                      </span>
                    </div>
                    <p className="text-gray-200 leading-relaxed italic">{currentParagraph}</p>
                  </div>
                )}
              </>
            )}
            <div ref={transcriptEndRef} />
          </div>
        </div>

        {/* RIGHT: Q&A Copilot */}
        <div className="w-1/2 flex flex-col">
          <div className="bg-gray-900 px-6 py-3 border-b border-gray-800 flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-white">Interview Copilot</h2>
              <p className="text-xs text-gray-500 mt-0.5">AI Q&A (Automatic)</p>
            </div>
            <div className="text-sm text-gray-400 flex items-center gap-2">
              <span className="bg-gray-800 px-3 py-1 rounded-full">
                {qaList.length} answer{qaList.length !== 1 ? 's' : ''}
              </span>
              {isGenerating && (
                <span className="flex items-center gap-1 text-purple-400">
                  <span className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></span>
                  generating...
                </span>
              )}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-6 py-4">
            {qaList.length === 0 && !currentQuestion ? (
              <div className="h-full flex items-center justify-center">
                <div className="text-center">
                  <div className="mb-4">
                    <div className="w-16 h-16 mx-auto bg-gray-800 rounded-full flex items-center justify-center">
                      <span className="text-3xl">🤖</span>
                    </div>
                  </div>
                  <p className="text-gray-500 mb-2 font-medium">AI answers appear here</p>
                  <p className="text-gray-600 text-sm">Click "Start Recording" to begin</p>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {qaList.length > 0 && <QAList qaList={qaList} />}

                {currentQuestion && (
                  <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 border-2 border-purple-500/50 rounded-lg p-5 shadow-lg">
                    <div className="mb-4">
                      <div className="flex items-center gap-2 mb-3">
                        <span className="text-xs font-semibold text-purple-300 bg-purple-900/50 px-3 py-1 rounded-full">
                          ❓ CURRENT QUESTION
                        </span>
                      </div>
                      <p className="text-white font-medium text-lg leading-relaxed">{currentQuestion}</p>
                    </div>

                    <div className="border-t border-purple-500/30 pt-4">
                      <div className="flex items-center gap-2 mb-3">
                        <span className="text-xs font-semibold text-green-300 bg-green-900/50 px-3 py-1 rounded-full">
                          💬 ANSWER
                        </span>
                        {isGenerating && !currentAnswer && (
                          <span className="text-xs text-purple-300 flex items-center gap-1">
                            <span className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></span>
                            Analyzing...
                          </span>
                        )}
                        {currentAnswer && !isStreamingComplete && (
                          <span className="text-xs text-green-400 flex items-center gap-1">
                            <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse"></span>
                            streaming...
                          </span>
                        )}
                      </div>

                      {currentAnswer ? (
                        <StreamingAnswer text={currentAnswer} isComplete={isStreamingComplete} />
                      ) : (
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
                          <div
                            className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"
                            style={{ animationDelay: "0.2s" }}
                          ></div>
                          <div
                            className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"
                            style={{ animationDelay: "0.4s" }}
                          ></div>
                          <span className="text-sm text-gray-300 ml-2">Preparing answer...</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
            <div ref={copilotEndRef} />
          </div>
        </div>
      </div>

      {/* Status Bar */}
      <div className="bg-gray-900 border-t border-gray-800 px-6 py-3 flex items-center justify-center gap-6">
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              deepgramStatus.includes("Recording")
                ? "bg-green-500 animate-pulse"
                : deepgramStatus.includes("Error") || deepgramStatus.includes("Reconnecting")
                ? "bg-yellow-500"
                : "bg-gray-600"
            }`}
          />
          <span className="text-sm text-gray-400">Deepgram: {deepgramStatus}</span>
        </div>

        <div className="h-4 w-px bg-gray-700" />

        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              qaStatus.includes("Active")
                ? "bg-blue-500 animate-pulse"
                : qaStatus.includes("Error") || qaStatus.includes("Reconnecting")
                ? "bg-yellow-500"
                : "bg-gray-600"
            }`}
          />
          <span className="text-sm text-gray-400">Q&A: {qaStatus}</span>
        </div>

        {isGenerating && (
          <>
            <div className="h-4 w-px bg-gray-700" />
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" />
              <span className="text-sm text-purple-400">Processing...</span>
            </div>
          </>
        )}

        {isPaused && (
          <>
            <div className="h-4 w-px bg-gray-700" />
            <div className="flex items-center gap-2">
              <Pause className="w-3 h-3 text-orange-400" />
              <span className="text-sm text-orange-400">Paused</span>
            </div>
          </>
        )}
      </div>

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 backdrop-blur-sm">
          <div className="bg-gray-900 rounded-xl p-6 w-full max-w-md border border-gray-800 shadow-2xl">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xl font-semibold text-white">Settings</h3>
              <button onClick={() => setShowSettings(false)} className="text-gray-400 hover:text-gray-200">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4">
              <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                <h4 className="text-sm font-medium mb-3 text-purple-400">Configuration</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between py-1">
                    <span className="text-gray-400">Style:</span>
                    <span className="text-white font-medium">{settings.selectedResponseStyleId || settings.responseStyle}</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-gray-400">Language:</span>
                    <span className="text-white font-medium">{settings.audioLanguage}</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-gray-400">Pause Time:</span>
                    <span className="text-white font-medium">{settings.pauseInterval}s</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-gray-400">Coding:</span>
                    <span className="text-white font-medium">{settings.programmingLanguage}</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-gray-400">Model:</span>
                    <span className="text-white font-medium">{settings.defaultModel}</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-gray-400">Backend:</span>
                    <span className="text-green-400 font-medium text-xs break-all">{BACKEND_URL}</span>
                  </div>
                </div>
              </div>

              <div className="bg-blue-900/20 border border-blue-800/50 rounded-lg p-4">
                <h4 className="text-sm font-medium mb-2 text-blue-300">💡 Vercel HTTP Edition (WAV Fixed)</h4>
                <ul className="text-gray-400 text-sm space-y-1 list-disc list-inside">
                  <li>HTTP POST API with proper WAV encoding</li>
                  <li>~{settings.pauseInterval}s buffered transcription</li>
                  <li>Fixed Deepgram audio format error</li>
                  <li>Serverless auto-scaling</li>
                </ul>
              </div>

              <button
                onClick={() => setShowSettings(false)}
                className="w-full bg-purple-600 hover:bg-purple-700 text-white py-2.5 rounded-lg font-medium"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}