package com.auto_fe.auto_fe.audio

import android.content.Context
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.*

/**
 * TTSManager - Quản lý Text-to-Speech (TTS)
 * Singleton pattern để đảm bảo chỉ có 1 instance TTS trong app
 */
class TTSManager private constructor(private val context: Context) {
    
    companion object {
        private const val TAG = "TTSManager"
        
        @Volatile
        private var INSTANCE: TTSManager? = null
        
        fun getInstance(context: Context): TTSManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: TTSManager(context.applicationContext).also { INSTANCE = it }
            }
        }
    }
    
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var pendingSpeakText: String? = null
    
    init {
        initTTS()
    }
    
    private fun initTTS() {
        if (tts == null) {
            ttsReady = false
            tts = TextToSpeech(context) { status ->
                if (status == TextToSpeech.SUCCESS) {
                    tts?.language = Locale("vi", "VN")
                    ttsReady = true
                    Log.d(TAG, "TTS initialized successfully")
                    
                    // Thực thi yêu cầu đang chờ (nếu có)
                    pendingSpeakText?.let { text ->
                        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
                    }
                    pendingSpeakText = null
                } else {
                    Log.e(TAG, "TTS initialization failed with status: $status")
                    ttsReady = false
                }
            }
        }
    }
    
    /**
     * Nói text (xóa hàng đợi cũ, nói ngay)
     * @param text Text cần nói
     */
    fun speak(text: String) {
        if (tts == null) {
            pendingSpeakText = text
            initTTS()
            return
        }
        if (!ttsReady) {
            pendingSpeakText = text
            return
        }
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
        Log.d(TAG, "Speaking: $text")
    }
    
    /**
     * Nói text (thêm vào hàng đợi, không xóa cũ)
     * Dùng khi cần nói nhiều câu liên tiếp
     * @param text Text cần nói
     */
    fun speakNoFlush(text: String) {
        if (tts == null) {
            pendingSpeakText = text
            initTTS()
            return
        }
        if (!ttsReady) {
            pendingSpeakText = text
            return
        }
        tts?.speak(text, TextToSpeech.QUEUE_ADD, null, null)
        Log.d(TAG, "Speaking (no flush): $text")
    }
    
    /**
     * Dừng nói ngay lập tức
     */
    fun stopSpeaking() {
        try {
            tts?.stop()
            Log.d(TAG, "Stopped speaking")
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping speech: ${e.message}")
        }
    }
    
    /**
     * Giải phóng tài nguyên TTS
     */
    fun release() {
        try {
            tts?.stop()
            tts?.shutdown()
            tts = null
            ttsReady = false
            pendingSpeakText = null
            Log.d(TAG, "TTS released")
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing TTS: ${e.message}")
        }
    }
    
    /**
     * Kiểm tra TTS đã sẵn sàng chưa
     */
    fun isReady(): Boolean {
        return ttsReady
    }
}

