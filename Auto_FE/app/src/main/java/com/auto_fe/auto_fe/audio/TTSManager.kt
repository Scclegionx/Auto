package com.auto_fe.auto_fe.audio

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import kotlinx.coroutines.suspendCancellableCoroutine
import java.util.*
import kotlin.coroutines.resume

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

                    // Xử lý pending text
                    pendingSpeakText?.let { text ->
                        speak(text)
                    }
                    pendingSpeakText = null
                } else {
                    Log.e(TAG, "TTS initialization failed with status: $status")
                    ttsReady = false
                }
            }
        }
    }

    suspend fun speakAndAwait(text: String) = suspendCancellableCoroutine<Unit> { continuation ->
        if (tts == null) initTTS()

        if (!ttsReady) {
            Log.e(TAG, "TTS not ready yet, skipping speakAndAwait")
            // Resume ngay để không bị treo app
            if (continuation.isActive) continuation.resume(Unit)
            return@suspendCancellableCoroutine
        }

        val utteranceId = UUID.randomUUID().toString()

        // Đăng ký Listener để nghe sự kiện
        tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {
                Log.d(TAG, "TTS Started: $utteranceId")
            }

            override fun onDone(utteranceId: String?) {
                Log.d(TAG, "TTS Done: $utteranceId")
                // Khi nói xong -> Resume coroutine
                if (continuation.isActive) {
                    continuation.resume(Unit)
                }
            }

            @Deprecated("Deprecated")
            override fun onError(utteranceId: String?) {
                Log.e(TAG, "TTS Error: $utteranceId")
                // Gặp lỗi cũng phải resume để app chạy tiếp
                if (continuation.isActive) {
                    continuation.resume(Unit)
                }
            }

            override fun onError(utteranceId: String?, errorCode: Int) {
                Log.e(TAG, "TTS Error ($errorCode): $utteranceId")
                if (continuation.isActive) {
                    continuation.resume(Unit)
                }
            }
        })

        // Cấu hình params
        val params = Bundle()
        params.putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, utteranceId)

        // Ra lệnh nói
        val result = tts?.speak(text, TextToSpeech.QUEUE_FLUSH, params, utteranceId)

        if (result == TextToSpeech.ERROR) {
            Log.e(TAG, "Error initiating speech")
            if (continuation.isActive) continuation.resume(Unit)
        }

        continuation.invokeOnCancellation {
            stopSpeaking()
        }
    }

    // Các hàm cũ

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
        tts?.setOnUtteranceProgressListener(null)
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
        Log.d(TAG, "Speaking: $text")
    }

    fun speakNoFlush(text: String) {
        if (!ttsReady) return
        tts?.speak(text, TextToSpeech.QUEUE_ADD, null, null)
    }

    fun stopSpeaking() {
        try {
            tts?.stop()
            Log.d(TAG, "Stopped speaking")
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping speech: ${e.message}")
        }
    }

    fun release() {
        try {
            tts?.stop()
            tts?.shutdown()
            tts = null
            ttsReady = false
            INSTANCE = null
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing TTS: ${e.message}")
        }
    }
}