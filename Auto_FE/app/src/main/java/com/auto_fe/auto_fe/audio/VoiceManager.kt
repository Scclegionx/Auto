package com.auto_fe.auto_fe.audio

import android.content.Context
import android.speech.SpeechRecognizer
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.content.Intent
import android.os.Bundle
import android.util.Log
import java.util.*

/**
 * VoiceManager - Quản lý voice interaction với SpeechRecognizer singleton
 */
class VoiceManager private constructor(private val context: Context) {
    private var audioRecorder: AudioRecorder? = null
    private var isBusy = false
    private var pendingCallback: VoiceControllerCallback? = null
    
    companion object {
        @Volatile
        private var INSTANCE: VoiceManager? = null
        
        fun getInstance(context: Context): VoiceManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: VoiceManager(context.applicationContext).also { INSTANCE = it }
            }
        }
    }
    
    interface VoiceControllerCallback {
        fun onSpeechResult(spokenText: String)
        fun onConfirmationResult(confirmed: Boolean)
        fun onError(error: String)
        fun onAudioLevelChanged(level: Int) // Thêm callback cho audio level
    }
    
    init {
        audioRecorder = AudioRecorder.getInstance(context)
    }
    
    private fun ensureAudioRecorder(): AudioRecorder {
        if (audioRecorder == null) {
            audioRecorder = AudioRecorder.getInstance(context)
        }
        return audioRecorder as AudioRecorder
    }

    
    /**
     * API duy nhất: Text-to-Speech với delay tùy chỉnh
     * @param text Text cần nói
     * @param delaySeconds Số giây delay trước khi bắt đầu STT
     * @param callback Callback để nhận kết quả STT
     */
    fun textToSpeech(text: String, delaySeconds: Int, callback: VoiceControllerCallback) {
        if (isBusy) {
            Log.d("VoiceManager", "VoiceManager is busy, rejecting request")
            callback.onError("VoiceManager đang bận, vui lòng thử lại sau")
            return
        }
        
        isBusy = true
        Log.d("VoiceManager", "Speaking: $text")
        
        // Bước 1: Nói text
        ensureAudioRecorder().speak(text)
        
        // Bước 2: Đợi delaySeconds rồi bắt đầu STT
        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
            startSpeechRecognition(callback)
        }, delaySeconds * 1000L) // Convert to milliseconds
    }
    
    
    private fun startSpeechRecognition(callback: VoiceControllerCallback) {
        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            callback.onError("Thiết bị không hỗ trợ nhận diện giọng nói")
            releaseBusyState()
            return
        }
        
        // Sử dụng SpeechRecognizerManager singleton
        val speechManager = SpeechRecognizerManager.getInstance(context)
        speechManager.startRecognition(
            object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {
                    // Chỉ gửi level khi đang busy (recording)
                    if (isBusy) {
                        // Convert RMS dB to level 0-3
                        val level = when {
                            rmsdB < 0.1f -> 0
                            rmsdB < 0.3f -> 1
                            rmsdB < 0.6f -> 2
                            else -> 3
                        }
                        callback.onAudioLevelChanged(level)
                    }
                }
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {}
                
                override fun onError(error: Int) {
                    val errorMessage = when (error) {
                        SpeechRecognizer.ERROR_AUDIO -> "Lỗi âm thanh"
                        SpeechRecognizer.ERROR_CLIENT -> "Lỗi client"
                        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Không đủ quyền"
                        SpeechRecognizer.ERROR_NETWORK -> "Lỗi mạng"
                        SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Timeout mạng"
                        SpeechRecognizer.ERROR_NO_MATCH -> "Không nhận diện được"
                        SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Recognizer đang bận"
                        SpeechRecognizer.ERROR_SERVER -> "Lỗi server"
                        SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Timeout giọng nói"
                        else -> "Lỗi không xác định"
                    }
                    callback.onError(errorMessage)
                    speechManager.release() // Release SpeechRecognizer
                    releaseBusyState()
                }
                
                override fun onResults(results: Bundle?) {
                    val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    if (!matches.isNullOrEmpty()) {
                        val spokenText = matches[0]
                        callback.onSpeechResult(spokenText)
                    } else {
                        callback.onError("Không nhận diện được giọng nói")
                    }
                    speechManager.release() // Release SpeechRecognizer
                    releaseBusyState()
                }
                
                override fun onPartialResults(partialResults: Bundle?) {}
                override fun onEvent(eventType: Int, params: Bundle?) {}
            }
        )
    }
    
    
    fun speak(text: String) {
        ensureAudioRecorder().speak(text)
    }
    
    private fun releaseBusyState() {
        isBusy = false
        Log.d("VoiceManager", "Released busy state")
        // Không cần pending request nữa - FE đã disable button
    }
    
    fun resetBusyState() {
        isBusy = false
        pendingCallback = null
        // Dừng ghi âm khi reset busy state
        SpeechRecognizerManager.getInstance(context).stopRecognition()
        Log.d("VoiceManager", "Reset busy state")
    }
    
    fun release() {
        // Release SpeechRecognizerManager
        SpeechRecognizerManager.getInstance(context).release()
        
        audioRecorder?.release()
        audioRecorder = null
        isBusy = false
    }
}

/**
 * SpeechRecognizerManager - Singleton quản lý SpeechRecognizer
 */
class SpeechRecognizerManager private constructor(private val context: Context) {
    private var speechRecognizer: SpeechRecognizer? = null
    private var isInUse = false
    
    companion object {
        @Volatile
        private var INSTANCE: SpeechRecognizerManager? = null
        
        fun getInstance(context: Context): SpeechRecognizerManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: SpeechRecognizerManager(context.applicationContext).also { INSTANCE = it }
            }
        }
    }
    
    fun startRecognition(listener: RecognitionListener) {
        if (isInUse) {
            Log.w("SpeechRecognizerManager", "SpeechRecognizer is already in use, forcing release")
            release()
        }
        
        try {
            // Create new recognizer
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context)
            speechRecognizer?.setRecognitionListener(listener)
            
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, "vi-VN")
                putExtra(RecognizerIntent.EXTRA_PROMPT, "Hãy nói lệnh của bạn")
            }
            
            isInUse = true
            speechRecognizer?.startListening(intent)
            Log.d("SpeechRecognizerManager", "Started recognition")
            
        } catch (e: Exception) {
            Log.e("SpeechRecognizerManager", "Error starting recognition: ${e.message}")
            isInUse = false
        }
    }
    
    fun stopRecognition() {
        if (speechRecognizer != null && isInUse) {
            speechRecognizer?.stopListening()
            Log.d("SpeechRecognizerManager", "Stopped recognition")
        }
    }
    
    fun cancelRecognition() {
        if (speechRecognizer != null && isInUse) {
            speechRecognizer?.cancel()
            Log.d("SpeechRecognizerManager", "Cancelled recognition")
        }
    }
    
    fun release() {
        if (speechRecognizer != null) {
            speechRecognizer?.destroy()
            speechRecognizer = null
            isInUse = false
            Log.d("SpeechRecognizerManager", "Released SpeechRecognizer")
        }
    }
}
