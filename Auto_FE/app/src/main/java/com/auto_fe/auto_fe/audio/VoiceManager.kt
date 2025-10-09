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
        audioRecorder = AudioRecorder(context)
    }
    
    fun startVoiceInteraction(callback: VoiceControllerCallback) {
        if (isBusy) {
            Log.d("VoiceManager", "VoiceManager is busy, queuing request")
            pendingCallback = callback
            return
        }
        
        isBusy = true
        // Bước 1: Nói câu hỏi
        audioRecorder?.speak("Bạn cần tôi trợ giúp bạn điều gì?")
        
        // Bước 2: Bắt đầu nhận diện giọng nói
        startSpeechRecognition(callback)
    }
    
    fun startVoiceInteractionWithMessage(message: String, callback: VoiceControllerCallback) {
        if (isBusy) {
            Log.d("VoiceManager", "VoiceManager is busy, queuing request")
            pendingCallback = callback
            return
        }
        
        isBusy = true
        // Bước 1: Nói thông báo tùy chỉnh
        audioRecorder?.speak(message)
        
        // Bước 2: Bắt đầu nhận diện giọng nói
        startSpeechRecognition(callback)
    }
    
    fun startVoiceInteractionSilent(callback: VoiceControllerCallback) {
        if (isBusy) {
            Log.d("VoiceManager", "VoiceManager is busy, queuing request")
            pendingCallback = callback
            return
        }
        
        isBusy = true
        // Bắt đầu nhận diện ngay lập tức mà không nói gì
        startSpeechRecognition(callback)
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
    
    fun confirmCommand(command: String, callback: VoiceControllerCallback) {
        // Nói câu xác nhận
        val confirmationText = "Có phải bạn muốn $command?"
        audioRecorder?.speak(confirmationText)
        
        // Đợi một chút để TTS hoàn thành
        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
            // Bắt đầu nhận diện xác nhận
            startConfirmationRecognition(callback)
        }, 2000) // Đợi 2 giây
    }
    
    private fun startConfirmationRecognition(callback: VoiceControllerCallback) {
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
                    callback.onError("Lỗi khi xác nhận: $error")
                    speechManager.release() // Release SpeechRecognizer
                    releaseBusyState()
                }
                
                override fun onResults(results: Bundle?) {
                    val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    if (!matches.isNullOrEmpty()) {
                        val response = matches[0].lowercase()
                        Log.d("VoiceManager", "Confirmation response: $response")
                        val confirmed = when {
                            response.contains("không") || response.contains("không phải") || 
                            response.contains("sai") || response.contains("no") -> false
                            response.contains("có") || response.contains("đúng") || 
                            response.contains("phải") || response.contains("yes") -> true
                            else -> false
                        }
                        Log.d("VoiceManager", "Confirmation result: $confirmed")
                        callback.onConfirmationResult(confirmed)
                    } else {
                        Log.d("VoiceManager", "No confirmation matches found")
                        callback.onConfirmationResult(false)
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
        audioRecorder?.speak(text)
    }
    
    private fun releaseBusyState() {
        isBusy = false
        Log.d("VoiceManager", "Released busy state")
        
        // Process pending request if any
        pendingCallback?.let { callback ->
            Log.d("VoiceManager", "Processing pending request")
            pendingCallback = null
            // Retry the last request
            android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                startVoiceInteraction(callback)
            }, 1000) // Wait 1 second before retry
        }
    }
    
    fun release() {
        // Release SpeechRecognizerManager
        SpeechRecognizerManager.getInstance(context).release()
        
        audioRecorder?.release()
        audioRecorder = null
        isBusy = false
        pendingCallback = null
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
