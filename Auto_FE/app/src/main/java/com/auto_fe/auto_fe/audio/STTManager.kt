package com.auto_fe.auto_fe.audio

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log

/**
 * STTManager - Quản lý Speech-to-Text (STT)
 * Singleton pattern để đảm bảo chỉ có 1 instance SpeechRecognizer trong app
 */
class STTManager private constructor(private val context: Context) {
    
    companion object {
        private const val TAG = "STTManager"
        
        @Volatile
        private var INSTANCE: STTManager? = null
        
        fun getInstance(context: Context): STTManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: STTManager(context.applicationContext).also { INSTANCE = it }
            }
        }
    }
    
    interface STTCallback {
        /**
         * Kết quả nhận diện giọng nói
         * @param spokenText Text đã được nhận diện
         */
        fun onSpeechResult(spokenText: String)
        
        /**
         * Lỗi xảy ra trong quá trình nhận diện
         * @param error Thông báo lỗi
         */
        fun onError(error: String)
        
        /**
         * Mức độ âm thanh thay đổi (để hiển thị visual feedback)
         * @param level Mức độ từ 0-3 (0 = im lặng, 3 = to nhất)
         */
        fun onAudioLevelChanged(level: Int)
    }
    
    private var speechRecognizer: SpeechRecognizer? = null
    private var isListening = false
    private var currentCallback: STTCallback? = null
    
    /**
     * Bắt đầu lắng nghe giọng nói
     * @param callback Callback để nhận kết quả
     */
    fun startListening(callback: STTCallback) {
        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            Log.e(TAG, "Speech recognition not available")
            callback.onError("Thiết bị không hỗ trợ nhận diện giọng nói")
            return
        }
        
        // Nếu đang lắng nghe, dừng trước
        if (isListening) {
            Log.w(TAG, "Already listening, stopping previous session")
            stopListening()
        }
        
        try {
            // Tạo SpeechRecognizer mới
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context)
            currentCallback = callback
            
            speechRecognizer?.setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {
                    Log.d(TAG, "Ready for speech")
                }
                
                override fun onBeginningOfSpeech() {
                    Log.d(TAG, "Beginning of speech")
                }
                
                override fun onRmsChanged(rmsdB: Float) {
                    // Convert RMS dB to level 0-3
                    val level = when {
                        rmsdB < 0.1f -> 0
                        rmsdB < 0.3f -> 1
                        rmsdB < 0.6f -> 2
                        else -> 3
                    }
                    currentCallback?.onAudioLevelChanged(level)
                }
                
                override fun onBufferReceived(buffer: ByteArray?) {
                    // Không cần xử lý
                }
                
                override fun onEndOfSpeech() {
                    Log.d(TAG, "End of speech")
                }
                
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
                    Log.e(TAG, "Recognition error: $errorMessage")
                    currentCallback?.onError(errorMessage)
                    releaseRecognizer()
                }
                
                override fun onResults(results: Bundle?) {
                    val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    if (!matches.isNullOrEmpty()) {
                        val spokenText = matches[0]
                        Log.d(TAG, "Recognition result: $spokenText")
                        currentCallback?.onSpeechResult(spokenText)
                    } else {
                        Log.w(TAG, "No recognition results")
                        currentCallback?.onError("Không nhận diện được giọng nói")
                    }
                    releaseRecognizer()
                }
                
                override fun onPartialResults(partialResults: Bundle?) {
                    // Có thể xử lý partial results nếu cần
                }
                
                override fun onEvent(eventType: Int, params: Bundle?) {
                    // Không cần xử lý
                }
            })
            
            // Tạo Intent cho Speech Recognition
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, "vi-VN")
                putExtra(RecognizerIntent.EXTRA_PROMPT, "Hãy nói lệnh của bạn")
            }
            
            isListening = true
            speechRecognizer?.startListening(intent)
            Log.d(TAG, "Started listening")
            
        } catch (e: Exception) {
            Log.e(TAG, "Error starting recognition: ${e.message}", e)
            isListening = false
            currentCallback?.onError("Lỗi khởi động nhận diện giọng nói: ${e.message}")
            currentCallback = null
        }
    }
    
    /**
     * Dừng lắng nghe (giữ kết quả đã nhận diện được)
     */
    fun stopListening() {
        if (speechRecognizer != null && isListening) {
            try {
                speechRecognizer?.stopListening()
                isListening = false
                Log.d(TAG, "Stopped listening")
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping recognition: ${e.message}", e)
            }
        }
    }
    
    /**
     * Hủy lắng nghe (không giữ kết quả)
     */
    fun cancelListening() {
        if (speechRecognizer != null && isListening) {
            try {
                speechRecognizer?.cancel()
                isListening = false
                currentCallback = null
                Log.d(TAG, "Cancelled listening")
            } catch (e: Exception) {
                Log.e(TAG, "Error cancelling recognition: ${e.message}", e)
            }
        }
    }
    
    /**
     * Kiểm tra đang lắng nghe hay không
     */
    fun isListening(): Boolean {
        return isListening
    }
    
    /**
     * Giải phóng tài nguyên SpeechRecognizer
     */
    fun release() {
        releaseRecognizer()
        Log.d(TAG, "STT released")
    }
    
    private fun releaseRecognizer() {
        if (speechRecognizer != null) {
            try {
                speechRecognizer?.destroy()
                speechRecognizer = null
                isListening = false
                currentCallback = null
            } catch (e: Exception) {
                Log.e(TAG, "Error releasing recognizer: ${e.message}", e)
            }
        }
    }
}

