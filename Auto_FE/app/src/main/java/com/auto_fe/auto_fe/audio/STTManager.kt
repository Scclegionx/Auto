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
    private var isWaitingForResult = false // Đang chờ kết quả sau khi dừng listening
    private var currentCallback: STTCallback? = null
    private var lastSpeechTime = 0L
    private var silenceStartTime = 0L
    private val silenceThreshold = 0.15f // Ngưỡng RMS để coi là im lặng
    private val silenceDurationMs = 800L // Thời gian im lặng
    private val silenceCheckHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private var silenceCheckRunnable: Runnable? = null
    private val timeoutHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private var timeoutRunnable: Runnable? = null
    private val timeoutDurationMs = 5000L // 5 giây timeout
    
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
                    lastSpeechTime = System.currentTimeMillis()
                    silenceStartTime = 0L
                    // Khởi động lại timeout sau khi bắt đầu nói (để phát hiện nếu người dùng ngừng nói)
                    startTimeoutTimer()
                }
                
                override fun onRmsChanged(rmsdB: Float) {
                    val currentTime = System.currentTimeMillis()
                    
                    // Convert RMS dB to level 0-3
                    val level = when {
                        rmsdB < 0.1f -> 0
                        rmsdB < 0.3f -> 1
                        rmsdB < 0.6f -> 2
                        else -> 3
                    }
                    currentCallback?.onAudioLevelChanged(level)
                    
                    // Phát hiện im lặng để tự động dừng nhanh hơn
                    if (rmsdB < silenceThreshold) {
                        // Đang im lặng
                        if (silenceStartTime == 0L) {
                            silenceStartTime = currentTime
                        } else {
                            val silenceDuration = currentTime - silenceStartTime
                            // Nếu im lặng quá lâu, tự động dừng
                            if (silenceDuration >= silenceDurationMs && lastSpeechTime > 0) {
                                Log.d(TAG, "Detected silence for ${silenceDuration}ms, stopping automatically")
                                try {
                                    speechRecognizer?.stopListening()
                                    isListening = false
                                    silenceStartTime = 0L
                                } catch (e: Exception) {
                                    Log.e(TAG, "Error auto-stopping on silence: ${e.message}", e)
                                }
                            }
                        }
                    } else {
                        // Có âm thanh, reset silence timer
                        lastSpeechTime = currentTime
                        silenceStartTime = 0L
                    }
                }
                
                override fun onBufferReceived(buffer: ByteArray?) {
                    // Không cần xử lý
                }
                
                override fun onEndOfSpeech() {
                    Log.d(TAG, "End of speech - stopping listening immediately")
                    // Dừng lắng nghe ngay khi người dùng nói xong để giảm thời gian chờ
                    try {
                        speechRecognizer?.stopListening()
                        isListening = false
                        isWaitingForResult = true // Đang chờ kết quả
                        silenceStartTime = 0L
                        lastSpeechTime = 0L
                        // Khởi động lại timeout để chờ kết quả
                        startTimeoutTimer()
                    } catch (e: Exception) {
                        Log.e(TAG, "Error stopping on end of speech: ${e.message}", e)
                    }
                }
                
                override fun onError(error: Int) {
                    // Hủy timeout khi có lỗi
                    cancelTimeoutTimer()
                    isWaitingForResult = false
                    
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
                    // Hủy timeout khi có kết quả
                    cancelTimeoutTimer()
                    isWaitingForResult = false
                    
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
                    // Xử lý partial results để có thể dừng sớm hơn
                    val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    if (!matches.isNullOrEmpty()) {
                        val partialText = matches[0]
                        Log.d(TAG, "Partial result: $partialText")
                        // Có thể dùng partial results để hiển thị real-time nếu cần
                    }
                }
                
                override fun onEvent(eventType: Int, params: Bundle?) {
                    // Không cần xử lý
                }
            })
            
            // Reset silence detection
            lastSpeechTime = 0L
            silenceStartTime = 0L
            
            // Tạo Intent cho Speech Recognition
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, "vi-VN")
                putExtra(RecognizerIntent.EXTRA_PROMPT, "Hãy nói lệnh của bạn")
                // Giảm thời gian chờ im lặng để phản hồi nhanh hơn
                // Thời gian im lặng trước khi coi là nói xong (giảm xuống 300ms)
                putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 300L)
                // Thời gian im lặng có thể hoàn thành (giảm xuống 200ms)
                putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 200L)
                // Bật partial results để xử lý nhanh hơn
                putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            }
            
            isListening = true
            speechRecognizer?.startListening(intent)
            Log.d(TAG, "Started listening")
            
            // Bắt đầu timeout timer
            startTimeoutTimer()
            
        } catch (e: Exception) {
            Log.e(TAG, "Error starting recognition: ${e.message}", e)
            isListening = false
            currentCallback?.onError("Lỗi khởi động nhận diện giọng nói: ${e.message}")
            currentCallback = null
            cancelTimeoutTimer()
        }
    }
    
    /**
     * Bắt đầu timer timeout
     */
    private fun startTimeoutTimer() {
        cancelTimeoutTimer() // Hủy timer cũ nếu có
        
        timeoutRunnable = Runnable {
            // Timeout nếu đang listening hoặc đang chờ kết quả
            if (isListening || isWaitingForResult) {
                Log.d(TAG, "Timeout reached after ${timeoutDurationMs}ms, no response detected")
                currentCallback?.onError("Không nghe được bác đang nói gì. Bác hãy thử lại")
                releaseRecognizer()
            } else {
                Log.d(TAG, "Timeout triggered but not listening/waiting anymore, ignoring")
            }
        }
        
        timeoutHandler.postDelayed(timeoutRunnable!!, timeoutDurationMs)
        Log.d(TAG, "Timeout timer started, will trigger after ${timeoutDurationMs}ms")
    }
    
    /**
     * Hủy timer timeout
     */
    private fun cancelTimeoutTimer() {
        timeoutRunnable?.let {
            timeoutHandler.removeCallbacks(it)
            timeoutRunnable = null
            Log.d(TAG, "Timeout timer cancelled")
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
                isWaitingForResult = true // Đang chờ kết quả
                silenceStartTime = 0L
                lastSpeechTime = 0L
                silenceCheckRunnable?.let { silenceCheckHandler.removeCallbacks(it) }
                // Không hủy timeout, vì đang chờ kết quả
                Log.d(TAG, "Stopped listening, waiting for result")
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping recognition: ${e.message}", e)
            }
        }
    }
    
    /**
     * Hủy lắng nghe (không giữ kết quả)
     */
    fun cancelListening() {
        if (speechRecognizer != null && (isListening || isWaitingForResult)) {
            try {
                speechRecognizer?.cancel()
                isListening = false
                isWaitingForResult = false
                currentCallback = null
                silenceStartTime = 0L
                lastSpeechTime = 0L
                silenceCheckRunnable?.let { silenceCheckHandler.removeCallbacks(it) }
                cancelTimeoutTimer()
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
                isWaitingForResult = false
                currentCallback = null
                silenceStartTime = 0L
                lastSpeechTime = 0L
                silenceCheckRunnable?.let { silenceCheckHandler.removeCallbacks(it) }
                cancelTimeoutTimer()
            } catch (e: Exception) {
                Log.e(TAG, "Error releasing recognizer: ${e.message}", e)
            }
        }
    }
}

