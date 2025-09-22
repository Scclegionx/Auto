package com.auto_fe.auto_fe.audio

import android.content.Context
import android.media.AudioManager
import android.speech.SpeechRecognizer
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.content.Intent
import android.os.Bundle
import android.util.Log
import java.util.*

class AudioManager(private val context: Context) {
    private var speechRecognizer: SpeechRecognizer? = null
    private var audioRecorder: AudioRecorder? = null
    
    interface AudioManagerCallback {
        fun onSpeechResult(spokenText: String)
        fun onConfirmationResult(confirmed: Boolean)
        fun onError(error: String)
    }
    
    init {
        audioRecorder = AudioRecorder(context)
    }
    
    fun startVoiceInteraction(callback: AudioManagerCallback) {
        // Bước 1: Nói câu hỏi
        audioRecorder?.speak("Bạn cần tôi trợ giúp bạn điều gì?")
        
        // Bước 2: Bắt đầu nhận diện giọng nói
        startSpeechRecognition(callback)
    }
    
    private fun startSpeechRecognition(callback: AudioManagerCallback) {
        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            callback.onError("Thiết bị không hỗ trợ nhận diện giọng nói")
            return
        }
        
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context)
        speechRecognizer?.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
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
            }
            
            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!matches.isNullOrEmpty()) {
                    val spokenText = matches[0]
                    callback.onSpeechResult(spokenText)
                } else {
                    callback.onError("Không nhận diện được giọng nói")
                }
            }
            
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
        
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "vi-VN")
            putExtra(RecognizerIntent.EXTRA_PROMPT, "Hãy nói lệnh của bạn")
        }
        
        speechRecognizer?.startListening(intent)
    }
    
    fun confirmCommand(command: String, callback: AudioManagerCallback) {
        // Nói câu xác nhận
        val confirmationText = "Có phải bạn muốn $command?"
        audioRecorder?.speak(confirmationText)
        
        // Đợi một chút để TTS hoàn thành
        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
            // Bắt đầu nhận diện xác nhận
            startConfirmationRecognition(callback)
        }, 2000) // Đợi 2 giây
    }
    
    private fun startConfirmationRecognition(callback: AudioManagerCallback) {
        speechRecognizer?.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            
            override fun onError(error: Int) {
                callback.onError("Lỗi khi xác nhận: $error")
            }
            
            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!matches.isNullOrEmpty()) {
                    val response = matches[0].lowercase()
                    Log.d("AudioManager", "Confirmation response: $response")
                    val confirmed = response.contains("có") || response.contains("đúng") || 
                                  response.contains("phải") || response.contains("yes")
                    Log.d("AudioManager", "Confirmation result: $confirmed")
                    callback.onConfirmationResult(confirmed)
                } else {
                    Log.d("AudioManager", "No confirmation matches found")
                    callback.onConfirmationResult(false)
                }
            }
            
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
        
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "vi-VN")
            putExtra(RecognizerIntent.EXTRA_PROMPT, "Trả lời có hoặc không")
        }
        
        speechRecognizer?.startListening(intent)
    }
    
    fun release() {
        speechRecognizer?.destroy()
        speechRecognizer = null
        audioRecorder?.release()
        audioRecorder = null
    }
}
