package com.auto_fe.auto_fe.audio

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.AudioManager
import android.speech.tts.TextToSpeech
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.*

class AudioRecorder(private val context: Context) {
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var recordingFile: File? = null
    private var pendingSpeakText: String? = null
    private var pendingSpeakCallback: (() -> Unit)? = null
    
    companion object {
        @Volatile
        private var INSTANCE: AudioRecorder? = null
        
        fun getInstance(context: Context): AudioRecorder {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: AudioRecorder(context.applicationContext).also { INSTANCE = it }
            }
        }
    }
    
    interface AudioRecorderCallback {
        fun onRecordingComplete(audioFile: File)
        fun onError(error: String)
    }
    
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
                    // Thực thi yêu cầu đang chờ (nếu có)
                    pendingSpeakText?.let { text ->
                        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
                        // Nếu có callback chờ, gọi sau 2s
                        pendingSpeakCallback?.let { cb ->
                            android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                                cb()
                            }, 2000)
                        }
                    }
                    pendingSpeakText = null
                    pendingSpeakCallback = null
                }
            }
        }
    }
    
    fun speak(text: String) {
        if (tts == null) {
            pendingSpeakText = text
            pendingSpeakCallback = null
            initTTS()
            return
        }
        if (!ttsReady) {
            pendingSpeakText = text
            pendingSpeakCallback = null
            return
        }
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
    }
    
    fun speakNoFlush(text: String) {
        if (tts == null) {
            pendingSpeakText = text
            pendingSpeakCallback = null
            initTTS()
            return
        }
        if (!ttsReady) {
            pendingSpeakText = text
            pendingSpeakCallback = null
            return
        }
        tts?.speak(text, TextToSpeech.QUEUE_ADD, null, null)
    }

    fun speak(text: String, callback: () -> Unit) {
        if (tts == null) {
            pendingSpeakText = text
            pendingSpeakCallback = callback
            initTTS()
            return
        }
        if (!ttsReady) {
            pendingSpeakText = text
            pendingSpeakCallback = callback
            return
        }
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
        // Đợi TTS hoàn thành rồi gọi callback (đơn giản hoá bằng delay)
        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
            callback()
        }, 2000)
    }

    fun speakNoFlush(text: String, callback: () -> Unit) {
        if (tts == null) {
            pendingSpeakText = text
            pendingSpeakCallback = callback
            initTTS()
            return
        }
        if (!ttsReady) {
            pendingSpeakText = text
            pendingSpeakCallback = callback
            return
        }
        tts?.speak(text, TextToSpeech.QUEUE_ADD, null, null)
        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
            callback()
        }, 2000)
    }
    
    
    fun release() {
        tts?.stop()
        tts?.shutdown()
        tts = null
        ttsReady = false
        pendingSpeakText = null
        pendingSpeakCallback = null
    }

    fun stopSpeaking() {
        try {
            tts?.stop()
        } catch (_: Exception) {}
    }
}
