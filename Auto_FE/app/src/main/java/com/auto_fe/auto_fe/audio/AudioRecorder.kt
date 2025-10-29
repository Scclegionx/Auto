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
    private var recordingFile: File? = null
    
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
            tts = TextToSpeech(context) { status ->
                if (status == TextToSpeech.SUCCESS) {
                    tts?.language = Locale("vi", "VN")
                }
            }
        }
    }
    
    fun speak(text: String) {
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
    }
    
    fun speak(text: String, callback: () -> Unit) {
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
        // Đợi TTS hoàn thành rồi gọi callback
        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
            callback()
        }, 2000) // Đợi 2 giây
    }
    
    
    fun release() {
        tts?.stop()
        tts?.shutdown()
        tts = null
    }
}
