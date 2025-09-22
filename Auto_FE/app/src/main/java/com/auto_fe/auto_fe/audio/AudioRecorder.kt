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
    
    interface AudioRecorderCallback {
        fun onRecordingComplete(audioFile: File)
        fun onError(error: String)
    }
    
    init {
        initTTS()
    }
    
    private fun initTTS() {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale("vi", "VN")
            }
        }
    }
    
    fun speak(text: String) {
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
    }
    
    fun startRecording(callback: AudioRecorderCallback) {
        if (isRecording) return
        
        try {
            val sampleRate = 44100
            val channelConfig = AudioFormat.CHANNEL_IN_MONO
            val audioFormat = AudioFormat.ENCODING_PCM_16BIT
            
            val bufferSize = AudioRecord.getMinBufferSize(
                sampleRate, channelConfig, audioFormat
            )
            
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                channelConfig,
                audioFormat,
                bufferSize
            )
            
            recordingFile = File(context.cacheDir, "recording_${System.currentTimeMillis()}.wav")
            
            audioRecord?.startRecording()
            isRecording = true
            
            // Bắt đầu ghi âm trong background thread
            Thread {
                val buffer = ByteArray(bufferSize)
                val outputStream = FileOutputStream(recordingFile)
                
                try {
                    while (isRecording) {
                        val bytesRead = audioRecord?.read(buffer, 0, bufferSize) ?: 0
                        if (bytesRead > 0) {
                            outputStream.write(buffer, 0, bytesRead)
                        }
                    }
                } catch (e: IOException) {
                    callback.onError("Lỗi khi ghi âm: ${e.message}")
                } finally {
                    outputStream.close()
                }
            }.start()
            
        } catch (e: Exception) {
            callback.onError("Không thể bắt đầu ghi âm: ${e.message}")
        }
    }
    
    fun stopRecording(callback: AudioRecorderCallback) {
        if (!isRecording) return
        
        isRecording = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        
        recordingFile?.let { file ->
            if (file.exists() && file.length() > 0) {
                callback.onRecordingComplete(file)
            } else {
                callback.onError("File ghi âm không hợp lệ")
            }
        }
    }
    
    fun release() {
        tts?.stop()
        tts?.shutdown()
        tts = null
        
        if (isRecording) {
            stopRecording(object : AudioRecorderCallback {
                override fun onRecordingComplete(audioFile: File) {}
                override fun onError(error: String) {}
            })
        }
    }
}
