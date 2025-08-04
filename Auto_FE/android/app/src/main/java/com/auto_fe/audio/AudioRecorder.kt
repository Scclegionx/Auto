package com.auto_fe.audio

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.MediaRecorder
import android.widget.Toast
import androidx.core.content.ContextCompat
import java.io.File
import java.io.IOException

class AudioRecorder(private val context: Context) {
    private var mediaRecorder: MediaRecorder? = null
    private var isRecording = false
    private var outputFile: String? = null
    
    fun isRecording(): Boolean = isRecording
    
    fun startRecording() {
        // Kiểm tra quyền ghi âm
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(context, "Cần quyền ghi âm để sử dụng tính năng này", Toast.LENGTH_SHORT).show()
            return
        }
        
        try {
            // Tạo file output
            val dir = File(context.getExternalFilesDir(null), "Recordings")
            if (!dir.exists()) {
                dir.mkdirs()
            }
            
            outputFile = File(dir, "recording_${System.currentTimeMillis()}.mp3").absolutePath
            
            mediaRecorder = MediaRecorder().apply {
                setAudioSource(MediaRecorder.AudioSource.MIC)
                setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
                setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
                setOutputFile(outputFile)
                prepare()
                start()
            }
            
            isRecording = true
            
        } catch (e: IOException) {
            Toast.makeText(context, "Lỗi ghi âm: ${e.message}", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(context, "Lỗi khởi tạo ghi âm: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
    
    fun stopRecording() {
        try {
            mediaRecorder?.apply {
                stop()
                release()
            }
            mediaRecorder = null
            
            isRecording = false
            Toast.makeText(context, "Đã lưu ghi âm: $outputFile", Toast.LENGTH_LONG).show()
            
        } catch (e: Exception) {
            Toast.makeText(context, "Lỗi dừng ghi âm: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
    
    fun release() {
        if (isRecording) {
            stopRecording()
        }
        mediaRecorder?.release()
        mediaRecorder = null
    }
} 