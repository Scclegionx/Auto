package com.auto_fe.auto_fe.workflows

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.audio.STTManager
import com.auto_fe.auto_fe.audio.TTSManager
import com.auto_fe.auto_fe.base.AutomationState
import com.auto_fe.auto_fe.base.AutomationTask
import kotlinx.coroutines.delay
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

class AutomationWorkflowManager(
    private val context: Context,
    private val task: AutomationTask,
    private val greetingText: String = "Bạn cần tôi trợ giúp điều gì?"
) {
    companion object {
        private const val TAG = "AutomationWorkflowManager"
    }
    
    private val ttsManager = TTSManager.getInstance(context)
    private val sttManager = STTManager.getInstance(context)
    
    private var currentState: AutomationState = AutomationState.Speaking(greetingText)
    
    // Callbacks cho UI
    var onStateChanged: ((AutomationState) -> Unit)? = null
    var onAudioLevelChanged: ((Int) -> Unit)? = null
    
    /**
     * Bắt đầu workflow
     */
    suspend fun start() {
        Log.d(TAG, "Starting automation workflow")
        
        while (true) {
            when (val state = currentState) {
                is AutomationState.Speaking -> {
                    Log.d(TAG, "App nói: ${state.text}")
                    onStateChanged?.invoke(state)
                    
                    // Nói text
                    ttsManager.speak(state.text)
                    
                    // Đợi một chút để TTS hoàn thành (ước tính 2s)
                    delay(2000)
                    
                    // Chuyển sang Listening
                    currentState = AutomationState.Listening
                }
                
                is AutomationState.Listening -> {
                    Log.d(TAG, "App đang lắng nghe...")
                    onStateChanged?.invoke(state)
                    
                    // Bắt đầu nhận diện giọng nói
                    val voiceText = listenToUser()
                    
                    // Chuyển sang Processing với text đã nhận diện
                    currentState = AutomationState.Processing(voiceText)
                }
                
                is AutomationState.Processing -> {
                    Log.d(TAG, "Đang xử lý logic cho: ${state.rawInput}")
                    onStateChanged?.invoke(state)
                    
                    try {
                        // Manager "treo" ở đây đợi Task xử lý xong
                        val result = task.execute(state.rawInput)
                        
                        // Chuyển sang Success
                        currentState = AutomationState.Success(result)
                    } catch (e: Exception) {
                        Log.e(TAG, "Error processing task: ${e.message}", e)
                        currentState = AutomationState.Error(e.message ?: "Lỗi hệ thống")
                    }
                }
                
                is AutomationState.Success -> {
                    Log.d(TAG, "Xong: ${state.message}")
                    onStateChanged?.invoke(state)
                    
                    // Nói thông báo thành công
                    ttsManager.speak(state.message)
                    
                    // Đợi một chút rồi kết thúc
                    delay(2000)
                    break
                }
                
                is AutomationState.Error -> {
                    Log.e(TAG, "Thất bại: ${state.message}")
                    onStateChanged?.invoke(state)
                    
                    // Nói thông báo lỗi
                    ttsManager.speak(state.message)
                    
                    // Đợi một chút rồi kết thúc
                    delay(2000)
                    break
                }
            }
        }
        
        Log.d(TAG, "Workflow completed")
    }
    
    /**
     * Lắng nghe người dùng nói và trả về text
     * Sử dụng STTManager để nhận diện giọng nói
     */
    private suspend fun listenToUser(): String = suspendCancellableCoroutine { continuation ->
        continuation.invokeOnCancellation {
            sttManager.cancelListening()
        }
        sttManager.startListening(object : STTManager.STTCallback {
            override fun onSpeechResult(spokenText: String) {
                Log.d(TAG, "Speech recognition result: $spokenText")
                continuation.resume(spokenText)
            }
            
            override fun onError(error: String) {
                Log.e(TAG, "Speech recognition error: $error")
                continuation.resumeWithException(Exception(error))
            }
            
            override fun onAudioLevelChanged(level: Int) {
                onAudioLevelChanged?.invoke(level)
            }
        })
    }
    
    /**
     * Dừng workflow (cancel)
     */
    fun cancel() {
        Log.d(TAG, "Cancelling workflow")
        sttManager.cancelListening()
        ttsManager.stopSpeaking()
        currentState = AutomationState.Error("Đã hủy")
        onStateChanged?.invoke(currentState)
    }
    
    /**
     * Lấy state hiện tại
     */
    fun getCurrentState(): AutomationState {
        return currentState
    }
}

