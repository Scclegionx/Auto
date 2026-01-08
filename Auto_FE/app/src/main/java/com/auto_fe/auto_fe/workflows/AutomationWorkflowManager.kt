package com.auto_fe.auto_fe.workflows

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.audio.STTManager
import com.auto_fe.auto_fe.audio.TTSManager
import com.auto_fe.auto_fe.base.AutomationState
import com.auto_fe.auto_fe.base.AutomationTask
import com.auto_fe.auto_fe.base.ConfirmationRequirement
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
                    } catch (e: ConfirmationRequirement) {
                        // Xử lý yêu cầu xác nhận
                        Log.d(TAG, "Confirmation required: ${e.confirmationQuestion}")
                        // Chuyển sang Confirmation state để UI hiển thị
                        currentState = AutomationState.Confirmation(e.confirmationQuestion)
                        onStateChanged?.invoke(currentState)
                        handleConfirmation(e)
                    } catch (e: Exception) {
                        Log.e(TAG, "Error processing task: ${e.message}", e)
                        currentState = AutomationState.Error(e.message ?: "Lỗi hệ thống")
                    }
                }
                
                is AutomationState.Confirmation -> {
                    // State này chỉ để UI hiển thị, logic xử lý đã ở handleConfirmation()
                    // Không cần làm gì ở đây vì handleConfirmation() đã xử lý xong
                    Log.d(TAG, "Confirmation state: ${state.question}")
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
     * Xử lý yêu cầu xác nhận từ người dùng
     */
    private suspend fun handleConfirmation(requirement: ConfirmationRequirement) {
        // 1. Nói câu hỏi xác nhận
        Log.d(TAG, "Asking confirmation: ${requirement.confirmationQuestion}")
        ttsManager.speak(requirement.confirmationQuestion)
        delay(2000) // Đợi TTS hoàn thành
        
        // 2. Lắng nghe phản hồi từ người dùng
        Log.d(TAG, "Listening for confirmation response...")
        val response = listenToUser()
        Log.d(TAG, "User response: $response")
        
        // 3. Phân tích phản hồi (có/đúng/rồi vs không/sai)
        // QUAN TRỌNG: Kiểm tra từ phủ định TRƯỚC để tránh "không phải" bị match với "phải"
        val normalizedResponse = response.lowercase().trim()
        val isConfirmed = when {
            // Kiểm tra từ phủ định TRƯỚC
            normalizedResponse.contains("không phải") ||
            normalizedResponse.contains("không") || 
            normalizedResponse == "không" -> {
                Log.d(TAG, "User declined confirmation")
                false
            }
            // Sau đó mới kiểm tra từ xác nhận
            normalizedResponse.contains("có") || 
            normalizedResponse.contains("đúng") || 
            normalizedResponse.contains("đúng rồi") || 
            normalizedResponse == "có" ||
            normalizedResponse == "đúng" ||
            normalizedResponse == "đúng rồi" -> {
                Log.d(TAG, "User confirmed")
                true
            }
            else -> {
                // Mặc định là không xác nhận nếu không rõ ràng
                Log.w(TAG, "Unclear response: '$normalizedResponse', defaulting to not confirmed")
                false
            }
        }
        
        if (isConfirmed) {
            // 4. Nếu xác nhận, thực thi callback
            Log.d(TAG, "User confirmed, executing action...")
            try {
                val result = requirement.onConfirmed()
                currentState = AutomationState.Success(result)
            } catch (e: Exception) {
                Log.e(TAG, "Error executing confirmed action: ${e.message}", e)
                currentState = AutomationState.Error(e.message ?: "Lỗi thực thi")
            }
        } else {
            // 5. Nếu không xác nhận, hủy luồng
            Log.d(TAG, "User declined, cancelling workflow")
            currentState = AutomationState.Success("Dạ, con đã hủy lệnh rồi ạ.")
        }
    }
    
    /**
     * Lấy state hiện tại
     */
    fun getCurrentState(): AutomationState {
        return currentState
    }
}

