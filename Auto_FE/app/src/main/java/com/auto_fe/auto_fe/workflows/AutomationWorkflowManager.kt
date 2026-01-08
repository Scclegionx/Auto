package com.auto_fe.auto_fe.workflows

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.audio.STTManager
import com.auto_fe.auto_fe.audio.TTSManager
import com.auto_fe.auto_fe.base.AutomationState
import com.auto_fe.auto_fe.base.AutomationTask
import com.auto_fe.auto_fe.base.ConfirmationRequirement
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
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
        ttsManager.speakAndAwait(requirement.confirmationQuestion)
        
        // 2. Lắng nghe phản hồi từ người dùng
        Log.d(TAG, "Listening for confirmation response...")
        val response = withContext(Dispatchers.Main) {
            listenToUser()
        }
        Log.d(TAG, "User response: $response")
        
        // 3. Xử lý theo loại xác nhận
        try {
            when {
                requirement.actionType == "contact_add_name" -> {
                    // Bước 1: Thu thập tên liên hệ
                    handleContactAddName(requirement, response)
                }
                requirement.actionType == "contact_add_phone" -> {
                    // Bước 2: Thu thập số điện thoại và mở màn hình thêm liên hệ
                    handleContactAddPhone(requirement, response)
                }
                requirement.isMultipleContacts -> {
                    // Trường hợp nhiều liên hệ (>1): Người dùng sẽ nói tên
                    handleMultipleContactsConfirmation(requirement, response)
                }
                else -> {
                    // Trường hợp 1 liên hệ: Phân tích phản hồi (có/đúng/rồi vs không/sai/không phải)
                    handleSingleContactConfirmation(requirement, response)
                }
            }
        } catch (e: ConfirmationRequirement) {
            // Nếu có ConfirmationRequirement mới được ném (ví dụ từ handleContactAddName),
            // cập nhật state và xử lý lại
            Log.d(TAG, "New confirmation required: ${e.confirmationQuestion}")
            currentState = AutomationState.Confirmation(e.confirmationQuestion)
            onStateChanged?.invoke(currentState)
            handleConfirmation(e)
        }
    }
    
    /**
     * Xử lý xác nhận khi có nhiều liên hệ - người dùng sẽ nói tên
     */
    private suspend fun handleMultipleContactsConfirmation(requirement: ConfirmationRequirement, response: String) {
        Log.d(TAG, "Handling multiple contacts confirmation, user said: $response")
        Log.d(TAG, "Action type: ${requirement.actionType}, Action data: ${requirement.actionData}")
        
        val normalizedResponse = response.lowercase().trim()
        
        // Kiểm tra xem người dùng có muốn hủy lệnh không
        val isCancelCommand = normalizedResponse.contains("hủy") ||
                              normalizedResponse.contains("huy") ||
                              normalizedResponse.contains("thôi") ||
                              normalizedResponse.contains("không") ||
                              normalizedResponse.contains("dừng") ||
                              normalizedResponse.contains("stop") ||
                              normalizedResponse == "không" ||
                              normalizedResponse == "thôi" ||
                              normalizedResponse == "hủy" ||
                              normalizedResponse.contains("không gửi") ||
                              normalizedResponse.contains("không gọi") ||
                              normalizedResponse.contains("thôi không")
        
        if (isCancelCommand) {
            // Người dùng muốn hủy lệnh
            Log.d(TAG, "User cancelled the command")
            currentState = AutomationState.Success("Dạ, con đã hủy lệnh rồi ạ.")
            return
        }
        
        // Kiểm tra setting hỗ trợ nói
        val settingsManager = com.auto_fe.auto_fe.utils.common.SettingsManager(context)
        val isSupportSpeakEnabled = settingsManager.isSupportSpeakEnabled()
        
        // Tìm exact match theo tên người dùng nói
        val contactName = response.trim()
        val exactMatch = com.auto_fe.auto_fe.utils.nlp.ContactUtils.smartFindContact(context, contactName)
        
        if (exactMatch != null) {
            // Tìm thấy exact match, tiếp tục luồng
            Log.d(TAG, "Found exact match: ${exactMatch.first} with phone: ${exactMatch.second}")
            try {
                val result = when (requirement.actionType) {
                    "sms" -> {
                        val smsAutomation = com.auto_fe.auto_fe.automation.msg.SMSAutomation(context)
                        if (isSupportSpeakEnabled) {
                            // Bật hỗ trợ nói: Gửi trực tiếp
                            smsAutomation.sendSMSDirect(exactMatch.first, requirement.actionData)
                        } else {
                            // Tắt hỗ trợ nói: Mở hộp soạn tin
                            smsAutomation.openSmsComposeDirect(exactMatch.first, requirement.actionData)
                        }
                    }
                    "phone" -> {
                        val phoneAutomation = com.auto_fe.auto_fe.automation.phone.PhoneAutomation(context)
                        if (isSupportSpeakEnabled) {
                            // Bật hỗ trợ nói: Gọi trực tiếp
                            phoneAutomation.makeCallDirect(exactMatch.first, requirement.actionData)
                        } else {
                            // Tắt hỗ trợ nói: Mở màn hình quay số
                            phoneAutomation.makeCall(exactMatch.first, requirement.actionData, requireConfirmation = false)
                        }
                    }
                    else -> {
                        throw Exception("Unknown action type: ${requirement.actionType}")
                    }
                }
                currentState = AutomationState.Success(result)
            } catch (e: Exception) {
                Log.e(TAG, "Error executing action: ${e.message}", e)
                val errorMessage = when (requirement.actionType) {
                    "sms" -> "Dạ, con không thể gửi tin nhắn ạ."
                    "phone" -> "Dạ, con không thể gọi điện ạ."
                    else -> e.message ?: "Lỗi thực thi"
                }
                currentState = AutomationState.Error(errorMessage)
            }
        } else {
            // Không tìm thấy exact match - thông báo rõ ràng với tên cụ thể
            Log.d(TAG, "No exact match found for: $contactName")
            val errorMessage = when (requirement.actionType) {
                "sms" -> "Dạ, con không tìm thấy $contactName trong danh bạ ạ. Bác vui lòng xem hướng dẫn thêm liên hệ tự động, sau đó hãy thử lại lệnh gửi tin nhắn nhé."
                "phone" -> "Dạ, con không tìm thấy $contactName trong danh bạ ạ. Bác vui lòng xem hướng dẫn thêm liên hệ tự động, sau đó hãy thử lại lệnh gọi điện nhé."
                else -> "Dạ, con không tìm thấy $contactName trong danh bạ ạ."
            }
            currentState = AutomationState.Error(errorMessage)
        }
    }
    
    /**
     * Xử lý xác nhận khi có 1 liên hệ - người dùng sẽ nói có/không
     */
    private suspend fun handleSingleContactConfirmation(requirement: ConfirmationRequirement, response: String) {
        // Phân tích phản hồi (có/đúng/rồi vs không/sai)
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
            // Nếu xác nhận, thực thi callback
            Log.d(TAG, "User confirmed, executing action...")
            try {
                val result = requirement.onConfirmed()
                currentState = AutomationState.Success(result)
            } catch (e: Exception) {
                Log.e(TAG, "Error executing confirmed action: ${e.message}", e)
                currentState = AutomationState.Error(e.message ?: "Lỗi thực thi")
            }
        } else {
            // Nếu không xác nhận, hủy luồng
            Log.d(TAG, "User declined, cancelling workflow")
            currentState = AutomationState.Success("Dạ, con đã hủy lệnh rồi ạ.")
        }
    }
    
    /**
     * Xử lý bước 1: Thu thập tên liên hệ
     */
    private suspend fun handleContactAddName(requirement: ConfirmationRequirement, response: String) {
        Log.d(TAG, "Handling contact add name, user said: $response")
        
        val contactName = response.trim()
        
        if (contactName.isEmpty()) {
            currentState = AutomationState.Error("Dạ, con chưa nghe rõ tên liên hệ ạ. Bác vui lòng nói lại nhé.")
            return
        }
        
        // Kiểm tra xem người dùng có muốn hủy không
        val normalizedResponse = contactName.lowercase()
        val isCancelCommand = normalizedResponse.contains("hủy") ||
                              normalizedResponse.contains("huy") ||
                              normalizedResponse.contains("thôi") ||
                              normalizedResponse.contains("dừng") ||
                              normalizedResponse == "thôi" ||
                              normalizedResponse == "hủy"
        
        if (isCancelCommand) {
            Log.d(TAG, "User cancelled contact add")
            currentState = AutomationState.Success("Dạ, con đã hủy lệnh rồi ạ.")
            return
        }
        
        // Lưu tên vào actionData và ném ConfirmationRequirement mới để hỏi số điện thoại
        // Điều này sẽ cập nhật state và UI sẽ hiển thị câu hỏi mới
        Log.d(TAG, "Contact name collected: $contactName, asking for phone number")
        throw ConfirmationRequirement(
            originalInput = "",
            confirmationQuestion = "Dạ, bác hãy nói số điện thoại ạ.",
            onConfirmed = {
                throw Exception("Contact phone collection - should be handled in workflow")
            },
            isMultipleContacts = false,
            actionType = "contact_add_phone",
            actionData = contactName // Lưu tên vào actionData
        )
    }
    
    /**
     * Xử lý bước 2: Thu thập số điện thoại và mở màn hình thêm liên hệ
     */
    private suspend fun handleContactAddPhone(requirement: ConfirmationRequirement, response: String) {
        Log.d(TAG, "Handling contact add phone, user said: $response")
        
        val phoneNumber = response.trim().replace(Regex("[^0-9+]"), "") // Chỉ giữ số và dấu +
        val contactName = requirement.actionData // Tên đã lưu từ bước trước
        
        if (phoneNumber.isEmpty()) {
            currentState = AutomationState.Error("Dạ, con chưa nghe rõ số điện thoại ạ. Bác vui lòng nói lại nhé.")
            return
        }
        
        // Kiểm tra xem người dùng có muốn hủy không
        val normalizedResponse = response.lowercase().trim()
        val isCancelCommand = normalizedResponse.contains("hủy") ||
                              normalizedResponse.contains("huy") ||
                              normalizedResponse.contains("thôi") ||
                              normalizedResponse.contains("dừng") ||
                              normalizedResponse == "thôi" ||
                              normalizedResponse == "hủy"
        
        if (isCancelCommand) {
            Log.d(TAG, "User cancelled contact add")
            currentState = AutomationState.Success("Dạ, con đã hủy lệnh rồi ạ.")
            return
        }
        
        // Mở màn hình thêm liên hệ với tên và số đã điền
        try {
            val contactAutomation = com.auto_fe.auto_fe.automation.phone.ContactAutomation(context)
            val result = contactAutomation.insertContactWithData(contactName, phoneNumber)
            currentState = AutomationState.Success(result)
        } catch (e: Exception) {
            Log.e(TAG, "Error opening contact add screen: ${e.message}", e)
            currentState = AutomationState.Error(e.message ?: "Dạ, con không thể mở màn hình thêm danh bạ ạ.")
        }
    }
    
    /**
     * Lấy state hiện tại
     */
    fun getCurrentState(): AutomationState {
        return currentState
    }
}

