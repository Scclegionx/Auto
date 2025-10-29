package com.auto_fe.auto_fe.usecase

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.automation.msg.SMSAutomation
import com.auto_fe.auto_fe.domain.VoiceEvent
import com.auto_fe.auto_fe.domain.VoiceState
import com.auto_fe.auto_fe.domain.VoiceStateMachine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay

/**
 * State Machine cho luồng gửi SMS tự động
 * Xử lý toàn bộ flow từ nhận lệnh -> xác nhận -> tìm contact -> gửi SMS
 */
class SendSMSStateMachine(
    private val context: Context,
    private val voiceManager: VoiceManager,
    private val smsAutomation: SMSAutomation
) : VoiceStateMachine() {

    private val coroutineScope = CoroutineScope(Dispatchers.Main)

    companion object {
        private const val TAG = "SendSMSStateMachine"
    }

    // Lưu context data trong quá trình xử lý
    private var currentReceiver: String = ""
    private var currentMessage: String = ""
    
    // Callback cho audio level
    var onAudioLevelChanged: ((Int) -> Unit)? = null
    
    // Callback cho transcript
    var onTranscriptUpdated: ((String) -> Unit)? = null

    override fun getNextState(currentState: VoiceState, event: VoiceEvent): VoiceState? {
        return when (currentState) {
            is VoiceState.Idle -> {
                when (event) {
                    is VoiceEvent.StartRecording -> VoiceState.ListeningForSMSCommand
                    is VoiceEvent.StartSMSCommand -> VoiceState.ConfirmingSMSCommand(currentReceiver, currentMessage)
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.ListeningForSMSCommand -> {
                when (event) {
                    is VoiceEvent.SMSCommandParsed -> {
                        currentReceiver = event.receiver
                        currentMessage = event.message
                        VoiceState.ConfirmingSMSCommand(event.receiver, event.message)
                    }
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    is VoiceEvent.SMSCommandParseFailed -> VoiceState.Error(event.reason)
                    else -> null
                }
            }
            is VoiceState.ConfirmingSMSCommand -> {
                when (event) {
                    is VoiceEvent.SMSConfirmed -> VoiceState.ExecutingSMSCommand
                    is VoiceEvent.SMSCancelled -> VoiceState.Idle
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.ExecutingSMSCommand -> {
                when (event) {
                    is VoiceEvent.SMSSentSuccessfully -> VoiceState.Success
                    is VoiceEvent.SMSSendFailed -> VoiceState.Error(event.error)
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Success -> {
                when (event) {
                    is VoiceEvent.SMSSentSuccessfully -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Error -> {
                when (event) {
                    is VoiceEvent.SMSSendFailed -> VoiceState.Idle
                    else -> null
                }
            }
            else -> null
        }
    }

    override fun onEnterState(state: VoiceState, event: VoiceEvent) {
        when (state) {
            is VoiceState.ListeningForSMSCommand -> {
                Log.d(TAG, "Listening for SMS command...")
                listenForSMSCommand()
            }
            is VoiceState.ConfirmingSMSCommand -> {
                Log.d(TAG, "Asking for SMS confirmation: $currentReceiver -> $currentMessage")
                askForConfirmation()
            }
            is VoiceState.ExecutingSMSCommand -> {
                Log.d(TAG, "Executing SMS command: $currentReceiver -> $currentMessage")
                executeSMS()
            }

            is VoiceState.Success -> {
                speak("Đã gửi tin nhắn thành công!")
                // Auto transition to Idle after 2 seconds
                coroutineScope.launch {
                    delay(2000)
                    processEvent(VoiceEvent.SMSSentSuccessfully)
                }
            }

            is VoiceState.Error -> {
                speak("Có lỗi xảy ra khi gửi tin nhắn.")
                // Auto transition to Idle after 3 seconds
                coroutineScope.launch {
                    delay(3000)
                    processEvent(VoiceEvent.SMSSendFailed("Unknown error"))
                }
            }

            is VoiceState.Idle -> {
                if (event is VoiceEvent.UserCancelled || event is VoiceEvent.SMSCancelled) {
                    speak("Đã hủy lệnh gửi tin nhắn.")
                }
                voiceManager.resetBusyState()
                resetContext()
            }
            else -> {
                // Handle other states if needed
            }
        }
    }

    /**
     * Được gọi từ CommandProcessor để thực hiện SMS command
     */
    fun executeSMSCommand(receiver: String, message: String) {
        Log.d(TAG, "SMS command received from CommandProcessor: $receiver -> $message")
        currentReceiver = receiver
        currentMessage = message
        // Chuyển trực tiếp sang ConfirmingSMSCommand state
        processEvent(VoiceEvent.StartSMSCommand)
    }

    /**
     * Lắng nghe lệnh SMS từ người dùng
     */
    private fun listenForSMSCommand() {
        Log.d(TAG, "Listening for SMS command from user...")
        
        val promptText = "Bạn muốn nhắn tin cho ai và nội dung gì?"
        
        // Nói câu hỏi và lắng nghe lệnh SMS
        voiceManager.textToSpeech(promptText, 0, object : VoiceManager.VoiceControllerCallback {
            override fun onSpeechResult(spokenText: String) {
                Log.d(TAG, "Speech result: $spokenText")
                
                // Update transcript
                onTranscriptUpdated?.invoke(spokenText)
                
                // Parse SMS command từ speech text
                parseSMSCommand(spokenText)
            }

            override fun onConfirmationResult(confirmed: Boolean) {}

            override fun onError(error: String) {
                Log.e(TAG, "Speech recognition error: $error")
                processEvent(VoiceEvent.SMSCommandParseFailed("Không nhận dạng được giọng nói: $error"))
            }

            override fun onAudioLevelChanged(level: Int) {
                onAudioLevelChanged?.invoke(level)
            }
        })
    }
    
    /**
     * Phân tích lệnh SMS từ speech text
     */
    private fun parseSMSCommand(spokenText: String) {
        Log.d(TAG, "Parsing SMS command: $spokenText")
        
        // Simple parsing logic
        // Format: "nhắn tin cho [tên] là [nội dung]"
        // hoặc: "gửi tin nhắn cho [tên] nội dung [nội dung]"
        
        val lowerText = spokenText.lowercase()
        
        // Extract receiver and message
        var receiver = ""
        var message = ""
        
        // Try pattern 1: "nhắn tin cho X là Y"
        val pattern1 = Regex("nhắn tin cho (.+?) là (.+)")
        val match1 = pattern1.find(lowerText)
        if (match1 != null) {
            receiver = match1.groupValues[1].trim()
            message = match1.groupValues[2].trim()
        } else {
            // Try pattern 2: "gửi tin nhắn cho X nội dung Y"
            val pattern2 = Regex("gửi tin nhắn cho (.+?) nội dung (.+)")
            val match2 = pattern2.find(lowerText)
            if (match2 != null) {
                receiver = match2.groupValues[1].trim()
                message = match2.groupValues[2].trim()
            } else {
                // Try pattern 3: simple "X [nội dung]"
                val parts = spokenText.split(" ", limit = 2)
                if (parts.size == 2) {
                    receiver = parts[0].trim()
                    message = parts[1].trim()
                }
            }
        }
        
        // Validate
        if (receiver.isNotEmpty() && message.isNotEmpty()) {
            Log.d(TAG, "SMS command parsed: receiver=$receiver, message=$message")
            processEvent(VoiceEvent.SMSCommandParsed(receiver, message))
        } else {
            Log.e(TAG, "Failed to parse SMS command")
            processEvent(VoiceEvent.SMSCommandParseFailed("Không hiểu lệnh. Vui lòng nói rõ người nhận và nội dung tin nhắn."))
        }
    }

    private fun askForConfirmation() {
        Log.d(TAG, "Asking for SMS confirmation")
        
        val confirmText = "Có phải bạn muốn tôi nhắn tin cho $currentReceiver là \"$currentMessage\"?"
        
        // Nói câu hỏi xác nhận và lắng nghe phản hồi
        voiceManager.textToSpeech(confirmText, 1, object : VoiceManager.VoiceControllerCallback {
            override fun onSpeechResult(spokenText: String) {
                Log.d(TAG, "Confirmation response: $spokenText")
                
                // Kiểm tra phản hồi xác nhận
                val confirmed = isConfirmationPositive(spokenText)
                
                if (confirmed) {
                    Log.d(TAG, "User confirmed SMS")
                    processEvent(VoiceEvent.SMSConfirmed)
                } else {
                    Log.d(TAG, "User cancelled SMS")
                    processEvent(VoiceEvent.SMSCancelled)
                }
            }

            override fun onConfirmationResult(confirmed: Boolean) {}

            override fun onError(error: String) {
                Log.e(TAG, "Confirmation error: $error")
                // Nếu lỗi, coi như từ chối
                processEvent(VoiceEvent.SMSCancelled)
            }

            override fun onAudioLevelChanged(level: Int) {
                onAudioLevelChanged?.invoke(level)
            }
        })
    }
    
    private fun isConfirmationPositive(text: String): Boolean {
        val lowerText = text.lowercase()
        return lowerText.contains("có") ||
                lowerText.contains("đúng") ||
                lowerText.contains("yes") ||
                lowerText.contains("ừ") ||
                lowerText.contains("được") ||
                lowerText.contains("ok") ||
                lowerText.contains("chính xác") ||
                lowerText.contains("đúng rồi")
    }

    private fun executeSMS() {
        Log.d(TAG, "Executing SMS")
        
        smsAutomation.sendSMSWithSmartHandling(currentReceiver, currentMessage, object : SMSAutomation.SMSConversationCallback {
            override fun onSuccess() {
                Log.d(TAG, "SMS sent successfully")
                processEvent(VoiceEvent.SMSSentSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "SMS send failed: $error")
                processEvent(VoiceEvent.SMSSendFailed(error))
            }

            override fun onNeedConfirmation(similarContacts: List<String>, originalName: String) {
                // Simplified: just try to send with first similar contact or fail
                if (similarContacts.isNotEmpty()) {
                    val phoneNumber = smsAutomation.findPhoneNumberByName(similarContacts[0])
                    if (phoneNumber.isNotEmpty()) {
                        smsAutomation.sendSMS(phoneNumber, currentMessage, object : SMSAutomation.SMSCallback {
                            override fun onSuccess() {
                                processEvent(VoiceEvent.SMSSentSuccessfully)
                            }
                            override fun onError(error: String) {
                                processEvent(VoiceEvent.SMSSendFailed(error))
                            }
                        })
                    } else {
                        processEvent(VoiceEvent.SMSSendFailed("Không tìm thấy số điện thoại"))
                    }
                } else {
                    processEvent(VoiceEvent.SMSSendFailed("Không tìm thấy liên hệ"))
                }
            }
        })
    }

    private fun speak(text: String) {
        voiceManager.speak(text)
    }

    private fun resetContext() {
        currentReceiver = ""
        currentMessage = ""
    }

    fun cleanup() {
        // Cleanup resources if needed
    }
}