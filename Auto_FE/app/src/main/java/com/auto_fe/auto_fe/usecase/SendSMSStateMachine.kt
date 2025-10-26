package com.auto_fe.auto_fe.usecase

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.automation.msg.SMSAutomation
import com.auto_fe.auto_fe.domain.VoiceEvent
import com.auto_fe.auto_fe.domain.VoiceState
import com.auto_fe.auto_fe.domain.VoiceStateMachine
import com.auto_fe.auto_fe.core.CommandProcessor

/**
 * State Machine cho luồng gửi SMS tự động
 * Xử lý toàn bộ flow từ nhận lệnh -> xác nhận -> tìm contact -> gửi SMS
 */
class SendSMSStateMachine(
    private val context: Context,
    private val voiceManager: VoiceManager,
    private val smsAutomation: SMSAutomation
) : VoiceStateMachine() {
    
    private val commandProcessor = CommandProcessor(context)

    companion object {
        private const val TAG = "SendSMSStateMachine"
        private const val MAX_RETRY_COUNT = 2
    }

    // Lưu context data trong quá trình xử lý
    private var currentReceiver: String = ""
    private var currentMessage: String = ""
    private var similarContacts: List<String> = emptyList()
    
    // Callback cho audio level
    var onAudioLevelChanged: ((Int) -> Unit)? = null
    
    // Callback cho transcript
    var onTranscriptUpdated: ((String) -> Unit)? = null

    /**
     * Định nghĩa State Transitions
     */
    override fun getNextState(currentState: VoiceState, event: VoiceEvent): VoiceState? {
        return when (currentState) {

            // IDLE -> Start listening
            is VoiceState.Idle -> {
                when (event) {
                    is VoiceEvent.StartRecording -> VoiceState.ListeningForSMSCommand
                    is VoiceEvent.UserCancelled -> VoiceState.Idle // Already idle
                    else -> null
                }
            }

            // LISTENING -> Parse command
            is VoiceState.ListeningForSMSCommand -> {
                when (event) {
                    is VoiceEvent.SMSCommandReceived -> {
                        VoiceState.ParsingSMSCommand
                    }
                    is VoiceEvent.SpeechRecognitionFailed -> {
                        VoiceState.Error("Hiện tại tôi không hỗ trợ lệnh này. Vui lòng vào tab 'Hướng dẫn' để xem các lệnh được hỗ trợ.")
                    }
                    is VoiceEvent.Timeout -> {
                        VoiceState.Error("Không nhận được phản hồi.")
                    }
                    is VoiceEvent.UserCancelled -> {
                        VoiceState.Idle
                    }
                    else -> null
                }
            }

            // PARSING -> Confirm with user
            is VoiceState.ParsingSMSCommand -> {
                when (event) {
                    is VoiceEvent.SMSCommandParsed -> {
                        // Lưu data
                        currentReceiver = event.receiver
                        currentMessage = event.message
                        VoiceState.ConfirmingSMSCommand(event.receiver, event.message)
                    }
                    is VoiceEvent.SMSCommandParseFailed -> {
                        VoiceState.Error(event.reason)
                    }
                    is VoiceEvent.UserCancelled -> {
                        VoiceState.Idle
                    }
                    else -> null
                }
            }

            // CONFIRMING -> Wait for user confirmation
            is VoiceState.ConfirmingSMSCommand -> {
                when (event) {
                    is VoiceEvent.UserConfirmed -> {
                        if (event.confirmed) {
                            VoiceState.WaitingForUserConfirmation
                        } else {
                            VoiceState.Error("Tôi không thực hiện gửi tin nhắn.")
                        }
                    }
                    is VoiceEvent.Timeout -> {
                        VoiceState.Error("Tôi không thực hiện gửi tin nhắn.")
                    }
                    is VoiceEvent.UserCancelled -> {
                        VoiceState.Idle
                    }
                    else -> null
                }
            }

            // WAITING FOR CONFIRMATION -> Search contact
            is VoiceState.WaitingForUserConfirmation -> {
                when (event) {
                    is VoiceEvent.UserConfirmed -> {
                        if (event.confirmed) {
                            VoiceState.SearchingContact(currentReceiver, currentMessage)
                        } else {
                            VoiceState.Error("Tôi không thực hiện gửi tin nhắn.")
                        }
                    }
                    is VoiceEvent.PermissionError -> {
                        VoiceState.Error("Ứng dụng cần được cấp quyền truy cập danh bạ và gửi tin nhắn để tiếp tục.")
                    }
                    is VoiceEvent.Timeout -> {
                        VoiceState.Error("Tôi không thực hiện gửi tin nhắn.")
                    }
                    is VoiceEvent.UserCancelled -> {
                        VoiceState.Idle
                    }
                    else -> null
                }
            }

            // SEARCHING CONTACT -> Handle search results
            is VoiceState.SearchingContact -> {
                when (event) {
                    is VoiceEvent.ExactContactFound -> {
                        VoiceState.SendingSMS(
                            phoneNumber = event.phoneNumber,
                            message = currentState.message,
                            contactName = event.contactName
                        )
                    }
                    is VoiceEvent.SimilarContactsFound -> {
                        similarContacts = event.similarContacts
                        VoiceState.SuggestingSimilarContacts(
                            originalName = event.originalName,
                            similarContacts = event.similarContacts,
                            message = currentState.message,
                            retryCount = 0
                        )
                    }
                    is VoiceEvent.NoContactFound -> {
                        VoiceState.Error("Không tìm thấy người này trong danh bạ")
                    }
                    is VoiceEvent.UserCancelled -> {
                        VoiceState.Idle
                    }
                    else -> null
                }
            }

            // SUGGESTING SIMILAR -> Wait for new name or decline
            is VoiceState.SuggestingSimilarContacts -> {
                when (event) {
                    is VoiceEvent.NewContactNameProvided -> {
                        // Update receiver
                        currentReceiver = event.newName

                        // Check retry count
                        val newRetryCount = currentState.retryCount + 1
                        if (newRetryCount > MAX_RETRY_COUNT) {
                            VoiceState.Error("Tôi vẫn không tìm thấy liên hệ đó.")
                        } else {
                            VoiceState.SearchingContact(event.newName, currentState.message)
                        }
                    }
                    is VoiceEvent.UserDeclinedRetry -> {
                        VoiceState.Error("Tôi không thực hiện gửi tin nhắn.")
                    }
                    is VoiceEvent.Timeout -> {
                        VoiceState.Error("Tôi không thực hiện gửi tin nhắn.")
                    }
                    is VoiceEvent.UserCancelled -> {
                        VoiceState.Idle
                    }
                    else -> null
                }
            }

            // WAITING FOR NEW NAME -> Parse new name
            is VoiceState.WaitingForNewContactName -> {
                when (event) {
                    is VoiceEvent.NewContactNameProvided -> {
                        currentReceiver = event.newName
                        VoiceState.SearchingContact(event.newName, currentMessage)
                    }
                    is VoiceEvent.UserDeclinedRetry -> {
                        VoiceState.Error("Tôi không thực hiện gửi tin nhắn.")
                    }
                    is VoiceEvent.UserCancelled -> {
                        VoiceState.Idle
                    }
                    else -> null
                }
            }

            // SENDING SMS -> Handle result
            is VoiceState.SendingSMS -> {
                when (event) {
                    is VoiceEvent.SMSSentSuccessfully -> {
                        VoiceState.Success
                    }
                    is VoiceEvent.SMSSendFailed -> {
                        VoiceState.Error("Gửi tin nhắn thất bại, vui lòng thử lại sau.")
                    }
                    is VoiceEvent.UserCancelled -> {
                        VoiceState.Idle
                    }
                    else -> null
                }
            }

            // Terminal states - no more transitions
            is VoiceState.Success, is VoiceState.Error -> null

            else -> null
        }
    }

    /**
     * Side Effects khi enter state
     */
    override fun onEnterState(state: VoiceState, event: VoiceEvent) {
        Log.d(TAG, "onEnterState: ${state.getName()}")

        when (state) {
            is VoiceState.ListeningForSMSCommand -> {
                // Reset VoiceManager trước khi bắt đầu
                voiceManager.resetBusyState()
                // Phát câu hỏi và bắt đầu lắng nghe
                speakAndListen("Bạn cần tôi trợ giúp điều gì?")
            }

            is VoiceState.ParsingSMSCommand -> {
                // Parse command từ event
                if (event is VoiceEvent.SMSCommandReceived) {
                    parseCommandAsync(event.rawCommand)
                }
            }

            is VoiceState.ConfirmingSMSCommand -> {
                // Xác nhận lệnh với người dùng
                val confirmText = "Có phải bạn muốn tôi nhắn tin cho ${state.receiver} là '${state.message}'?"
                speakAndListen(confirmText)
            }

            is VoiceState.WaitingForUserConfirmation -> {
                // Chờ xác nhận - đã được xử lý ở ConfirmingSMSCommand
                // State này chỉ để tách biệt logic
            }

            is VoiceState.SearchingContact -> {
                // Tìm kiếm contact
                searchContactAsync(state.contactName, state.message)
            }

            is VoiceState.SuggestingSimilarContacts -> {
                // Gợi ý contact tương tự
                val contactList = state.similarContacts.joinToString(" và ")
                val suggestText = "Không tìm thấy danh bạ ${state.originalName} nhưng tìm được ${state.similarContacts.size} danh bạ có tên gần giống là $contactList. Liệu bạn có nhầm lẫn tên người gửi không? Nếu nhầm lẫn bạn hãy nói lại tên"
                speakAndListen(suggestText)
            }

            is VoiceState.WaitingForNewContactName -> {
                // Đã speak ở SuggestingSimilarContacts, chỉ cần chờ
            }

            is VoiceState.SendingSMS -> {
                // Gửi SMS
                sendSMSAsync(state.phoneNumber, state.message, state.contactName)
            }

            is VoiceState.Success -> {
                speak("Đã gửi tin nhắn.")
            }

            is VoiceState.Error -> {
                speak(state.errorMessage)
            }

            is VoiceState.Idle -> {
                // Reset state when cancelled
                if (event is VoiceEvent.UserCancelled) {
                    speak("Đã hủy ghi âm")
                    // Reset all context data
                    currentReceiver = ""
                    currentMessage = ""
                    similarContacts = emptyList()
                    // Reset VoiceManager busy state
                    voiceManager.resetBusyState()
                }
            }

            else -> {
                // Do nothing
            }
        }
    }

    // ========== HELPER METHODS ==========

    /**
     * Phát giọng nói và bắt đầu lắng nghe
     */
    private fun speakAndListen(text: String, delaySeconds: Int = 2) {
        // Sử dụng API mới với delay tùy chỉnh
        voiceManager.textToSpeech(text, delaySeconds, object : VoiceManager.VoiceControllerCallback {
            override fun onSpeechResult(spokenText: String) {
                handleSpeechResult(spokenText)
            }

            override fun onConfirmationResult(confirmed: Boolean) {}

            override fun onError(error: String) {
                processEvent(VoiceEvent.SpeechRecognitionFailed)
            }

            override fun onAudioLevelChanged(level: Int) {
                // Forward audio level to UI
                onAudioLevelChanged?.invoke(level)
            }
        })
    }

    /**
     * Chỉ phát giọng nói
     */
    private fun speak(text: String) {
        voiceManager.speak(text)
    }


    /**
     * Xử lý kết quả speech recognition dựa vào state hiện tại
     */
    private fun handleSpeechResult(spokenText: String) {
        try {
            Log.d(TAG, "Handling speech result: $spokenText")
            
            // Update transcript in popup
            onTranscriptUpdated?.invoke(spokenText)
            
            when (currentState) {
                is VoiceState.ListeningForSMSCommand -> {
                    processEvent(VoiceEvent.SMSCommandReceived(spokenText))
                }

                is VoiceState.ConfirmingSMSCommand,
                is VoiceState.WaitingForUserConfirmation -> {
                    val confirmed = isConfirmationPositive(spokenText)
                    processEvent(VoiceEvent.UserConfirmed(confirmed))
                }

                is VoiceState.SuggestingSimilarContacts,
                is VoiceState.WaitingForNewContactName -> {
                    // Check if user declined
                    if (isDeclined(spokenText)) {
                        processEvent(VoiceEvent.UserDeclinedRetry)
                    } else {
                        processEvent(VoiceEvent.NewContactNameProvided(spokenText.trim()))
                    }
                }

                else -> {
                    Log.w(TAG, "Unexpected speech result in state: ${currentState.getName()}")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in handleSpeechResult: ${e.message}")
            processEvent(VoiceEvent.SpeechRecognitionFailed)
        }
    }

    /**
     * Parse SMS command using NLP service
     */
    private fun parseCommandAsync(command: String) {
        Log.d(TAG, "Parsing command with NLP: $command")
        
        try {
            commandProcessor.processCommand(command, object : CommandProcessor.CommandProcessorCallback {
                override fun onCommandExecuted(success: Boolean, message: String) {
                    try {
                        if (success) {
                            Log.d(TAG, "Command executed successfully: $message")
                            processEvent(VoiceEvent.SMSCommandParsed("", "")) // CommandProcessor handles the actual execution
                        } else {
                            Log.e(TAG, "Command execution failed: $message")
                            processEvent(VoiceEvent.SMSCommandParseFailed(message))
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Error in onCommandExecuted: ${e.message}")
                        processEvent(VoiceEvent.SMSCommandParseFailed("Lỗi xử lý lệnh: ${e.message}"))
                    }
                }

                override fun onError(error: String) {
                    try {
                        Log.e(TAG, "NLP Error: $error")
                        // Phân biệt lỗi hệ thống vs lỗi không hỗ trợ lệnh
                        if (error.contains("Lỗi NLP") || error.contains("Server error") || error.contains("Connection")) {
                            processEvent(VoiceEvent.SMSCommandParseFailed("Lỗi hệ thống: $error"))
                        } else {
                            processEvent(VoiceEvent.SMSCommandParseFailed("Hiện tại tôi không hỗ trợ lệnh này. Vui lòng vào tab 'Hướng dẫn' để xem các lệnh được hỗ trợ."))
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Error in onError: ${e.message}")
                        processEvent(VoiceEvent.SMSCommandParseFailed("Lỗi xử lý: ${e.message}"))
                    }
                }

                override fun onNeedConfirmation(command: String, receiver: String, message: String) {
                    try {
                        Log.d(TAG, "Need confirmation: $command -> $receiver: $message")
                        // Lưu thông tin để xác nhận
                        currentReceiver = receiver
                        currentMessage = message
                        processEvent(VoiceEvent.SMSCommandParsed(receiver, message))
                    } catch (e: Exception) {
                        Log.e(TAG, "Error in onNeedConfirmation: ${e.message}")
                        processEvent(VoiceEvent.SMSCommandParseFailed("Lỗi xác nhận: ${e.message}"))
                    }
                }
            })
        } catch (e: Exception) {
            Log.e(TAG, "Error in parseCommandAsync: ${e.message}")
            processEvent(VoiceEvent.SMSCommandParseFailed("Lỗi phân tích lệnh: ${e.message}"))
        }
    }

    /**
     * Tìm kiếm contact
     */
    private fun searchContactAsync(contactName: String, message: String) {
        smsAutomation.sendSMSWithSmartHandling(contactName, message, object : SMSAutomation.SMSConversationCallback {
            override fun onSuccess() {
                // Sẽ không vào đây vì sendSMSWithSmartHandling không gửi trực tiếp
            }

            override fun onError(error: String) {
                if (error.contains("Không tìm thấy người này trong danh bạ")) {
                    processEvent(VoiceEvent.NoContactFound(contactName))
                } else {
                    processEvent(VoiceEvent.SMSSendFailed(error))
                }
            }

            override fun onNeedConfirmation(similarContacts: List<String>, originalName: String) {
                // Kiểm tra nếu chỉ có 1 contact và tên khớp 100%
                if (similarContacts.size == 1 &&
                    similarContacts[0].equals(originalName, ignoreCase = true)) {
                    // Exact match - get phone number and send
                    val phoneNumber = smsAutomation.findPhoneNumberByName(similarContacts[0])
                    if (phoneNumber.isNotEmpty()) {
                        processEvent(VoiceEvent.ExactContactFound(similarContacts[0], phoneNumber))
                    } else {
                        processEvent(VoiceEvent.NoContactFound(originalName))
                    }
                } else {
                    // Multiple similar contacts - need confirmation
                    processEvent(VoiceEvent.SimilarContactsFound(originalName, similarContacts))
                }
            }
        })
    }

    /**
     * Gửi SMS
     */
    private fun sendSMSAsync(phoneNumber: String, message: String, contactName: String) {
        smsAutomation.sendSMS(phoneNumber, message, object : SMSAutomation.SMSCallback {
            override fun onSuccess() {
                processEvent(VoiceEvent.SMSSentSuccessfully)
            }

            override fun onError(error: String) {
                processEvent(VoiceEvent.SMSSendFailed(error))
            }
        })
    }

    /**
     * Kiểm tra xem user có xác nhận positive không
     */
    private fun isConfirmationPositive(text: String): Boolean {
        val lowerText = text.lowercase()
        return lowerText.contains("có") ||
                lowerText.contains("đúng") ||
                lowerText.contains("yes") ||
                lowerText.contains("ừ") ||
                lowerText.contains("được") ||
                lowerText.contains("ok")
    }

    /**
     * Kiểm tra xem user có từ chối không
     */
    private fun isDeclined(text: String): Boolean {
        val lowerText = text.lowercase()
        return lowerText.contains("không") ||
                lowerText.contains("không phải") ||
                lowerText.contains("sai") ||
                lowerText.contains("no") ||
                lowerText.contains("thôi") ||
                lowerText.contains("hủy")
    }

    /**
     * Cleanup resources
     */
    fun cleanup() {
        reset()
        currentReceiver = ""
        currentMessage = ""
        similarContacts = emptyList()
        commandProcessor.release()
    }
}