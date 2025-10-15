package com.auto_fe.auto_fe.usecase

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.automation.msg.SMSAutomation
import com.auto_fe.auto_fe.domain.VoiceEvent
import com.auto_fe.auto_fe.domain.VoiceState
import com.auto_fe.auto_fe.domain.VoiceStateMachine

/**
 * State Machine cho luồng gửi SMS tự động
 * Xử lý toàn bộ flow từ nhận lệnh -> xác nhận -> tìm contact -> gửi SMS
 */
class SendSMSStateMachine(
    private val context: Context,
    private val voiceManager: VoiceManager,
    private val smsAutomation: SMSAutomation
) : VoiceStateMachine() {

    companion object {
        private const val TAG = "SendSMSStateMachine"
        private const val MAX_RETRY_COUNT = 2
    }

    // Lưu context data trong quá trình xử lý
    private var currentReceiver: String = ""
    private var currentMessage: String = ""
    private var similarContacts: List<String> = emptyList()

    /**
     * Định nghĩa State Transitions
     */
    override fun getNextState(currentState: VoiceState, event: VoiceEvent): VoiceState? {
        return when (currentState) {

            // IDLE -> Start listening
            is VoiceState.Idle -> {
                when (event) {
                    is VoiceEvent.StartRecording -> VoiceState.ListeningForSMSCommand
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
                        VoiceState.Error("Tôi chưa hiểu yêu cầu của bạn, vui lòng thử lại.")
                    }
                    is VoiceEvent.Timeout -> {
                        VoiceState.Error("Không nhận được phản hồi.")
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
                // Audio level callback - có thể dùng để hiển thị visual feedback
                // Hiện tại không cần xử lý gì đặc biệt
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
    }

    /**
     * Parse SMS command
     */
    private fun parseCommandAsync(command: String) {
        try {
            val regex = """nhắn tin cho (.+?) là (.+)""".toRegex(RegexOption.IGNORE_CASE)
            val matchResult = regex.find(command)

            if (matchResult != null) {
                val receiver = matchResult.groupValues[1].trim()
                val message = matchResult.groupValues[2].trim()
                processEvent(VoiceEvent.SMSCommandParsed(receiver, message))
            } else {
                processEvent(VoiceEvent.SMSCommandParseFailed("Tôi chưa hiểu yêu cầu của bạn, vui lòng thử lại."))
            }
        } catch (e: Exception) {
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
    }
}