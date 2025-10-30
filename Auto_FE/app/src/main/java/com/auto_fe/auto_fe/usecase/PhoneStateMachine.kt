package com.auto_fe.auto_fe.usecase

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.automation.phone.PhoneAutomation
import com.auto_fe.auto_fe.domain.VoiceEvent
import com.auto_fe.auto_fe.domain.VoiceState
import com.auto_fe.auto_fe.domain.VoiceStateMachine
import com.auto_fe.auto_fe.utils.SettingsManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay

/**
 * State Machine cho luồng gọi điện tự động
 * Xử lý toàn bộ flow từ nhận lệnh -> confirmation -> gọi điện
 */
class PhoneStateMachine(
    private val context: Context,
    private val voiceManager: VoiceManager,
    private val phoneAutomation: PhoneAutomation
) : VoiceStateMachine() {

    companion object {
        private const val TAG = "PhoneStateMachine"
    }

    // Lưu context data trong quá trình xử lý
    private var currentContact: String = ""
    private var currentPlatform: String = "phone"
    
    // Coroutine scope cho delayed operations
    private val coroutineScope = CoroutineScope(Dispatchers.Main)
    
    // Callback cho audio level
    var onAudioLevelChanged: ((Int) -> Unit)? = null
    
    // Callback cho transcript
    var onTranscriptUpdated: ((String) -> Unit)? = null

    override fun getNextState(currentState: VoiceState, event: VoiceEvent): VoiceState? {
        return when (currentState) {
            is VoiceState.Idle -> {
                when (event) {
                    is VoiceEvent.StartPhoneCommand -> VoiceState.ConfirmingPhoneCommand
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
                    is VoiceState.ConfirmingPhoneCommand -> {
                        when (event) {
                            is VoiceEvent.PhoneConfirmed -> VoiceState.ExecutingPhoneCommand
                            is VoiceEvent.PhoneCancelled -> VoiceState.Cancel
                            is VoiceEvent.UserCancelled -> VoiceState.Cancel
                            else -> null
                        }
                    }
            is VoiceState.ExecutingPhoneCommand -> {
                when (event) {
                    is VoiceEvent.CallMadeSuccessfully -> VoiceState.Success
                    is VoiceEvent.CallFailed -> VoiceState.Error(event.error)
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Success -> {
                when (event) {
                    is VoiceEvent.CallMadeSuccessfully -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Error -> {
                when (event) {
                    is VoiceEvent.CallFailed -> VoiceState.Idle
                    else -> null
                }
            }
            else -> null
        }
    }

    override fun onEnterState(state: VoiceState, event: VoiceEvent) {
        when (state) {
            is VoiceState.ConfirmingPhoneCommand -> {
                val isSupportSpeakEnabled = SettingsManager(context).isSupportSpeakEnabled()
                if (isSupportSpeakEnabled) {
                    Log.d(TAG, "Asking for phone confirmation: $currentContact")
                    askForConfirmation()
                } else {
                    Log.d(TAG, "Support speak OFF - skip confirmation and proceed calling: $currentContact")
                    processEvent(VoiceEvent.PhoneConfirmed)
                }
            }
            is VoiceState.ExecutingPhoneCommand -> {
                Log.d(TAG, "Executing phone command: $currentContact")
                executePhoneCall()
            }

            is VoiceState.Success -> {
                val isSupportSpeakEnabled = SettingsManager(context).isSupportSpeakEnabled()
                if (isSupportSpeakEnabled) {
                    speak("Đã gọi điện thành công!")
                } else {
                    speak("Đã tìm thấy liên hệ! Hãy bấm gọi.")
                }
                voiceManager.resetBusyState()
                // Auto transition to Idle after 2 seconds
                coroutineScope.launch {
                    delay(2000)
                    processEvent(VoiceEvent.CallMadeSuccessfully)
                }
            }

            is VoiceState.Cancel -> {
                speak("Đã hủy lệnh gọi điện.")
                voiceManager.resetBusyState()
                // Auto transition to Idle after 2 seconds
                coroutineScope.launch {
                    delay(2000)
                    processEvent(VoiceEvent.PhoneCancelled)
                }
            }

            is VoiceState.Error -> {
                speak("Có lỗi xảy ra khi gọi điện.")
                voiceManager.resetBusyState()
                // Auto transition to Idle after 3 seconds
                coroutineScope.launch {
                    delay(3000)
                    processEvent(VoiceEvent.CallFailed("Unknown error"))
                }
            }

            is VoiceState.Idle -> {
                if (event is VoiceEvent.UserCancelled || event is VoiceEvent.PhoneCancelled) {
                    speak("Đã hủy lệnh gọi điện.")
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
     * Được gọi từ CommandProcessor để thực hiện phone command
     */
    fun executePhoneCommand(contact: String, platform: String = "phone") {
        Log.d(TAG, "Phone command received from CommandProcessor: $contact, platform: $platform")
        currentContact = contact
        currentPlatform = platform
        // Chuyển trực tiếp sang ConfirmingPhoneCommand state
        processEvent(VoiceEvent.StartPhoneCommand)
    }

    private fun askForConfirmation() {
        Log.d(TAG, "Asking for phone confirmation")
        
        val confirmText = "Có phải bạn muốn tôi gọi điện cho $currentContact?"
        
        // Nói câu hỏi xác nhận và lắng nghe phản hồi
        voiceManager.textToSpeech(confirmText, 1, object : VoiceManager.VoiceControllerCallback {
            override fun onSpeechResult(spokenText: String) {
                Log.d(TAG, "Confirmation response: $spokenText")
                
                // Kiểm tra phản hồi xác nhận
                val confirmed = isConfirmationPositive(spokenText)
                
                if (confirmed) {
                    Log.d(TAG, "User confirmed phone call")
                    processEvent(VoiceEvent.PhoneConfirmed)
                } else {
                    Log.d(TAG, "User cancelled phone call")
                    processEvent(VoiceEvent.PhoneCancelled)
                }
            }

            override fun onConfirmationResult(confirmed: Boolean) {}

            override fun onError(error: String) {
                Log.e(TAG, "Confirmation error: $error")
                // Nếu lỗi, coi như từ chối
                processEvent(VoiceEvent.PhoneCancelled)
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

    private fun executePhoneCall() {
        Log.d(TAG, "Executing phone call")
        
        // Kiểm tra permission trước khi gọi
        Log.d(TAG, "Checking permission for phone call...")
        phoneAutomation.checkAndRequestPermission(object : PhoneAutomation.PhoneCallback {
            override fun onSuccess() {
                // Permission đã có, thực hiện gọi điện
                Log.d(TAG, "Permission granted, calling makeCallDirectly()")
                makeCallDirectly()
            }

            override fun onError(error: String) {
                Log.e(TAG, "Permission check failed: $error")
                processEvent(VoiceEvent.CallFailed(error))
            }

            override fun onPermissionRequired() {
                Log.d(TAG, "Permission required for phone call")
                speak("Cần cấp quyền gọi điện để thực hiện lệnh này")
                processEvent(VoiceEvent.CallFailed("Cần cấp quyền gọi điện"))
            }
        })
    }
    
    private fun makeCallDirectly() {
        Log.d(TAG, "makeCallDirectly() called with contact: $currentContact, platform: $currentPlatform")
        phoneAutomation.makeCall(currentContact, currentPlatform, object : PhoneAutomation.PhoneCallback {
            override fun onSuccess() {
                Log.d(TAG, "Phone call made successfully")
                processEvent(VoiceEvent.CallMadeSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "Phone call failed: $error")
                processEvent(VoiceEvent.CallFailed(error))
            }

            override fun onPermissionRequired() {
                // Không cần xử lý ở đây vì đã check trước đó
            }
        })
    }

    private fun speak(text: String) {
        voiceManager.speak(text)
    }

    private fun resetContext() {
        currentContact = ""
        currentPlatform = "phone"
    }

    fun cleanup() {
        coroutineScope.launch {
            // Cleanup if needed
        }
    }
}