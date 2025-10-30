package com.auto_fe.auto_fe.usecase

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.automation.msg.WAAutomation
import com.auto_fe.auto_fe.domain.VoiceEvent
import com.auto_fe.auto_fe.domain.VoiceState
import com.auto_fe.auto_fe.domain.VoiceStateMachine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay

/**
 * State Machine cho luồng gửi WhatsApp tự động
 * Xử lý toàn bộ flow từ nhận lệnh -> xác nhận -> tìm contact -> gửi WhatsApp
 */
class SendWAStateMachine(
    private val context: Context,
    private val voiceManager: VoiceManager,
    private val waAutomation: WAAutomation
) : VoiceStateMachine() {

    private val coroutineScope = CoroutineScope(Dispatchers.Main)

    companion object {
        private const val TAG = "SendWAStateMachine"
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
                    is VoiceEvent.StartWACommand -> VoiceState.ConfirmingWACommand(currentReceiver, currentMessage)
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.ConfirmingWACommand -> {
                when (event) {
                    is VoiceEvent.WAConfirmed -> VoiceState.ExecutingWACommand
                    is VoiceEvent.WACancelled -> VoiceState.Idle
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.ExecutingWACommand -> {
                when (event) {
                    is VoiceEvent.WASentSuccessfully -> VoiceState.Success
                    is VoiceEvent.WASendFailed -> VoiceState.Error(event.error)
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Success -> {
                when (event) {
                    is VoiceEvent.WASentSuccessfully -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Error -> {
                when (event) {
                    is VoiceEvent.WASendFailed -> VoiceState.Idle
                    else -> null
                }
            }
            else -> null
        }
    }

    override fun onEnterState(state: VoiceState, event: VoiceEvent) {
        when (state) {
            is VoiceState.ConfirmingWACommand -> {
                Log.d(TAG, "Asking for WA confirmation: $currentReceiver -> $currentMessage")
                askForConfirmation()
            }
            is VoiceState.ExecutingWACommand -> {
                Log.d(TAG, "Executing WA command: $currentReceiver -> $currentMessage")
                executeWA()
            }

            is VoiceState.Success -> {
                speak("Đã gửi tin nhắn WhatsApp thành công!")
                // Auto transition to Idle after 2 seconds
                coroutineScope.launch {
                    delay(2000)
                    processEvent(VoiceEvent.WASentSuccessfully)
                }
            }

            is VoiceState.Error -> {
                speak("Có lỗi xảy ra khi gửi tin nhắn WhatsApp.")
                // Auto transition to Idle after 3 seconds
                coroutineScope.launch {
                    delay(3000)
                    processEvent(VoiceEvent.WASendFailed("Unknown error"))
                }
            }

            is VoiceState.Idle -> {
                if (event is VoiceEvent.UserCancelled || event is VoiceEvent.WACancelled) {
                    speak("Đã hủy lệnh gửi tin nhắn WhatsApp.")
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
     * Được gọi từ CommandProcessor để thực hiện WA command
     */
    fun executeWACommand(receiver: String, message: String) {
        Log.d(TAG, "WA command received from CommandProcessor: $receiver -> $message")
        currentReceiver = receiver
        currentMessage = message
        // Chuyển trực tiếp sang ConfirmingWACommand state
        processEvent(VoiceEvent.StartWACommand)
    }

    private fun askForConfirmation() {
        Log.d(TAG, "Asking for WA confirmation")
        
        val confirmText = "Có phải bạn muốn tôi nhắn tin WhatsApp cho $currentReceiver là \"$currentMessage\"?"
        
        // Nói câu hỏi xác nhận và lắng nghe phản hồi
        voiceManager.textToSpeech(confirmText, 1, object : VoiceManager.VoiceControllerCallback {
            override fun onSpeechResult(spokenText: String) {
                Log.d(TAG, "Confirmation response: $spokenText")
                
                // Kiểm tra phản hồi xác nhận
                val confirmed = isConfirmationPositive(spokenText)
                
                if (confirmed) {
                    Log.d(TAG, "User confirmed WA")
                    processEvent(VoiceEvent.WAConfirmed)
                } else {
                    Log.d(TAG, "User cancelled WA")
                    processEvent(VoiceEvent.WACancelled)
                }
            }

            override fun onConfirmationResult(confirmed: Boolean) {}

            override fun onError(error: String) {
                Log.e(TAG, "Confirmation error: $error")
                // Nếu lỗi, coi như từ chối
                processEvent(VoiceEvent.WACancelled)
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

    private fun executeWA() {
        Log.d(TAG, "Executing WA")
        
        waAutomation.sendWA(currentReceiver, currentMessage, object : WAAutomation.WACallback {
            override fun onSuccess() {
                Log.d(TAG, "WA sent successfully")
                processEvent(VoiceEvent.WASentSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "WA send failed: $error")
                processEvent(VoiceEvent.WASendFailed(error))
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
