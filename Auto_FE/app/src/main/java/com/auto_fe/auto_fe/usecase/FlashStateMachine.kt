package com.auto_fe.auto_fe.usecase

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.automation.device.ControlDeviceAutomation
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.domain.VoiceEvent
import com.auto_fe.auto_fe.domain.VoiceState
import com.auto_fe.auto_fe.domain.VoiceStateMachine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay

class FlashStateMachine(
    private val context: Context,
    private val voiceManager: VoiceManager,
    private val controlDeviceAutomation: ControlDeviceAutomation
) : VoiceStateMachine() {

    private val coroutineScope = CoroutineScope(Dispatchers.Main)

    companion object {
        private const val TAG = "FlashStateMachine"
    }

    override fun getNextState(currentState: VoiceState, event: VoiceEvent): VoiceState? {
        return when (currentState) {
            is VoiceState.Idle -> {
                when (event) {
                    is VoiceEvent.StartFlashCommand -> VoiceState.ExecutingFlashCommand
                    else -> null
                }
            }
            is VoiceState.ExecutingFlashCommand -> {
                when (event) {
                    is VoiceEvent.FlashToggledSuccessfully -> VoiceState.Success
                    is VoiceEvent.FlashToggleFailed -> VoiceState.Error(event.error)
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Success -> {
                when (event) {
                    is VoiceEvent.FlashToggledSuccessfully -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Error -> {
                when (event) {
                    is VoiceEvent.FlashToggleFailed -> VoiceState.Idle
                    else -> null
                }
            }
            else -> null
        }
    }

    override fun onEnterState(state: VoiceState, event: VoiceEvent) {
        when (state) {
            is VoiceState.ExecutingFlashCommand -> {
                Log.d(TAG, "Executing flash command")
                // Logic sẽ được thực hiện trong executeFlashCommand()
            }

            is VoiceState.Success -> {
                speak("Đã thực hiện lệnh đèn flash thành công!")
                // Auto transition to Idle after 2 seconds
                coroutineScope.launch {
                    delay(2000)
                    processEvent(VoiceEvent.FlashToggledSuccessfully)
                }
            }

            is VoiceState.Error -> {
                speak("Có lỗi xảy ra khi thực hiện lệnh đèn flash.")
                // Auto transition to Idle after 3 seconds
                coroutineScope.launch {
                    delay(3000)
                    processEvent(VoiceEvent.FlashToggleFailed("Unknown error"))
                }
            }

            is VoiceState.Idle -> {
                if (event is VoiceEvent.UserCancelled) {
                    speak("Đã hủy lệnh đèn flash.")
                }
                voiceManager.resetBusyState()
            }
            else -> {
                // Handle other states if needed
            }
        }
    }

    /**
     * Thực hiện lệnh flash được gọi từ CommandProcessor
     */
    fun executeFlashCommand(action: String) {
        Log.d(TAG, "Executing flash command: $action")
        processEvent(VoiceEvent.StartFlashCommand)
        
        when (action.lowercase()) {
            "on", "bật" -> {
                executeEnableFlash()
            }
            "off", "tắt" -> {
                executeDisableFlash()
            }
            else -> {
                Log.e(TAG, "Unknown flash action: $action")
                processEvent(VoiceEvent.FlashToggleFailed("Hành động không được hỗ trợ: $action"))
            }
        }
    }

    private fun executeEnableFlash() {
        Log.d(TAG, "Executing enable flash")
        
        controlDeviceAutomation.enableFlash(object : ControlDeviceAutomation.DeviceCallback {
            override fun onSuccess() {
                Log.d(TAG, "Flash enabled successfully")
                processEvent(VoiceEvent.FlashToggledSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "Flash enable failed: $error")
                processEvent(VoiceEvent.FlashToggleFailed(error))
            }
        })
    }

    private fun executeDisableFlash() {
        Log.d(TAG, "Executing disable flash")
        
        controlDeviceAutomation.disableFlash(object : ControlDeviceAutomation.DeviceCallback {
            override fun onSuccess() {
                Log.d(TAG, "Flash disabled successfully")
                processEvent(VoiceEvent.FlashToggledSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "Flash disable failed: $error")
                processEvent(VoiceEvent.FlashToggleFailed(error))
            }
        })
    }


    private fun speak(text: String) {
        voiceManager.speak(text)
    }

    fun cleanup() {
        // Cleanup resources if needed
        Log.d(TAG, "FlashStateMachine cleanup completed")
    }
}
