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

class VolumnStateMachine(
    private val context: Context,
    private val voiceManager: VoiceManager,
    private val controlDeviceAutomation: ControlDeviceAutomation
) : VoiceStateMachine() {

    private val coroutineScope = CoroutineScope(Dispatchers.Main)

    companion object {
        private const val TAG = "VolumnStateMachine"
    }

    override fun getNextState(currentState: VoiceState, event: VoiceEvent): VoiceState? {
        return when (currentState) {
            is VoiceState.Idle -> {
                when (event) {
                    is VoiceEvent.StartVolumeCommand -> VoiceState.ExecutingVolumeCommand
                    else -> null
                }
            }
            is VoiceState.ExecutingVolumeCommand -> {
                when (event) {
                    is VoiceEvent.VolumeAdjustedSuccessfully -> VoiceState.Success
                    is VoiceEvent.VolumeAdjustmentFailed -> VoiceState.Error(event.error)
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Success -> {
                when (event) {
                    is VoiceEvent.VolumeAdjustedSuccessfully -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Error -> {
                when (event) {
                    is VoiceEvent.VolumeAdjustmentFailed -> VoiceState.Idle
                    else -> null
                }
            }
            else -> null
        }
    }

    override fun onEnterState(state: VoiceState, event: VoiceEvent) {
        when (state) {
            is VoiceState.ExecutingVolumeCommand -> {
                Log.d(TAG, "Executing volume command")
                // Logic sẽ được thực hiện trong executeVolumeCommand()
            }

            is VoiceState.Success -> {
                speak("Đã thực hiện lệnh âm lượng thành công!")
                // Auto transition to Idle after 2 seconds
                coroutineScope.launch {
                    delay(2000)
                    processEvent(VoiceEvent.VolumeAdjustedSuccessfully)
                }
            }

            is VoiceState.Error -> {
                speak("Có lỗi xảy ra khi thực hiện lệnh âm lượng.")
                // Auto transition to Idle after 3 seconds
                coroutineScope.launch {
                    delay(3000)
                    processEvent(VoiceEvent.VolumeAdjustmentFailed("Unknown error"))
                }
            }

            is VoiceState.Idle -> {
                if (event is VoiceEvent.UserCancelled) {
                    speak("Đã hủy lệnh âm lượng.")
                }
                voiceManager.resetBusyState()
            }
            else -> {
                // Handle other states if needed
            }
        }
    }

    /**
     * Thực hiện lệnh volume được gọi từ CommandProcessor
     */
    fun executeVolumeCommand(action: String) {
        Log.d(TAG, "Executing volume command: $action")
        processEvent(VoiceEvent.StartVolumeCommand)
        
        when (action.lowercase()) {
            "+", "tăng" -> {
                executeIncreaseVolume()
            }
            "-", "giảm" -> {
                executeDecreaseVolume()
            }
            else -> {
                Log.e(TAG, "Unknown volume action: $action")
                processEvent(VoiceEvent.VolumeAdjustmentFailed("Hành động không được hỗ trợ: $action"))
            }
        }
    }

    private fun executeIncreaseVolume() {
        Log.d(TAG, "Executing increase volume")
        
        controlDeviceAutomation.increaseVolume(object : ControlDeviceAutomation.DeviceCallback {
            override fun onSuccess() {
                Log.d(TAG, "Volume increased successfully")
                processEvent(VoiceEvent.VolumeAdjustedSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "Volume increase failed: $error")
                processEvent(VoiceEvent.VolumeAdjustmentFailed(error))
            }
        })
    }

    private fun executeDecreaseVolume() {
        Log.d(TAG, "Executing decrease volume")
        
        controlDeviceAutomation.decreaseVolume(object : ControlDeviceAutomation.DeviceCallback {
            override fun onSuccess() {
                Log.d(TAG, "Volume decreased successfully")
                processEvent(VoiceEvent.VolumeAdjustedSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "Volume decrease failed: $error")
                processEvent(VoiceEvent.VolumeAdjustmentFailed(error))
            }
        })
    }


    private fun speak(text: String) {
        voiceManager.speak(text)
    }

    fun cleanup() {
        // Cleanup resources if needed
        Log.d(TAG, "VolumnStateMachine cleanup completed")
    }
}