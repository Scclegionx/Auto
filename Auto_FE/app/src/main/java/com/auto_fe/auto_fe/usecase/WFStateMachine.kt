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

class WFStateMachine(
    private val context: Context,
    private val voiceManager: VoiceManager,
    private val controlDeviceAutomation: ControlDeviceAutomation
) : VoiceStateMachine() {

    private val coroutineScope = CoroutineScope(Dispatchers.Main)

    companion object {
        private const val TAG = "WFStateMachine"
    }

    override fun getNextState(currentState: VoiceState, event: VoiceEvent): VoiceState? {
        return when (currentState) {
            is VoiceState.Idle -> {
                when (event) {
                    is VoiceEvent.StartWifiCommand -> VoiceState.ExecutingWifiCommand
                    else -> null
                }
            }
            is VoiceState.ExecutingWifiCommand -> {
                when (event) {
                    is VoiceEvent.WifiToggledSuccessfully -> VoiceState.Success
                    is VoiceEvent.WifiToggleFailed -> VoiceState.Error(event.error)
                    is VoiceEvent.UserCancelled -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Success -> {
                when (event) {
                    is VoiceEvent.WifiToggledSuccessfully -> VoiceState.Idle
                    else -> null
                }
            }
            is VoiceState.Error -> {
                when (event) {
                    is VoiceEvent.WifiToggleFailed -> VoiceState.Idle
                    else -> null
                }
            }
            else -> null
        }
    }

    override fun onEnterState(state: VoiceState, event: VoiceEvent) {
        when (state) {
            is VoiceState.ExecutingWifiCommand -> {
                Log.d(TAG, "Executing WiFi command")
                // Logic sẽ được thực hiện trong executeWifiCommand()
            }

            is VoiceState.Success -> {
                speak("Đã thực hiện lệnh WiFi thành công!")
                // Auto transition to Idle after 2 seconds
                coroutineScope.launch {
                    delay(2000)
                    processEvent(VoiceEvent.WifiToggledSuccessfully)
                }
            }

            is VoiceState.Error -> {
                speak("Có lỗi xảy ra khi thực hiện lệnh WiFi.")
                // Auto transition to Idle after 3 seconds
                coroutineScope.launch {
                    delay(3000)
                    processEvent(VoiceEvent.WifiToggleFailed("Unknown error"))
                }
            }

            is VoiceState.Idle -> {
                if (event is VoiceEvent.UserCancelled) {
                    speak("Đã hủy lệnh WiFi.")
                }
                voiceManager.resetBusyState()
            }
            else -> {
                // Handle other states if needed
            }
        }
    }

    /**
     * Thực hiện lệnh WiFi được gọi từ CommandProcessor
     */
    fun executeWifiCommand(action: String) {
        Log.d(TAG, "Executing WiFi command: $action")
        processEvent(VoiceEvent.StartWifiCommand)
        
        when (action.lowercase()) {
            "on", "bật" -> {
                executeEnableWifi()
            }
            "off", "tắt" -> {
                executeDisableWifi()
            }
            else -> {
                Log.e(TAG, "Unknown WiFi action: $action")
                processEvent(VoiceEvent.WifiToggleFailed("Hành động không được hỗ trợ: $action"))
            }
        }
    }

    private fun executeEnableWifi() {
        Log.d(TAG, "Executing enable WiFi")
        
        controlDeviceAutomation.enableWifi(object : ControlDeviceAutomation.DeviceCallback {
            override fun onSuccess() {
                Log.d(TAG, "WiFi enabled successfully")
                processEvent(VoiceEvent.WifiToggledSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "WiFi enable failed: $error")
                processEvent(VoiceEvent.WifiToggleFailed(error))
            }
        })
    }

    private fun executeDisableWifi() {
        Log.d(TAG, "Executing disable WiFi")
        
        controlDeviceAutomation.disableWifi(object : ControlDeviceAutomation.DeviceCallback {
            override fun onSuccess() {
                Log.d(TAG, "WiFi disabled successfully")
                processEvent(VoiceEvent.WifiToggledSuccessfully)
            }

            override fun onError(error: String) {
                Log.e(TAG, "WiFi disable failed: $error")
                processEvent(VoiceEvent.WifiToggleFailed(error))
            }
        })
    }


    private fun speak(text: String) {
        voiceManager.speak(text)
    }

    fun cleanup() {
        // Cleanup resources if needed
        Log.d(TAG, "WFStateMachine cleanup completed")
    }
}